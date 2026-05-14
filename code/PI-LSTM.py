import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
from collections import defaultdict

DATA_DIR = r"D:\tyh\outputdata"
SAVE_WEIGHTS_PATH = r"D:\tyh\result\compare\best_pilstm_weights_final.weights.h5"
SAVE_DYN_SCALER = r"D:\tyh\result\compare\dyn_scaler_final.pkl"
SAVE_STAT_SCALER = r"D:\tyh\result\compare\stat_scaler_final.pkl"

DO_TRAIN = True

VAL_CASES = [9, 10]
TEST_CASES = [1, 18]
TRAIN_CASES = [i for i in range(1, 19) if i not in TEST_CASES + VAL_CASES]

SEQ_LEN = 3
BATCH_SIZE = 128
EPOCHS = 80
PATIENCE = 15
LR = 1e-3
LAMBDA_PDE = 0.05
SCALE_MOMENTUM = 1e-8

tf.random.set_seed(42)
np.random.seed(42)

def build_sequences_by_case(cases, dataset_name="Set"):
    print(f"📦 Loading {dataset_name} (Cases: {cases})...")
    seq_inputs, static_inputs, targets, meta_info = [], [], [], []
    dropped_count = 0

    for cid in cases:
        file_path = os.path.join(DATA_DIR, f"FlowField_Case{cid}.csv")
        if not os.path.exists(file_path): continue

        df = pd.read_csv(file_path)
        current_h = float(df['h'].iloc[0])

        dynamic_p_min = -105000.0
        dynamic_p_max = 4 * (1000.0 * 9.81 * current_h)

        X_static = df[['h', 'S', 'H', 'X', 'Y', 'Z']].values.astype(np.float32)
        Y_dynamic = df[['P', 'U', 'V', 'W', 'Alpha']].values.astype(np.float32)
        T_full = df['T'].values.astype(np.float32)

        group_dict = defaultdict(list)
        for i, row in enumerate(X_static):
            group_dict[tuple(row)].append(i)

        for key, indices in group_dict.items():
            if len(indices) < SEQ_LEN: continue
            indices = sorted(indices, key=lambda idx: T_full[idx])

            for start in range(len(indices) - SEQ_LEN + 1):
                window = indices[start:start + SEQ_LEN]
                curr_idx = window[-1]

                p_window = Y_dynamic[window, 0]
                alpha_window = Y_dynamic[window, 4]

                if np.any(p_window < dynamic_p_min) or np.any(p_window > dynamic_p_max):
                    dropped_count += 1
                    continue

                if np.any((alpha_window < 0.05) & (p_window > 10000.0)):
                    dropped_count += 1
                    continue

                t_seq = T_full[window[:-1]].reshape(-1, 1).astype(np.float32)
                hist_seq = Y_dynamic[window[:-1]].astype(np.float32)
                seq_inputs.append(np.hstack([t_seq, hist_seq]))

                stat_with_t = np.hstack([X_static[curr_idx], [T_full[curr_idx]]])
                static_inputs.append(stat_with_t)

                targets.append(Y_dynamic[curr_idx])

    print(f"   ✨ Dropped dirty sequences: {dropped_count}")
    return np.array(seq_inputs), np.array(static_inputs), np.array(targets)

X_seq_train, X_stat_train, Y_train = build_sequences_by_case(TRAIN_CASES, "Train Set")
X_seq_val, X_stat_val, Y_val = build_sequences_by_case(VAL_CASES, "Validation Set")

print("⚖️ Fitting RobustScaler and saving to disk...")
dyn_scaler = RobustScaler().fit(Y_train[:, 0:4])
stat_scaler = RobustScaler().fit(X_stat_train)

joblib.dump(dyn_scaler, SAVE_DYN_SCALER)
joblib.dump(stat_scaler, SAVE_STAT_SCALER)
print(f"   -> DYN_SCALER saved to: {SAVE_DYN_SCALER}")
print(f"   -> STAT_SCALER saved to: {SAVE_STAT_SCALER}")

std_P = tf.constant(dyn_scaler.scale_[0], dtype=tf.float32)
std_U = tf.constant(dyn_scaler.scale_[1], dtype=tf.float32)
std_V = tf.constant(dyn_scaler.scale_[2], dtype=tf.float32)
std_W = tf.constant(dyn_scaler.scale_[3], dtype=tf.float32)

mean_P = tf.constant(dyn_scaler.center_[0], dtype=tf.float32)
mean_U = tf.constant(dyn_scaler.center_[1], dtype=tf.float32)
mean_V = tf.constant(dyn_scaler.center_[2], dtype=tf.float32)
mean_W = tf.constant(dyn_scaler.center_[3], dtype=tf.float32)

std_X = tf.constant(stat_scaler.scale_[3], dtype=tf.float32)
std_Y = tf.constant(stat_scaler.scale_[4], dtype=tf.float32)
std_Z = tf.constant(stat_scaler.scale_[5], dtype=tf.float32)
std_T = tf.constant(stat_scaler.scale_[6], dtype=tf.float32)

def scale_dynamic(seq, y):
    seq_scaled = seq.copy()
    seq_scaled[:, :, 1:5] = dyn_scaler.transform(seq[:, :, 1:5].reshape(-1, 4)).reshape(seq.shape[0], seq.shape[1], 4)
    y_scaled = y.copy()
    y_scaled[:, 0:4] = dyn_scaler.transform(y[:, 0:4])
    return seq_scaled, y_scaled

X_seq_train_sc, Y_train_sc = scale_dynamic(X_seq_train, Y_train)
X_seq_val_sc, Y_val_sc = scale_dynamic(X_seq_val, Y_val)

X_stat_train_sc = stat_scaler.transform(X_stat_train)
X_stat_val_sc = stat_scaler.transform(X_stat_val)

dataset_train = tf.data.Dataset.from_tensor_slices((X_seq_train_sc, X_stat_train_sc, Y_train_sc)).shuffle(10000).batch(BATCH_SIZE)
dataset_val = tf.data.Dataset.from_tensor_slices((X_seq_val_sc, X_stat_val_sc, Y_val_sc)).batch(BATCH_SIZE)

class Multiphase_PILSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, activation='tanh')
        self.dense_shared = tf.keras.layers.Dense(64, activation='relu')
        self.dense_uvw = tf.keras.layers.Dense(64, activation='relu')
        self.out_uvw = tf.keras.layers.Dense(3, activation='linear')
        self.out_alpha = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense_p1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_p2 = tf.keras.layers.Dense(64, activation='relu')
        self.out_p = tf.keras.layers.Dense(1, activation='linear')

    def call(self, seq_in, static_in):
        h_scaled = static_in[:, 0:1]
        z_scaled = static_in[:, 5:6]
        z_rel_feature = h_scaled * z_scaled
        x = self.lstm(seq_in)
        x = tf.concat([x, static_in, z_rel_feature], axis=1)
        x = self.dense_shared(x)
        uvw_branch = self.dense_uvw(x)
        uvw = self.out_uvw(uvw_branch)
        alpha = self.out_alpha(uvw_branch)
        p = self.out_p(self.dense_p2(self.dense_p1(x)))
        return tf.concat([p, uvw, alpha], axis=1)

model = Multiphase_PILSTM()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def compute_physics_loss(model, seq_in, static_in):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(static_in)
        pred = model(seq_in, static_in)
        p_tot_scaled = pred[:, 0:1]
        u = pred[:, 1:2]
        v = pred[:, 2:3]
        w = pred[:, 3:4]
        alpha = pred[:, 4:5]

        u_real = u * std_U + mean_U
        v_real = v * std_V + mean_V
        w_real = w * std_W + mean_W
        p_tot_real = p_tot_scaled * std_P + mean_P
        rho = alpha * 998.2 + (1.0 - alpha) * 1.225

        dyn_p = 0.5 * rho * (tf.square(u_real) + tf.square(v_real) + tf.square(w_real))
        p_stat_real = p_tot_real - dyn_p

    grad_p_stat = tape.gradient(p_stat_real, static_in)
    grad_u = tape.gradient(u, static_in)
    grad_v = tape.gradient(v, static_in)
    grad_w = tape.gradient(w, static_in)
    grad_alpha = tape.gradient(alpha, static_in)
    del tape

    p_x = grad_p_stat[:, 3:4] / std_X
    p_y = grad_p_stat[:, 4:5] / std_Y
    p_z = grad_p_stat[:, 5:6] / std_Z
    u_t = grad_u[:, 6:7] / std_T
    u_x = grad_u[:, 3:4] * (std_U / std_X)
    u_y = grad_u[:, 4:5] * (std_U / std_Y)
    u_z = grad_u[:, 5:6] * (std_U / std_Z)
    v_t = grad_v[:, 6:7] / std_T
    v_x = grad_v[:, 3:4] * (std_V / std_X)
    v_y = grad_v[:, 4:5] * (std_V / std_Y)
    v_z = grad_v[:, 5:6] * (std_V / std_Z)
    w_t = grad_w[:, 6:7] / std_T
    w_x = grad_w[:, 3:4] * (std_W / std_X)
    w_y = grad_w[:, 4:5] * (std_W / std_Y)
    w_z = grad_w[:, 5:6] * (std_W / std_Z)
    alpha_t = grad_alpha[:, 6:7] / std_T
    alpha_x = grad_alpha[:, 3:4] / std_X
    alpha_y = grad_alpha[:, 4:5] / std_Y
    alpha_z = grad_alpha[:, 5:6] / std_Z

    g = 9.81
    f_cont = u_x + v_y + w_z
    f_vof = alpha_t + u_real * alpha_x + v_real * alpha_y + w_real * alpha_z

    f_mom_x = rho * (u_t + u_real * u_x + v_real * u_y + w_real * u_z) + p_x
    f_mom_y = rho * (v_t + u_real * v_x + v_real * v_y + w_real * v_z) + p_y
    f_mom_z = rho * (w_t + u_real * w_x + v_real * w_y + w_real * w_z) + p_z + rho * g

    water_mask = alpha

    loss_pde = (
            tf.reduce_mean(tf.square(f_cont)) +
            tf.reduce_mean(tf.square(f_vof)) +
            SCALE_MOMENTUM * (
                    tf.reduce_mean(water_mask * tf.square(f_mom_x)) +
                    tf.reduce_mean(water_mask * tf.square(f_mom_y)) +
                    tf.reduce_mean(water_mask * tf.square(f_mom_z))
            )
    )
    return loss_pde

@tf.function
def train_step(seq_in, static_in, y_true, current_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(seq_in, static_in)

        abs_p_true = tf.abs(y_true[:, 0:1])
        peak_weight = 1.0 + tf.square(abs_p_true) * 3.0
        loss_p_base = tf.reduce_mean(peak_weight * tf.square(y_pred[:, 0:1] - y_true[:, 0:1]))

        air_mask = 1.0 - y_true[:, 4:5]
        loss_p_air = tf.reduce_mean(air_mask * tf.square(y_pred[:, 0:1] - y_true[:, 0:1]))
        loss_p_total = 4.0 * loss_p_base + 3.0 * loss_p_air

        loss_uvw = tf.reduce_mean(tf.square(y_pred[:, 1:4] - y_true[:, 1:4]))
        interface_weight = 1.0 + 5.0 * tf.exp(-20.0 * tf.square(y_true[:, 4:5] - 0.5))
        loss_alpha = tf.reduce_mean(interface_weight * tf.square(y_pred[:, 4:5] - y_true[:, 4:5]))

        loss_data = loss_p_total + 1.0 * loss_uvw + 3.0 * loss_alpha

        loss_pde = compute_physics_loss(model, seq_in, static_in)
        pde_weight = tf.cond(current_epoch > 10.0, lambda: LAMBDA_PDE * tf.minimum(1.0, (current_epoch - 10.0) / 10.0),
                             lambda: tf.constant(0.0))
        loss_total = loss_data + pde_weight * loss_pde

    grads = tape.gradient(loss_total, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_data, loss_pde, loss_p_total

@tf.function
def val_step(seq_in, static_in, y_true):
    y_pred = model(seq_in, static_in)
    loss_p = tf.reduce_mean(tf.square(y_pred[:, 0:1] - y_true[:, 0:1]))
    loss_uvw = tf.reduce_mean(tf.square(y_pred[:, 1:4] - y_true[:, 1:4]))
    loss_alpha = tf.reduce_mean(tf.square(y_pred[:, 4:5] - y_true[:, 4:5]))
    return loss_p + loss_uvw + loss_alpha

if DO_TRAIN:
    print(f"\n🚀 Starting training...")
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_l_data = 0
        epoch_l_pde = 0
        steps = 0
        tf_epoch = tf.constant(epoch, dtype=tf.float32)

        for batch_seq, batch_stat, batch_y in dataset_train:
            l_data, l_pde, l_p = train_step(batch_seq, batch_stat, batch_y, tf_epoch)
            epoch_l_data += l_data
            epoch_l_pde += l_pde
            steps += 1

        val_loss_total = 0
        val_steps = 0
        for val_seq, val_stat, val_y in dataset_val:
            v_loss = val_step(val_seq, val_stat, val_y)
            val_loss_total += v_loss
            val_steps += 1

        avg_train_loss = epoch_l_data / steps
        avg_val_loss = val_loss_total / val_steps

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | PDE: {epoch_l_pde / steps:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(SAVE_WEIGHTS_PATH)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"🛑 Early stopping triggered! Training finished. Best weights saved to: {SAVE_WEIGHTS_PATH}")
                break