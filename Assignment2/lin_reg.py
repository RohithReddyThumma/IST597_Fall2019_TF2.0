import time, math, string, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt


cfg = dict(
    NUM_EXAMPLES   = 10000,
    TRUE_W         = 3.0,
    TRUE_B         = 2.0,
    n_epochs       = 20,          
    lr_init        = 0.05,        
    loss_name      = "HYBRID",    
    huber_delta    = 1.0,        
    hybrid_alpha   = 0.1,         
    hybrid_beta    = 0.1,         
    batch_size     = 512,         
    patience       = 3,           
    plateau_eps    = 1e-5,        
    init_W         = 0.0,         
    init_B         = 0.0,
    device_mode    = "AUTO",      
    
    data_noise_type= "gaussian",  
    data_noise_std = 1.0,         
    add_weight_noise_every = 1,   
    weight_noise_std = 1e-3,      
    lr_jitter_every = 1,          
    lr_jitter_std   = 0.05,       
    name_for_seed   = "Rohith",   
)

def name_to_seed(name: str) -> int:
    
    name = name.strip()
    acc = 0
    for ch in name:
        if ch in string.ascii_letters + string.digits:
            acc = (acc * 37 + ord(ch)) % (2**31-1)
    return acc if acc > 0 else 1234

SEED = name_to_seed(cfg["name_for_seed"])
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_noise(n, kind, scale):
    if kind == "gaussian":
        return tf.random.normal([n], stddev=scale, seed=SEED)
    if kind == "laplace":
        
        u = tf.random.uniform([n], minval=-0.5, maxval=0.5, seed=SEED)
        return -scale * tf.sign(u) * tf.math.log(1 - 2*tf.abs(u))
    if kind == "uniform":
        return tf.random.uniform([n], minval=-scale, maxval=scale, seed=SEED)
    raise ValueError("Unknown noise type")

# Create syntheticdata
N = cfg["NUM_EXAMPLES"]
X = tf.random.normal([N], dtype=tf.float32, seed=SEED)
eps = make_noise(N, cfg["data_noise_type"], cfg["data_noise_std"])
y  = cfg["TRUE_W"] * X + cfg["TRUE_B"] + eps

# dataset 
ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(4*cfg["batch_size"], seed=SEED).batch(cfg["batch_size"])


# Trainable variables

W = tf.Variable(cfg["init_W"], dtype=tf.float32, name="W")
b = tf.Variable(cfg["init_B"], dtype=tf.float32, name="b")

# Losses

def yhat(x): return W * x + b

def loss_MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def loss_L1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def loss_Huber(y_true, y_pred, delta):
    e = y_true - y_pred
    abs_e = tf.abs(e)
    quad = 0.5 * tf.square(e)
    lin  = delta * (abs_e - 0.5 * delta)
    return tf.reduce_mean(tf.where(abs_e <= delta, quad, lin))

def loss_HYBRID(y_true, y_pred, alpha, beta):
    # alpha * L1 + beta * L2  
    e = y_true - y_pred
    return alpha * tf.reduce_mean(tf.abs(e)) + beta * tf.reduce_mean(tf.square(e))

def get_loss_fn():
    n = cfg["loss_name"].upper()
    if n == "MSE":    return lambda yt, yp: loss_MSE(yt, yp)
    if n == "L1":     return lambda yt, yp: loss_L1(yt, yp)
    if n == "HUBER":  return lambda yt, yp: loss_Huber(yt, yp, cfg["huber_delta"])
    if n == "HYBRID": return lambda yt, yp: loss_HYBRID(yt, yp, cfg["hybrid_alpha"], cfg["hybrid_beta"])
    raise ValueError("loss_name must be one of: MSE, L1, HUBER, HYBRID")

loss_fn = get_loss_fn()

device_str = "/device:CPU:0"
if cfg["device_mode"] == "GPU" and tf.config.list_physical_devices("GPU"):
    device_str = "/device:GPU:0"
elif cfg["device_mode"] in ("AUTO", "GPU") and tf.config.list_physical_devices("GPU"):
    device_str = "/device:GPU:0"
print(f"Using device: {device_str} | Physical: {tf.config.list_physical_devices()}")


lr = tf.Variable(cfg["lr_init"], dtype=tf.float32)
best_loss = float("inf")
epochs_no_improve = 0

epoch_times = []
epoch_losses = []
residuals_all = []  # for histogram

with tf.device(device_str):
    for epoch in range(1, cfg["n_epochs"]+1):
        t0 = time.perf_counter()
        
        if cfg["lr_jitter_every"] and epoch % cfg["lr_jitter_every"] == 0:
            jitter = 1.0 + np.random.normal(0.0, cfg["lr_jitter_std"])
            lr.assign(tf.clip_by_value(lr * jitter, 1e-6, 10.0))

        # add weight noise every N epochs
        if cfg["add_weight_noise_every"] and epoch % cfg["add_weight_noise_every"] == 0 and epoch > 1:
            W.assign_add(tf.random.normal(shape=W.shape, stddev=cfg["weight_noise_std"], seed=SEED))
            b.assign_add(tf.random.normal(shape=b.shape, stddev=cfg["weight_noise_std"], seed=SEED))

        # mini batch GD
        running_loss = 0.0
        count = 0
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                pred = yhat(xb)
                loss = loss_fn(yb, pred)
            dW, db = tape.gradient(loss, [W, b])
            W.assign_sub(lr * dW)
            b.assign_sub(lr * db)
            running_loss += float(loss.numpy()) * xb.shape[0]
            count += int(xb.shape[0])

        epoch_loss = running_loss / max(1, count)
        epoch_losses.append(epoch_loss)

        # collect residuals for analysis
        r = (y - yhat(X)).numpy()
        residuals_all.append(np.std(r))

        # LR scheduling on plateau
        improved = best_loss - epoch_loss > cfg["plateau_eps"]
        if improved:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                lr.assign(lr * 0.5)
                epochs_no_improve = 0

        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)
        print(f"epoch {epoch:02d} | loss={epoch_loss:.6f} | W={W.numpy():.3f} b={b.numpy():.3f} "
              f"| lr={lr.numpy():.5f} | time={epoch_times[-1]:.4f}s")

print("\n=== Training Finished ===")
print(f"True params:  W*={cfg['TRUE_W']:.3f}, b*={cfg['TRUE_B']:.3f}")
print(f"Learned:      Ŵ={W.numpy():.3f}, b̂={b.numpy():.3f}")
print(f"Final loss:   {epoch_losses[-1]:.6f}  (loss={cfg['loss_name']})")
print(f"Avg epoch time on {cfg['device_mode']}: {np.mean(epoch_times):.4f}s "
      f"(median {np.median(epoch_times):.4f}s)")


# Plots
# 1) Scatter + fitted line
plt.figure()
plt.plot(X.numpy(), y.numpy(), 'bo', alpha=0.25, label='data')
x_line = tf.linspace(tf.reduce_min(X), tf.reduce_max(X), 300)
y_line = W.numpy() * x_line.numpy() + b.numpy()
plt.plot(x_line.numpy(), y_line, 'r', label=f'{cfg["loss_name"].lower()} fit')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title(f'Fit: Ŵ={W.numpy():.3f}, b̂={b.numpy():.3f}')
plt.tight_layout(); plt.show()

# 2) Loss curve
plt.figure()
plt.plot(np.arange(1, len(epoch_losses)+1), epoch_losses)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title(f'Loss curve ({cfg["loss_name"]}) with LR scheduling')
plt.tight_layout(); plt.show()

# 3) Residual std over epochs (proxy for robustness)
plt.figure()
plt.plot(np.arange(1, len(residuals_all)+1), residuals_all)
plt.xlabel('Epoch'); plt.ylabel('Std of residuals')
plt.title('Residual dispersion vs epoch')
plt.tight_layout(); plt.show()

# 4) Residual histogram at the end
final_res = (y - yhat(X)).numpy()
plt.figure()
plt.hist(final_res, bins=40)
plt.xlabel('Residual'); plt.ylabel('Frequency')
plt.title('Residuals (final model)')
plt.tight_layout(); plt.show()
