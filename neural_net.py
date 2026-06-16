import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

np.random.seed(42)

# ----------------------------------------------------------------
# PART 1: Manual forward pass — see exactly what's happening
# inside a tiny 2-layer network
# ----------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def forward_pass(x, W1, b1, W2, b2):
    """
    Two-layer network: input → hidden (ReLU) → output (sigmoid)
    x:  input vector (n_features,)
    W1: weights layer 1 (n_hidden, n_features)
    b1: biases layer 1  (n_hidden,)
    W2: weights layer 2 (1, n_hidden)
    b2: bias layer 2    (1,)
    """
    z1 = W1 @ x + b1       # linear combination, hidden layer
    a1 = relu(z1)           # ReLU activation
    z2 = W2 @ a1 + b2       # linear combination, output layer
    a2 = sigmoid(z2)        # sigmoid (output is a probability)
    return a1, a2
# Random small network
n_features, n_hidden = 3, 4
W1 = np.random.randn(n_hidden, n_features) * 0.1
b1 = np.zeros(n_hidden)
W2 = np.random.randn(1, n_hidden) * 0.1
b2 = np.zeros(1)
x_example = np.array([0.8, -0.3, 1.2])   # one observation
a1, a2 = forward_pass(x_example, W1, b1, W2, b2)
print("=== Manual Forward Pass ===")
print(f"Input:              {x_example}")
print(f"Hidden activations: {a1.round(4)}")
print(f"Output (prob):      {a2[0]:.4f}")
print(f"\nW1 shape: {W1.shape}  →  maps {n_features} inputs"
      f" to {n_hidden} hidden neurons")
print(f"W2 shape: {W2.shape}  →  maps {n_hidden} hidden"
      f" neurons to 1 output")
print(f"Total parameters:   "
      f"{W1.size + b1.size + W2.size + b2.size}")

# ----------------------------------------------------------------
# PART 2: Energy load forecasting with sklearn MLP
# Compare against gradient boosting from Block 3
# ----------------------------------------------------------------
# Rebuild the feature matrix from Block 3 (without real API data)
# Simulate realistic demand data with multiple seasonalities
T     = 8760
hours = np.arange(T)
# Multiple seasonalities: daily + annual + weekly
daily_pattern  = 8000 * np.sin(2*np.pi*(hours % 24 - 6)/24)
annual_pattern = 5000 * np.sin(2*np.pi*(hours/8760 - 0.25))
weekly_pattern = -3000 * ((hours // 24) % 7 >= 5).astype(float)
noise          = np.random.normal(0, 1500, T)
demand         = 50000 + daily_pattern + annual_pattern \
                 + weekly_pattern + noise
# Feature engineering — same as Block 3
X = np.column_stack([
    np.sin(2*np.pi*(hours % 24)/24),          # daily sin
    np.cos(2*np.pi*(hours % 24)/24),          # daily cos
    np.sin(2*np.pi*(hours/8760)),              # annual sin
    np.cos(2*np.pi*(hours/8760)),              # annual cos
    np.sin(2*np.pi*((hours//24) % 7)/7),      # weekly sin
    np.cos(2*np.pi*((hours//24) % 7)/7),      # weekly cos
    ((hours//24) % 7 >= 5).astype(float),     # is_weekend
    np.roll(demand, 1),                        # lag 1
    np.roll(demand, 24),                       # lag 24
    np.roll(demand, 168),                      # lag 168
])
# Drop first 168 rows (lag warmup period)
X = X[168:]
y = demand[168:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False   # no shuffle — time series
)
scaler   = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
# MLP — note scaling is REQUIRED for neural networks
# (gradient descent is sensitive to feature scale)
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # two hidden layers
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,           # hold out 10% for validation
    validation_fraction=0.1,
    n_iter_no_change=20,           # stop if no improvement for 20 epochs
    random_state=42,
    verbose=False
)
mlp.fit(X_train_sc, y_train)
y_pred_mlp = mlp.predict(X_test_sc)
mae_mlp  = mean_absolute_error(y_test, y_pred_mlp)
mape_mlp = np.abs((y_test - y_pred_mlp)/y_test).mean() * 100
print(f"\n=== MLP Load Forecasting ===")
print(f"Architecture:   input(10) → 64 → 32 → output(1)")
print(f"Training stopped at epoch: {mlp.n_iter_}")
print(f"MAE:   {mae_mlp:,.0f} MWh")
print(f"MAPE:  {mape_mlp:.2f}%")

# ----------------------------------------------------------------
# PART 3: Activation functions — visualize what they do
# ----------------------------------------------------------------
z = np.linspace(-4, 4, 300)
activations = {
    'Sigmoid':      sigmoid(z),
    'Tanh':         np.tanh(z),
    'ReLU':         relu(z),
    'Leaky ReLU':   np.where(z > 0, z, 0.01 * z),
}
gradients = {
    'Sigmoid':      sigmoid(z) * (1 - sigmoid(z)),
    'Tanh':         1 - np.tanh(z)**2,
    'ReLU':         (z > 0).astype(float),
    'Leaky ReLU':   np.where(z > 0, 1.0, 0.01),
}
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Activation functions
for (name, vals), col in zip(activations.items(), colors):
    axes[0].plot(z, vals, label=name, color=col, lw=2)
axes[0].set_title('Activation Functions')
axes[0].set_xlabel('z')
axes[0].set_ylabel('φ(z)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].axvline(0, color='black', lw=0.5)
# Gradients
for (name, vals), col in zip(gradients.items(), colors):
    axes[1].plot(z, vals, label=name, color=col, lw=2)
axes[1].set_title('Activation Gradients\n(vanishing gradient visible)')
axes[1].set_xlabel('z')
axes[1].set_ylabel("φ'(z)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='black', lw=0.5)
# Training loss curve
axes[2].plot(mlp.loss_curve_, color='steelblue',
             lw=2, label='Training loss')
if hasattr(mlp, 'validation_scores_'):
    axes[2].plot(mlp.validation_scores_, color='tomato',
                 lw=2, label='Validation score')
axes[2].set_title('Training Loss Curve')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neural_networks.png', dpi=150, bbox_inches='tight')
plt.show()