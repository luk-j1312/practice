import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------
# PART 1: Unconstrained optimization
# A simple quadratic to which we know the answer analytically, so we can
# verify scipy is doing the right thing.
# f(x) = (x1 - 2)^2 + (x2 + 3)^2
# Minimum is at (2, -3), f = 0
# ----------------------------------------------------------------
def f_quad(x):
    return (x[0]-2)**2 + (x[1]+3)**2
def grad_quad(x):
    # analytical gradient; scipy can estimate this numerically,
    # but providing it explicitly is faster and more accurate
    return np.array([2*(x[0]-2), 2*(x[1]+3)])
x0 = np.array([0.0,0.0])  # starting point
result = minimize(f_quad, x0, jac=grad_quad, method='BFGS')
print("UNCONSTRAINED QUADRATIC:")
print(f"Solution:     {result.x}")
print(f"f(x*):        {result.fun:.6f}")
print(f"Converged:    {result.success}")
print(f"Message:      {result.message}")
print(f"Iterations:   {result.nit}")

# ----------------------------------------------------------------
# PART 2: Non-convex function — local optima matter
# f(x) = sin(x) + 0.1*x^2  on [-10, 10]
# Multiple local minima — starting point determines which you find
# ----------------------------------------------------------------
def f_nonconvex(x):
    return np.sin(x[0]) + 0.1*x[0]**2
x_range = np.linspace(-10,10,500)
f_range = np.sin(x_range) + 0.1*x_range**2
starts = [-8.0, -3.0, 0.0, 5.0]
solutions = []
for x0 in starts:
    result = minimize(f_nonconvex, [x0], method='BFGS')
    solutions.append((x0, result.x[0], result.fun))
    print(f"Start={x0:5.1f}  ->  x* = {result.x[0]:6.3f}  "
          f"f(x*)={result.fun:.4f}")
    
# ----------------------------------------------------------------
# PART 3: Constrained optimization—a basic example
#
# A battery earns $p/MWh for discharging, pays $p/MWh for charging.
# We choose charge/discharge over 24 hours to maximize profit.
#
# Decision variables:
#   c[t] >= 0 : charge rate at hour t (MWh)
#   d[t] >= 0 : discharge rate at hour t (MWh)
#   s[t]      : state of charge at hour t (MWh)
#
# Objective: maximize sum_t price[t] * (d[t] - c[t])
#   = minimize sum_t price[t] * (c[t] - d[t])
#
# Constraints:
#   s[t] = s[t-1] + c[t]*eff_c - d[t]/eff_d  (energy balance)
#   0 <= s[t] <= capacity                      (SOC limits)
#   0 <= c[t] <= max_power                     (charge rate limit)
#   0 <= d[t] <= max_power                     (discharge rate limit)
#   s[0] = s_init                              (initial SOC)
#
# Here I use scipy; next time, I'll use PuLP/Pyomo, 
# which should handle this structure more cleanly
# ----------------------------------------------------------------
np.random.seed(42)
T = 24
# Simulate realistic hourly LMP prices (low overnight, high peak)
# I'll use real ERCOT data in Block 3; for now, synthetic
base_price = 30
price = (base_price
         + 20 * np.sin(2 * np.pi * (np.arange(T) - 6) / 24)
         + 15 * (np.arange(T) >= 15) * (np.arange(T) <= 20)
         + np.random.normal(0, 5, T))
print(f"\n=== Battery Dispatch Problem ===")
print(f"Peak price:   ${price.max():.2f}/MWh at hour "
      f"{price.argmax()}")
print(f"Min price:    ${price.min():.2f}/MWh at hour "
      f"{price.argmin()}")
# Battery parameters
capacity  = 4.0    # MWh — maximum energy storage
max_power = 1.0    # MW  — maximum charge/discharge rate per hour
eff_c     = 0.95   # charging efficiency
eff_d     = 0.95   # discharging efficiency
s0        = 2.0    # MWh — initial state of charge
# decision variables packed into one vector for scipy
# x = [c0, ..., c23, d0, ..., d23] (length 48)
def objective(x):
    c=x[:T]
    d=x[T:]
    # negative bc scipy minimizes by default and we want to maximize revenue
    return -np.sum(price*(d-c))
def get_soc(x):
    """Compute state-of-charge trajectory from c and d."""
    c, d = x[:T], x[T:]
    s = np.zeros(T)
    s[0] = s0 + c[0] * eff_c - d[0] / eff_d
    for t in range(1, T):
        s[t] = s[t-1] + c[t] * eff_c - d[t] / eff_d
    return s
# constraints: SOC must stay within [0, capacity] at each hour
constraints = []
for t in range(T):
    # s[t] >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda x, t=t: get_soc(x)[t]
    })
    # s[t] <= capacity
    constraints.append({
        'type': 'ineq',
        'fun': lambda x, t=t: capacity - get_soc(x)[t]
    })
# bounds: c[t], d[t] in [0, max_power]
bounds = [(0, max_power)]*T + [(0, max_power)]*T
# initial guess: do nothing
x0 = np.zeros(2*T)
result = minimize(
    objective, x0,
    method='SLSQP',          # Sequential Least Squares Programming
    bounds=bounds,           # box constraints on variables
    constraints=constraints, # SOC constraints
    options={'ftol': 1e-9, 'maxiter': 1000}
)
c_opt = result.x[:T]
d_opt = result.x[T:]
s_opt = get_soc(result.x)
revenue = -result.fun
print(f"Optimal revenue:  ${revenue:.2f}")
print(f"Converged:        {result.success}")
print(f"Total charged:    {c_opt.sum():.2f} MWh")
print(f"Total discharged: {d_opt.sum():.2f} MWh")

# plot
fig, axes = plt.subplots(3,1,figsize=(12,9))
axes[0].plot(price, color='steelblue', lw=2)
axes[0].set_ylabel('Price ($/MWh)')
axes[0].set_title('Hourly LMP Price')
axes[0].grid(True, alpha=0.3)
axes[1].bar(range(T), c_opt, color='tomato', alpha=0.7, label='Charging (cost)')
axes[1].bar(range(T), -d_opt, color='seagreen', alpha=0.7, label='Discharging (revenue)')
axes[1].set_ylabel('Power (MW)')
axes[1].set_title('Optimal Charge / Discharge Schedule')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[2].plot(s_opt, color='darkorange', lw=2)
axes[2].axhline(capacity, color='red', linestyle='--', lw=1, label=f'Capacity ({capacity} MWh)')
axes[2].axhline(0, color='black', linestyle='--', lw=1)
axes[2].set_ylabel('State of Charge (MWh)')
axes[2].set_xlabel('Hour of Day')
axes[2].set_title('Battery State of Charge')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('battery_dispatch.png', dpi=150, bbox_inches='tight')
plt.show()