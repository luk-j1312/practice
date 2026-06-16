import numpy as np
import matplotlib.pyplot as plt
import pulp

np.random.seed(42)
T = 24
# same prices as before
base_price = 30
price = (base_price
         + 20 * np.sin(2 * np.pi * (np.arange(T) - 6) / 24)
         + 15 * ((np.arange(T) >= 15) & (np.arange(T) <= 20)).astype(float)
         + np.random.normal(0, 5, T))
# battery parameters
capacity  = 4.0    # MWh — maximum state of charge
max_power = 1.0    # MW  — maximum charge/discharge rate
eff_c     = 0.95   # charging efficiency (grid → storage)
eff_d     = 0.95   # discharging efficiency (storage → grid)
s_init    = 2.0    # MWh — initial state of charge

# -----------------------------------------------------------------
# FORMULATE THE LP
# -----------------------------------------------------------------
prob = pulp.LpProblem("battery_dispatch", pulp.LpMaximize)
# Decision variables—PuLP enforces bounds directly
c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=max_power)
     for t in range(T)]   # charge rate (MW) at hour t
d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=max_power)
     for t in range(T)]   # discharge rate (MW) at hour t
s = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=capacity)
     for t in range(T)]   # state of charge (MWh) at end of hour t
# Objective: maximize sum of price * net discharge
prob += pulp.lpSum(price[t] * (d[t] - c[t]) for t in range(T)), \
       "total_revenue"
# Energy balance—named constraints (required for shadow prices)
prob += (s[0] == s_init + c[0] * eff_c - d[0] / eff_d,
         "energy_balance_0")
for t in range(1, T):
    prob += (s[t] == s[t-1] + c[t] * eff_c - d[t] / eff_d,
             f"energy_balance_{t}")
    
prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

status = pulp.LpStatus[prob.status]
revenue = pulp.value(prob.objective)
c_opt = np.array([pulp.value(c[t]) for t in range(T)])
d_opt = np.array([pulp.value(d[t]) for t in range(T)])
s_opt = np.array([pulp.value(s[t]) for t in range(T)])
print(f"Status:           {status}")
print(f"Optimal revenue:  ${revenue:.2f}")
print(f"Total charged:    {c_opt.sum():.2f} MWh")
print(f"Total discharged: {d_opt.sum():.2f} MWh")
print(f"Final SOC:        {s_opt[-1]:.3f} MWh")

# -----------------------------------------------------------------
# SHADOW PRICES — marginal value of each constraint
# -----------------------------------------------------------------
print("\nShadow prices (energy balance constraints):")
print(f"{'Hour':<6} {'Price':>10} {'Shadow Price':>14} {'Interpretation'}")
print("-" * 55)
for t in range(T):
    constraint = prob.constraints[f"energy_balance_{t}"]
    sp = constraint.pi   # dual variable / shadow price
    print(f"{t:<6} {price[t]:>9.2f}  {sp:>13.4f}")

# -----------------------------------------------------------------
# SENSITIVITY: what is one extra MWh of capacity worth?
# -----------------------------------------------------------------
# resolve with capacity + 1
prob2 = pulp.LpProblem("battery_dispatch_larger", pulp.LpMaximize)
c2 = [pulp.LpVariable(f"c2_{t}", lowBound=0, upBound=max_power)
      for t in range(T)]
d2 = [pulp.LpVariable(f"d2_{t}", lowBound=0, upBound=max_power)
      for t in range(T)]
s2 = [pulp.LpVariable(f"s2_{t}", lowBound=0,
                       upBound=capacity + 1.0)   # one extra MWh
      for t in range(T)]
prob2 += pulp.lpSum(price[t] * (d2[t] - c2[t]) for t in range(T))
prob2 += (s2[0] == s_init + c2[0] * eff_c - d2[0] / eff_d,
          "eb_0")
for t in range(1, T):
    prob2 += (s2[t] == s2[t-1] + c2[t] * eff_c - d2[t] / eff_d,
              f"eb_{t}")
prob2.solve(pulp.PULP_CBC_CMD(msg=0))
revenue2 = pulp.value(prob2.objective)
print(f"\nRevenue with 4 MWh capacity:  ${revenue:.2f}")
print(f"Revenue with 5 MWh capacity:  ${revenue2:.2f}")
print(f"Marginal value of +1 MWh:     ${revenue2 - revenue:.2f}")

# plot
fig, axes = plt.subplots(3,1,figsize=(12,9), sharex=True)
axes[0].plot(price, color='steelblue', lw=2, marker='o', ms=4)
axes[0].set_ylabel('LMP ($/MWh)')
axes[0].set_title('Hourly Price')
axes[0].grid(True, alpha=0.3)
axes[1].bar(range(T), c_opt, color='tomato', alpha=0.7, label='Charge (cost)')
axes[1].bar(range(T), -d_opt, color='seagreen', alpha=0.7, label='Discharge (revenue)')
axes[1].set_ylabel('Power (MW)')
axes[1].set_title('Optimal Dispatch Schedule (PuLP)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[2].plot(s_opt, color='darkorange', lw=2, marker='o', ms=4)
axes[2].axhline(capacity, color='red', linestyle='--', lw=1, label=f'Capacity ({capacity} MWh)')
axes[2].set_ylabel('State of Charge (MWh)')
axes[2].set_xlabel('Hour of Day')
axes[2].set_title('State of Charge')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('battery_pulp.png', dpi=150, bbox_inches='tight')
plt.show()
