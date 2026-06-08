import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import (ConcreteModel, Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals, maximize, value, SolverFactory)

np.random.seed(42)
T_range = range(24)
# hourly LMP prices
base_price = 30
price_data = {t: (base_price
                  + 20 * np.sin(2 * np.pi * (t - 6) / 24)
                  + 15 * (15 <= t <= 20)
                  + np.random.normal(0, 5))
              for t in T_range}

# ----------------------------------------------------------------
# PART 1: Battery dispatch in Pyomo — same problem as PuLP
# ----------------------------------------------------------------
def build_battery_model(price_data, capacity=4.0, max_power=1.0, eff_c=0.95, 
                        eff_d=0.95, s_init=2.0):
    m = ConcreteModel()
    # --- Sets ---
    m.T = Set(initialize=list(price_data.keys()), ordered=True)
    # --- Parameters ---
    m.price    = Param(m.T, initialize=price_data)
    m.cap      = Param(initialize=capacity)
    m.pmax     = Param(initialize=max_power)
    m.eff_c    = Param(initialize=eff_c)
    m.eff_d    = Param(initialize=eff_d)
    m.s_init   = Param(initialize=s_init)
    # --- Variables ---
    m.c = Var(m.T, within=NonNegativeReals, bounds=(0, max_power))
    m.d = Var(m.T, within=NonNegativeReals, bounds=(0, max_power))
    m.s = Var(m.T, within=NonNegativeReals, bounds=(0, capacity))
    # --- Objective ---
    m.obj = Objective(
        expr=sum(m.price[t] * (m.d[t] - m.c[t]) for t in m.T),
        sense=maximize
    )
    # --- Constraints: energy balance via rules ---
    def energy_balance_rule(m, t):
        T_list = list(m.T)
        idx = T_list.index(t)
        if idx == 0:
            return m.s[t] == m.s_init + m.c[t]*m.eff_c - m.d[t]/m.eff_d
        else:
            prev = T_list[idx - 1]
            return m.s[t] == m.s[prev] + m.c[t]*m.eff_c - m.d[t]/m.eff_d
    m.energy_balance = Constraint(m.T, rule=energy_balance_rule)
    return m
model = build_battery_model(price_data)
solver = SolverFactory('glpk')
result = solver.solve(model, tee=False)
c_opt = np.array([value(model.c[t]) for t in T_range])
d_opt = np.array([value(model.d[t]) for t in T_range])
s_opt = np.array([value(model.s[t]) for t in T_range])
rev1  = value(model.obj)
print("=== Battery Only ===")
print(f"Revenue:  ${rev1:.2f}")
print(f"Charged:  {c_opt.sum():.2f} MWh")
print(f"Discharged:{d_opt.sum():.2f} MWh")


# ----------------------------------------------------------------
# PART 2: Add a gas generator — MILP unit commitment
# A peaker gas plant can turn on (y[t]=1) or off (y[t]=0).
# When on: produces between p_min and p_max MW.
# Earns revenue by selling output at LMP.
# Costs: startup cost each time it turns on + fuel cost per MWh.
# Combined with battery: jointly optimize both assets.
# ----------------------------------------------------------------
# Gas generator parameters
p_min      = 0.5    # MW — minimum stable generation when on
p_max      = 2.0    # MW — maximum output
fuel_cost  = 25.0   # $/MWh — marginal fuel cost (heat rate * gas price)
startup    = 10.0   # $ — cost incurred each time generator starts up

def build_combined_model(price_data,
                         # battery
                         capacity=4.0, max_power=1.0,
                         eff_c=0.95, eff_d=0.95, s_init=2.0,
                         # gas generator
                         p_min=0.5, p_max=2.0,
                         fuel_cost=25.0, startup=10.0):
    m = ConcreteModel()
    m.T     = Set(initialize=list(price_data.keys()), ordered=True)
    m.price = Param(m.T, initialize=price_data)
    # Battery variables
    m.c = Var(m.T, within=NonNegativeReals, bounds=(0, max_power))
    m.d = Var(m.T, within=NonNegativeReals, bounds=(0, max_power))
    m.s = Var(m.T, within=NonNegativeReals, bounds=(0, capacity))
    # Generator variables
    m.gen  = Var(m.T, within=NonNegativeReals)  # output (MW)
    m.y    = Var(m.T, within=Binary)             # on/off (1/0)
    m.su   = Var(m.T, within=Binary)             # startup indicator
    # Objective: revenue from battery + revenue from gas - fuel costs - startup
    m.obj = Objective(
        expr=sum(
            m.price[t] * (m.d[t] - m.c[t])          # battery arbitrage
            + m.price[t] * m.gen[t]                   # gas revenue
            - fuel_cost * m.gen[t]                    # fuel cost
            - startup * m.su[t]                       # startup cost
            for t in m.T
        ),
        sense=maximize
    )
    # Battery energy balance
    def energy_balance_rule(m, t):
        T_list = list(m.T)
        idx = T_list.index(t)
        if idx == 0:
            return m.s[t] == s_init + m.c[t]*eff_c - m.d[t]/eff_d
        prev = T_list[idx - 1]
        return m.s[t] == m.s[prev] + m.c[t]*eff_c - m.d[t]/eff_d
    m.energy_balance = Constraint(m.T, rule=energy_balance_rule)
    # Generator output bounds — big-M constraints
    def gen_min_rule(m, t):
        return m.gen[t] >= p_min * m.y[t]
    def gen_max_rule(m, t):
        return m.gen[t] <= p_max * m.y[t]
    m.gen_min = Constraint(m.T, rule=gen_min_rule)
    m.gen_max = Constraint(m.T, rule=gen_max_rule)
    # Startup indicator: su[t] = 1 if y[t]=1 and y[t-1]=0
    # su[t] >= y[t] - y[t-1]  (turns on this hour)
    def startup_rule(m, t):
        T_list = list(m.T)
        idx = T_list.index(t)
        if idx == 0:
            return m.su[t] >= m.y[t]   # first hour: startup if on
        prev = T_list[idx - 1]
        return m.su[t] >= m.y[t] - m.y[prev]
    m.startup_con = Constraint(m.T, rule=startup_rule)
    return m
model2 = build_combined_model(price_data)
result2 = solver.solve(model2, tee=False)
c2   = np.array([value(model2.c[t])   for t in T_range])
d2   = np.array([value(model2.d[t])   for t in T_range])
s2   = np.array([value(model2.s[t])   for t in T_range])
gen2 = np.array([value(model2.gen[t]) for t in T_range])
y2   = np.array([value(model2.y[t])   for t in T_range])
rev2 = value(model2.obj)
prices = np.array(list(price_data.values()))
n_starts = int(y2.sum() > 0) + sum(
    1 for t in range(1, 24) if y2[t] > 0.5 and y2[t-1] < 0.5
)
print("\n=== Battery + Gas Generator (MILP) ===")
print(f"Revenue:          ${rev2:.2f}")
print(f"Generator hours:  {int(y2.sum())}")
print(f"Generator starts: {n_starts}")
print(f"Hours dispatched: {list(np.where(y2 > 0.5)[0])}")

# plot
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
prices_arr = np.array([price_data[t] for t in T_range])
axes[0].plot(prices_arr, color='steelblue', lw=2, marker='o', ms=4)
axes[0].axhline(fuel_cost, color='red', linestyle='--',
                lw=1, label=f'Fuel cost (${fuel_cost}/MWh)')
axes[0].set_ylabel('LMP ($/MWh)')
axes[0].set_title('Hourly LMP vs. Gas Fuel Cost')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].bar(T_range, c2, color='tomato', alpha=0.7, label='Charge')
axes[1].bar(T_range, -d2, color='seagreen', alpha=0.7, label='Discharge')
axes[1].set_ylabel('Power (MW)')
axes[1].set_title('Battery Dispatch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[2].bar(T_range, gen2, color='darkorange', alpha=0.8,
            label='Gas output')
axes[2].plot(T_range, y2 * p_min, 'r--', lw=1, label='Min output (if on)')
axes[2].set_ylabel('Output (MW)')
axes[2].set_title('Gas Generator Dispatch')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[3].plot(s2, color='purple', lw=2, marker='o', ms=4)
axes[3].axhline(4.0, color='red', linestyle='--', lw=1,
                label='Capacity (4 MWh)')
axes[3].set_ylabel('SOC (MWh)')
axes[3].set_xlabel('Hour of Day')
axes[3].set_title('Battery State of Charge')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pyomo_dispatch.png', dpi=150, bbox_inches='tight')
plt.show()