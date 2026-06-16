import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ----------------------------------------------------------------
# PART 1: Bayesian updating — coin flip
# We observe a sequence of coin flips and update our belief
# about the probability of heads (theta) after each flip.
# Prior: Beta(alpha, beta) — uninformative Beta(1,1) = Uniform
# Likelihood: Binomial
# Posterior: Beta(alpha + heads, beta + tails)
# ----------------------------------------------------------------
# True (unknown) probability of heads
theta_true = 0.65
# Generate 200 coin flips
flips = (np.random.uniform(0, 1, 200) < theta_true).astype(int)
# Bayesian updating after each observation
# Prior parameters: alpha=1, beta=1 (uniform — no prior knowledge)
alpha_prior, beta_prior = 1.0, 1.0
# Snapshots: show posterior after 1, 5, 20, 50, 200 flips
snapshots    = [1, 5, 20, 50, 200]
theta_range  = np.linspace(0, 1, 500)
fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 4))
for ax, n in zip(axes, snapshots):
    heads  = flips[:n].sum()
    tails  = n - heads
    alpha_post = alpha_prior + heads
    beta_post  = beta_prior  + tails
    prior     = stats.beta.pdf(theta_range, alpha_prior, beta_prior)
    posterior = stats.beta.pdf(theta_range, alpha_post, beta_post)
    ax.plot(theta_range, prior, color='gray', lw=1.5, linestyle='--', label='Prior')
    ax.plot(theta_range, posterior, color='steelblue', lw=2, 
            label=f'Posterior\n(n={n})')
    ax.axvline(theta_true, color='tomato', linestyle='--', lw=1.5, 
               label=f'True θ={theta_true}')
    ax.axvline(alpha_post/(alpha_post+beta_post), color='steelblue', linestyle=':', 
               lw=1.5, label=f'Post. mean={alpha_post/(alpha_post+beta_post):.2f}')
    ax.set_title(f'n={n} flips\n({heads}H, {tails}T)')
    ax.set_xlabel('θ (P(heads))')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Density')
plt.suptitle('Bayesian Updating: Posterior Converges to Truth', y=1.02)
plt.tight_layout()
plt.savefig('bayesian_updating.png', dpi=150, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------
# PART 2: Bayesian vs. frequentist — credible vs confidence intervals
# ----------------------------------------------------------------
n_obs    = 50
heads    = flips[:n_obs].sum()
alpha_p  = alpha_prior + heads
beta_p   = beta_prior + (n_obs - heads)
# Bayesian 95% credible interval
cred_low, cred_high = stats.beta.ppf([0.025, 0.975], alpha_p, beta_p)
# Frequentist 95% confidence interval (Wilson score interval)
p_hat    = heads / n_obs
z        = 1.96
conf_low  = p_hat - z * np.sqrt(p_hat*(1-p_hat)/n_obs)
conf_high = p_hat + z * np.sqrt(p_hat*(1-p_hat)/n_obs)
print("=== Bayesian vs. Frequentist (n=50 flips) ===")
print(f"True theta:              {theta_true}")
print(f"Observed proportion:     {p_hat:.3f} ({heads}/{n_obs})")
print(f"Posterior mean:          "
      f"{alpha_p/(alpha_p+beta_p):.3f}")
print(f"95% Credible interval:   [{cred_low:.3f}, {cred_high:.3f}]")
print(f"95% Confidence interval: [{conf_low:.3f}, {conf_high:.3f}]")

# ----------------------------------------------------------------
# PART 3: Monte Carlo integration
# Estimate the expected annual revenue of a battery under
# uncertain prices using Monte Carlo sampling.
# ----------------------------------------------------------------
def simple_battery_revenue(price_path, capacity=4.0,
                            max_power=1.0, eff=0.90):
    """
    Simplified battery revenue: buy at the n cheapest hours,
    sell at the n most expensive hours.
    Ignores SOC constraints for speed — pure price spread.
    """
    T       = len(price_path)
    n_hours = int(capacity / max_power)   # hours at max rate
    sorted_hours = np.argsort(price_path)
    buy_hours    = sorted_hours[:n_hours]
    sell_hours   = sorted_hours[-n_hours:]
    buy_cost     = price_path[buy_hours].sum()  * max_power
    sell_revenue = price_path[sell_hours].sum() * max_power * eff
    return sell_revenue - buy_cost
# price model: log-normal daily prices with realistic ERCOT parameters
# Mean ~$45/MWh, moderate volatility, some spike risk
n_scenarios = 10000
T = 8760  # hours in a year
print("\n=== Monte Carlo Battery Valuation ===")
print(f"Simulating {n_scenarios:,} annual price scenarios...")
revenues = []
for _ in range(n_scenarios):
    # daily mean prices varies seasonally
    daily_means = 30 + 20*np.sin(2*np.pi*(np.arange(365)-60)/365)
    hourly_means = np.repeat(daily_means, 24)
    # log-normal prices with spike risk
    prices = np.random.lognormal(
        mean=np.log(hourly_means)-0.3,
        sigma=0.5,
        size=T
    )
    # occasional price spikes
    n_spikes = np.random.poisson(lam=50)
    if n_spikes > 0:
        spike_hours = np.random.choice(T, n_spikes, replace=False)
        prices[spike_hours] *= np.random.uniform(5,20,n_spikes)
    revenues.append(simple_battery_revenue(prices))
revenues = np.array(revenues)
# Monte Carlo estimates converge as N grows
N_samples = [10, 50, 100, 500, 1000, 5000, 10000]
mc_means  = [revenues[:n].mean() for n in N_samples]
mc_stds   = [revenues[:n].std() / np.sqrt(n) for n in N_samples]
print(f"\nExpected annual revenue:      ${revenues.mean():,.0f}")
print(f"Standard deviation:           ${revenues.std():,.0f}")
print(f"5th percentile (downside):    ${np.percentile(revenues,5):,.0f}")
print(f"95th percentile (upside):     ${np.percentile(revenues,95):,.0f}")
print(f"P(revenue > $50,000):         "
      f"{(revenues > 50000).mean():.1%}")

# --- plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Monte Carlo convergence
axes[0].plot(N_samples, mc_means, 'o-', color='steelblue', lw=2)
axes[0].fill_between(
    N_samples,
    [m - 2*s for m,s in zip(mc_means, mc_stds)],
    [m + 2*s for m,s in zip(mc_means, mc_stds)],
    alpha=0.2, color='steelblue')
axes[0].axhline(revenues.mean(), color='tomato', linestyle='--',
                lw=1.5, label='True mean')
axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Scenarios')
axes[0].set_ylabel('Estimated Mean Revenue ($)')
axes[0].set_title('Monte Carlo Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Revenue distribution
axes[1].hist(revenues, bins=80, color='steelblue', edgecolor='white', alpha=0.8, 
             density=True)
axes[1].axvline(revenues.mean(), color='tomato', linestyle='--', 
                lw=2, label=f'Mean: ${revenues.mean():,.0f}')
axes[1].axvline(np.percentile(revenues, 5), color='orange', linestyle='--', lw=2,
                label=f'P5: ${np.percentile(revenues,5):,.0f}')
axes[1].axvline(np.percentile(revenues, 95), color='seagreen',
                linestyle='--', lw=2, label=f'P95: ${np.percentile(revenues,95):,.0f}')
axes[1].set_xlabel('Annual Revenue ($)')
axes[1].set_ylabel('Density')
axes[1].set_title('Battery Revenue Distribution\n(10,000 Scenarios)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
# Bayesian updating summary plot
alpha_vals = alpha_prior + np.cumsum(flips)
beta_vals  = beta_prior  + np.cumsum(1 - flips)
post_means = alpha_vals / (alpha_vals + beta_vals)
post_lower = stats.beta.ppf(0.025, alpha_vals, beta_vals)
post_upper = stats.beta.ppf(0.975, alpha_vals, beta_vals)
axes[2].plot(post_means, color='steelblue', lw=1.5, label='Posterior mean')
axes[2].fill_between(range(len(flips)), post_lower, post_upper, alpha=0.2, 
                     color='steelblue', label='95% credible interval')
axes[2].axhline(theta_true, color='tomato', linestyle='--', lw=2, 
                label=f'True θ={theta_true}')
axes[2].set_xlabel('Number of Observations')
axes[2].set_ylabel('θ (P(heads))')
axes[2].set_title('Posterior Mean and\n95% Credible Interval')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bayesian_montecarlo.png', dpi=150, bbox_inches='tight')
plt.show()