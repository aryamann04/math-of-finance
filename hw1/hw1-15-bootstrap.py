#------------------------------------------#
# Q15: bootstrap algorithm implementation  #
#------------------------------------------#

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

D = {float(0): 1.0, float(0.5): 1 - 0.0075 * 182 / 360}  # D(0.5) from T-Bill

instruments = [
    {'Maturity': 2.0, 'CouponRate': 1.375, 'MarketPrice': 99.95086},
    {'Maturity': 3.0, 'CouponRate': 2.125, 'MarketPrice': 100.36222},
    {'Maturity': 5.0, 'CouponRate': 2.5, 'MarketPrice': 100.0},
    {'Maturity': 7.0, 'CouponRate': 3.0, 'MarketPrice': 100.62942},
    {'Maturity': 10.0, 'CouponRate': 3.25, 'MarketPrice': 100.42501},
]

def bootstrap(instruments, interpolation='linear'):
    D_factors = D.copy()
    known_times = sorted(D_factors.keys())
    for instr in instruments:
        maturity = float(instr['Maturity'])
        coupon_rate = instr['CouponRate'] / 100
        price = instr['MarketPrice']
        periods = int(maturity * 2)
        cash_flows = np.array([coupon_rate / 2 * 100] * periods)
        cash_flows[-1] += 100  # Add principal
        times = np.arange(0.5, maturity + 0.1, 0.5)
        times = [float(t) for t in times]

        def equation(D_tn):
            D_factors_temp = D_factors.copy()
            D_factors_temp[maturity] = D_tn[0]
            pv = 0
            for t, cf in zip(times, cash_flows):
                D_t = interpolate_D(t, D_factors_temp, interpolation)
                pv += cf * D_t
            return pv - price

        D_tn_initial = [0.9]
        D_tn_solution = fsolve(equation, D_tn_initial)
        D_factors[maturity] = D_tn_solution[0]
        known_times.append(maturity)
        known_times.sort()
    return D_factors

def interpolate_D(t, D_factors_temp, interpolation):
    t = float(t)
    if t in D_factors_temp:
        return D_factors_temp[t]
    else:
        known_times_temp = sorted(D_factors_temp.keys())
        t1 = max([k for k in known_times_temp if k < t], default=None)
        t2 = min([k for k in known_times_temp if k > t], default=None)
        if t1 is None or t2 is None:
            raise ValueError(f"Cannot interpolate for t={t}: insufficient known discount factors.")
        D1 = D_factors_temp[t1]
        D2 = D_factors_temp[t2]
        if interpolation == 'linear':
            return D1 + (t - t1) / (t2 - t1) * (D2 - D1)
        elif interpolation == 'log-linear':
            return D1 * (D2 / D1) ** ((t - t1) / (t2 - t1))

D_linear = bootstrap(instruments, interpolation='linear')
D_loglinear = bootstrap(instruments, interpolation='log-linear')

def generate_full_discount_factors(D_factors, interpolation):
    times = np.arange(0.5, 10.1, 0.5)
    full_D = {}
    for t in times:
        if t in D_factors:
            full_D[t] = D_factors[t]
        else:
            full_D[t] = interpolate_D(t, D_factors, interpolation)
    return full_D

D_linear_full = generate_full_discount_factors(D_linear, 'linear')
D_loglinear_full = generate_full_discount_factors(D_loglinear, 'log-linear')

def zero_coupon_yields(D_factors):
    yields = {}
    for t in D_factors:
        if t > 0:
            s_t = 2 * (D_factors[t] ** (-1 / (2 * t)) - 1)
            yields[t] = s_t * 100  # Convert to percentage
    return yields

yields_linear = zero_coupon_yields(D_linear_full)
yields_loglinear = zero_coupon_yields(D_loglinear_full)

# Plotting the discount factors and zero-coupon yield curves
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Maturity (Years)')
ax1.set_ylabel('Discount Factors', color=color)
ax1.plot(sorted(D_linear_full.keys()), [D_linear_full[t] for t in sorted(D_linear_full.keys())],
         marker='o', color='blue', label='Discount Factors (Linear)')
ax1.plot(sorted(D_loglinear_full.keys()), [D_loglinear_full[t] for t in sorted(D_loglinear_full.keys())],
         marker='x', color='cyan', label='Discount Factors (Log-Linear)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_yaxis()  # Invert y-axis for discount factors

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Zero-Coupon Yield (%)', color=color)
ax2.plot(sorted(yields_linear.keys()), [yields_linear[t] for t in sorted(yields_linear.keys())],
         marker='s', linestyle='--', color='red', label='Zero-Coupon Yields (Linear)')
ax2.plot(sorted(yields_loglinear.keys()), [yields_loglinear[t] for t in sorted(yields_loglinear.keys())],
         marker='^', linestyle='--', color='magenta', label='Zero-Coupon Yields (Log-Linear)')
ax2.tick_params(axis='y', labelcolor=color)

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title('Discount Factors and Zero-Coupon Yields')
plt.grid(True)
plt.show()

# Pricing 5-year bonds using log-linear discount factors
def price_bond(D_factors, maturity, coupon_rate):
    periods = int(maturity * 2)
    cash_flows = np.array([coupon_rate / 2 * 100] * periods)
    cash_flows[-1] += 100  # Add principal
    times = np.arange(0.5, maturity + 0.1, 0.5)
    times = [float(t) for t in times]  # Ensure times are floats
    pv = sum(cf * D_factors[t] for cf, t in zip(cash_flows, times))
    return pv

def yield_to_maturity(price, maturity, coupon_rate):
    periods = int(maturity * 2)
    cash_flows = np.array([coupon_rate / 2 * 100] * periods)
    cash_flows[-1] += 100  # Add principal
    times = np.arange(1, periods + 1)

    def func(y):
        pv = sum(cf / (1 + y / 2) ** n for cf, n in zip(cash_flows, times))
        return pv - price

    ytm = fsolve(func, 0.05)[0]
    return ytm * 100

# 5-year bond with 1% coupon
price_1pct = price_bond(D_loglinear_full, 5.0, 0.01)
ytm_1pct = yield_to_maturity(price_1pct, 5.0, 0.01)
print(f"Price of 5-year bond with 1% coupon: ${price_1pct:.4f}")
print(f"Yield to maturity: {ytm_1pct:.4f}%")

# 5-year bond with 8% coupon
price_8pct = price_bond(D_loglinear_full, 5.0, 0.08)
ytm_8pct = yield_to_maturity(price_8pct, 5.0, 0.08)
print(f"Price of 5-year bond with 8% coupon: ${price_8pct:.4f}")
print(f"Yield to maturity: {ytm_8pct:.4f}%")
