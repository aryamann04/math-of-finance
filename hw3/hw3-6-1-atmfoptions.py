import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

def pv(x, r):
    return x * np.exp(-r * T)

def f_a(A, T, r):
    return A * np.exp(r * T)

def d(A, K, T, sigma, r):
    d1 = np.log(f_a(A, T, r)/K) / (sigma * np.sqrt(T)) + (1/2) * sigma * np.sqrt(T)
    return d1, d1 - sigma * np.sqrt(T)

def bs_call_price(A, K, T, sigma, r):
    d1, d2 = d(A, K, T, sigma, r)
    c0 = np.exp(-r * T) * (f_a(A, T, r) * norm.cdf(d1) - K * norm.cdf(d2))
    return c0

def implied_vol(option_price, A, K, T, r):

    def error_function(sigma):
        return option_price - bs_call_price(A, K, T, sigma, r)

    return fsolve(error_function, 0.1)[0]

def delta(A, K, T, sigma, r):
    d1, _ = d(A, K, T, sigma, r)
    return norm.cdf(d1)

def gamma(A, K, T, sigma, r):
    d1, _ = d(A, K, T, sigma, r)
    return norm.pdf(d1)/(A * sigma * np.sqrt(T))

def vega(A, K, T, sigma, r):
    d1, _ = d(A, K, T, sigma, r)
    return A * np.sqrt(T) * norm.pdf(d1)

def bs_put_price(A, K, T, sigma, r):
    return bs_call_price(A, K, T, sigma, r) + pv(K, r) - A

def atmf_prices(A, K, T, sigma, r):
    x = (1/2) * sigma * np.sqrt(T)
    c0 = 2 * K * np.exp(-r * T) * norm.cdf(x) - K * np.exp(-r * T)
    p0 = 2 * K * np.exp(-r * T) * norm.cdf(x) - A
    straddle = 4 * K * np.exp(-r * T) * norm.cdf(x) - (A + K * np.exp(-r * T))
    return c0, p0, straddle

# parameters
A = 100
T = 0.25
r = 0.04
K = f_a(A, T, r)
atmf_call_price = 2.5

# 1: implied volatility
sigma = implied_vol(atmf_call_price, A, K, T, r)
print(f"implied vol: {sigma * 100}%\n")

# 2: delta, gamma, vega
print(f"delta: {delta(A, K, T, sigma, r)}")
print(f"gamma: {gamma(A, K, T, sigma, r)}")
print(f"vega: {vega(A, K, T, sigma, r)}\n")

# 3: ATMF straddle
atmf_put_price = bs_put_price(A, K, T, sigma, r)
atmf_straddle_price = atmf_put_price + atmf_call_price
print(f"put price: {atmf_put_price}")
print(f"call price: {bs_call_price(A, K, T, sigma, r)}")
print(f"ATMF straddle price: {atmf_straddle_price}\n")

c0, p0, straddle = atmf_prices(A, K, T, sigma, r)
print(f"put price: {p0}")
print(f"call price: {c0}")
print(f"ATMF straddle price: {straddle}\n")

# 4: increase in asset price
A_new = 101
delta_change = (bs_call_price(A_new, K, T, sigma, r) + bs_put_price(A_new, K, T, sigma, r)) - atmf_straddle_price
print(f"change after $1 increase in A(0): {delta_change}\n")

# 5: increase in vol
sigma_new = sigma + 0.01
vol_change = (bs_call_price(A, K, T, sigma_new, r) + bs_put_price(A, K, T, sigma_new, r)) - atmf_straddle_price
print(f"change after 0.01 increase in sigma: {vol_change}\n")
