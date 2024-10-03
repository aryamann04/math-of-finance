#------------------------------------------#
# Q5: graphing clean and dirty bond prices #
#------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# define bond parameters

c = 0.04
m = 2

coupon_dates = pd.to_datetime(['2020-07-01', '2021-01-01', '2021-07-01', '2022-01-01'])
days_between_coupons = [182, 184, 181, 184]
dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')

# calculate dirty and clean bond prices
def calculate_prices(yield_rate):
    dirty_prices = []
    clean_prices = []

    for date in dates:
        N = sum(date < cd for cd in coupon_dates) # remaining cash flows
        prev_coupon_date = coupon_dates[4-N]
        days_since_last_coupon = (date - prev_coupon_date).days

        w = 1 + days_since_last_coupon / days_between_coupons[4-N]
        accrued_interest = w * (c/m)

        dirty_price = ((1 + yield_rate / m) ** w) * (
                (c / yield_rate) * (1 - 1 / (1 + yield_rate / m) ** N) +
                1 / (1 + yield_rate / m) ** N
        )

        clean_price = dirty_price - accrued_interest

        dirty_prices.append(dirty_price)
        clean_prices.append(clean_price)

    return dirty_prices, clean_prices

def plot_yield(yield_rate):
    dirty_prices, clean_prices = calculate_prices(yield_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, dirty_prices, label=f'Dirty Price (Yield: {yield_rate * 100}%)')
    plt.plot(dates, clean_prices, label=f'Clean Price (Yield: {yield_rate * 100}%)')
    plt.title(f'Bond Prices (Yield Constant at {yield_rate * 100}%)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    plt.show()

plot_yield(0.04)  # yield held constant at 4% - par bond
plot_yield(0.05)  # yield held constant at 5% - discount bond
plot_yield(0.03)  # yield held constant at 3% - premium bond
