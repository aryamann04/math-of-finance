import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

r = 0.039801
forward_times_to_exp = [3/12, 2/12, 1/12, 0]
spots = [100, 99, 100, 101]
K = spots[0] * np.exp(r * forward_times_to_exp[0])

f_a = [
    spots[0] * np.exp(r * forward_times_to_exp[0]),
    spots[1] * np.exp(r * forward_times_to_exp[1]),
    spots[2] * np.exp(r * forward_times_to_exp[2]),
    spots[3]
]

vf_a = [f_a[i] - K for i in range(len(spots))]

table = {
    "Date": ["April 1st", "May 1st", "June 1st", "July 1st"],
    "Spot Price ($)": spots,
    "Forward Expiry (Years)": forward_times_to_exp,
    "Forward Price ($)": f_a,
    "Value of Forward ($)": vf_a
}

# output table 4.1
df = pd.DataFrame(table)
df["Forward Price ($)"] = df["Forward Price ($)"].round(2)
df["Value of Forward ($)"] = df["Value of Forward ($)"].round(2)
print("Updated Table 4.1: 3-Month Evolution of Spot and Forward Prices")
print(df.to_string(index=False))

# output figure 4.2
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(table["Date"], spots, color='black', label='A(t)')
ax1.plot(table["Date"], f_a, color='grey', label='F_A(t, T)')
ax1.set_ylabel("Price ($)")
ax1.set_xlabel("Date")
ax1.set_yticks(range(97, 104, 1))
ax1.set_ylim(97, 103)
ax1.legend(loc="upper left")
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(table["Date"], vf_a, color='lightgrey', linestyle='dashed', label='VF_A(t, T, K)')
ax2.set_ylabel("Value of Forward ($)")
ax2.set_yticks(range(-3, 4, 1))
ax2.set_ylim(-3, 3)
ax2.legend(loc="upper right")
plt.title("3-Month Evolution of Spot and Forward Prices")
plt.show()
