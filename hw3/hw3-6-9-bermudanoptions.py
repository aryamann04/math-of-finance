import numpy as np

# parameters
A_0 = 100
r = 0.04
sigma = 0.10
T = 0.5
N = 6
K = 100

dt = T / N
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
p = (np.exp(r * dt) - d) / (u - d)

prices = np.zeros((N + 1, N + 1))
for n in range(N + 1):
    for j in range(n + 1):
        prices[n, j] = A_0 * (u ** j) * (d ** (n - j))

# backward induction for vanilla european puts
european_put_prices = []
for steps in range(1, N + 1):
    option_values = np.maximum(K - prices[steps, :steps + 1], 0)
    for n in range(steps - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
    european_put_prices.append(option_values[0])

# backward induction for bermudan puts
bermudan_put_values = np.maximum(K - prices[N, :], 0)
for n in range(N - 1, -1, -1):
    bermudan_put_values = np.maximum(K - prices[n, :n + 1],
                                     np.exp(-r * dt) * (p * bermudan_put_values[1:] + (1 - p) * bermudan_put_values[:-1]))
bermudan_put_price = bermudan_put_values[0]

# backward induction for bermudan calls
bermudan_call_values = np.maximum(prices[N, :] - K, 0)
for n in range(N - 1, -1, -1):
    bermudan_call_values = np.maximum(prices[n, :n + 1] - K,
                                      np.exp(-r * dt) * (p * bermudan_call_values[1:] + (1 - p) * bermudan_call_values[:-1]))
bermudan_call_price = bermudan_call_values[0]

# call price for comparison
european_call_values = np.maximum(prices[N, :] - K, 0)
for n in range(N - 1, -1, -1):
    european_call_values = np.exp(-r * dt) * (p * european_call_values[1:] + (1 - p) * european_call_values[:-1])

european_call_price = european_call_values[0]

print(bermudan_put_price)
print(european_put_prices)
print(bermudan_call_price)
print(european_call_price)
