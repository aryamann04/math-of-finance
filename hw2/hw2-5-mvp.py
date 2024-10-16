#------------------------------------------#
#    Q5.1: feasible portfolio set graph    #
#------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

# given parameters

mu_1 = 0.04
sigma_1 = 0.1

mu_2 = 0.06
sigma_2 = 0.06

rho = 0.25
cov = rho * sigma_1 * sigma_2

# calculating full hyperbola of portfolios

all_weights = np.linspace(-1,2,200)

all_mu_p = all_weights * mu_1 + (1 - all_weights) * mu_2
all_sigma_p = np.sqrt((all_weights**2 * sigma_1**2)
                   + ((1 - all_weights)**2 * sigma_2**2)
                   + (2 * all_weights * (1 - all_weights) * cov))

# calculating feasible set with R1 and R2

weights = np.linspace(0,1,100)

mu_p = weights * mu_1 + (1 - weights) * mu_2
sigma_p = np.sqrt((weights**2 * sigma_1**2)
                   + ((1 - weights)**2 * sigma_2**2)
                   + (2 * weights * (1 - weights) * cov))

plt.figure(figsize=(10,6))

plt.plot(all_sigma_p, all_mu_p, label="", color = "black") # plot full set
plt.plot(sigma_p, mu_p, label="feasible set", color = "blue") # plot feasible set

# plot and annotate start/end of feasible set

w1_0 = (sigma_p[0], mu_p[0])
w1_1 = (sigma_p[-1], mu_p[-1])

plt.scatter(*w1_0, color='red', label='w1=0 (all R2)', zorder=5)
plt.scatter(*w1_1, color='green', label='w1=1 (all R1)', zorder=5)
plt.annotate('w1=0', (w1_0[0], w1_0[1]),
             textcoords="offset points", xytext=(-10,-10), ha='center', color='red')
plt.annotate('w1=1', (w1_1[0], w1_1[1]),
             textcoords="offset points", xytext=(-10,10), ha='center', color='green')

plt.xlabel('risk (sigma_p)')
plt.ylabel('return (mu_p)')
plt.title("feasible set of portfolios with R1 & R2")
plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------#
#      Q5.2: minimum variance weights      #
#------------------------------------------#

def mvp_weights(sigma_1, sigma_2, rho):
    d = sigma_1**2 + sigma_2**2 - 2*rho*sigma_1*sigma_2
    w1 = (sigma_2**2 - rho*sigma_1*sigma_2)/d

    return w1, 1-w1

def mvp_return_profile(mu_1, mu_2, sigma_1, sigma_2, rho):
    d = sigma_1**2 + sigma_2**2 - 2*rho*sigma_1*sigma_2
    mu_p = (mu_1 * sigma_2**2 + mu_2 * sigma_1**2 - (mu_1 + mu_2)*rho*sigma_1*sigma_2)/d
    sigma_p = np.sqrt((sigma_1**2 * sigma_2**2 * (1 - rho**2))/d)

    return mu_p, sigma_p

w1, w2 = mvp_weights(sigma_1, sigma_2, rho)
mu_p, sigma_p = mvp_return_profile(mu_1, mu_2, sigma_1, sigma_2, rho)

print("5.2: optimal weights")
print(f"w1: {w1:.4f}     w2: {w2:.4f}")
print()
print("5.3: expected return and std dev. of MVP")
print(f"mu_p: {mu_p:.4f}   sigma_p: {sigma_p:.4f}")
