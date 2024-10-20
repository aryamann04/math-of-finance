#------------------------------------------#
#     Q7.1: calculating vectors a and b    #
#------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

# given asset parameters

mu = np.array([0.05, 0.06, 0.03])
sigma = np.array([0.06, 0.08, 0.10])
rho = np.array([[1, 0.95, 0.9], [0.95, 1, 0.8], [0.9, 0.8, 1]])
u = np.ones(3)

# convert correlation matrix to covariance

C = np.outer(sigma, sigma) * rho
C_inv = np.linalg.inv(C)

# build and calculate D

muT_Cinv_mu = mu.T @ C_inv @ mu
muT_Cinv_u = mu.T @ C_inv @ u
uT_Cinv_mu = u.T @ C_inv @ mu
uT_Cinv_u = u.T @ C_inv @ u

D_inv = np.array([[muT_Cinv_mu, muT_Cinv_u], [uT_Cinv_mu, uT_Cinv_u]])
D = np.linalg.inv(D_inv)

# calculate vectors a and b

a = D[0, 0] * (C_inv @ mu) + D[1, 0] * (C_inv @ u)
b = D[0, 1] * (C_inv @ mu) + D[1, 1] * (C_inv @ u)

print("C:", C)
print("\nD:", D)
print("\na:", a)
print("\nb:", b)

#------------------------------------------#
#           Q7.2: MVF at mu = 6%           #
#------------------------------------------#

w_6 = a * 0.06 + b
print("\nw(6%):", w_6)
print(np.dot(w_6, mu))
print(np.sqrt(w_6.T @ C @ w_6))


#------------------------------------------#
#     Q7.3: MVF graph for mu = 0% - 12%    #
#------------------------------------------#

mu_values = np.linspace(0.00,0.12,100)
sigma_values = []

for mu in mu_values:
    w = a * mu + b
    sigma_values.append(np.sqrt(np.dot(np.dot(w.T, C), w)))

sigma_values = np.array(sigma_values)

plt.figure(figsize=(8,6))
plt.plot(sigma_values, mu_values, label='mvf', color='blue')

plt.xlabel('risk (sigma_p)')
plt.ylabel('return (mu_p)')
plt.title("mvf for 0% < mu < 12%")

#------------------------------------------#
#          Q7.4: market portfolio          #
#------------------------------------------#

# given asset parameters

mu = np.array([0.05, 0.06, 0.03])
sigma = np.array([0.06, 0.08, 0.10])
rho = np.array([[1, 0.95, 0.9], [0.95, 1, 0.8], [0.9, 0.8, 1]])
u = np.ones(3)

C = np.outer(sigma, sigma) * rho
C_inv = np.linalg.inv(C)

r_0 = 0.03  # risk-free rate

# calculate market portfolio weights via formula (3.11)

x = mu - r_0 * u
w_m = (np.dot(C_inv, x)) / (np.dot(u.T, np.dot(C_inv, x)))

mu_m = np.dot(w_m.T, mu)
sigma_m = np.sqrt(np.dot(np.dot(w_m.T, C), w_m))

cml = (mu_m - r_0)/ sigma_m  # slope that connects market portfolio to r_0

print(f"\nweights: {w_m}\n")
print(f"\nmu_m: {mu_m*100:.2f}%\n")
print(f"\nsigma_m: {sigma_m*100:.2f}%\n")
print(f"\ncml slope: {cml:.4f}")


# for 7.3
# add market portfolio (red point)
plt.scatter(sigma_m, mu_m, color='red', zorder=5)

# add cml line
sigma_cml = np.linspace(0, max(sigma_values), 100)
mu_cml = r_0 + cml * sigma_cml
plt.plot(sigma_cml, mu_cml, label='cml', color='black', linestyle='--')

plt.grid(True)
plt.legend()
plt.show()
