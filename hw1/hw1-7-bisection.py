#------------------------------------------#
#           Q7: bisection method           #
#------------------------------------------#

def bond_price(y, C, m, n):
    price = 0
    for t in range(1, n + 1):
        price += (C/m) / (1 + y/m) ** t
    price += 1 / (1 + y/m) ** n
    return price

def bisection_method_bond_yield(C, m, n, P_market, x1, x2, tol=1e-4):

    def f(y):
        return bond_price(y, C, m, n) - P_market

    iteration = 0
    while abs( f((x1 + x2) / 2) ) > tol:
        x_mid = (x1 + x2) / 2

        print(
            f"Iteration {iteration+1}: Bracket = ({x1:.5f}, {x2:.5f}), Midpoint = {x_mid:.5f}, Error = {abs(f(x_mid)):.5f}")

        if f(x1) * f(x_mid) < 0:
            x2 = x_mid
        else:
            x1 = x_mid

        iteration += 1

    final_yield = (x1 + x2) / 2
    print(f"\nFinal yield = {final_yield*100:.3f}%")
    print(f"Final error =  {f(final_yield):.5f}")
    return final_yield

C = 0.02
m = 2
T = 5
n = m * T

P_market = 0.98
x1 = 0.0
x2 = 0.10

y = bisection_method_bond_yield(C, m, n, P_market, x1, x2)
print(f"Final bond price = {bond_price(y,C,m,n)*100:.3f}%")
