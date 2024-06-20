import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iterations = 1000
num_points = 200

def vasicek_monte_carlo(x, r0, kappa, theta, sigma, T):
    dt = T/num_points
    result = []
    for i in range(iterations):
        rates = [r0]
        for j in range(num_points):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1]+dr)
        result.append(rates)

    simulation_data = pd.DataFrame(result)
    simulation_data = simulation_data.T
    integral_sum = simulation_data.sum()*dt
    #present value of future cash flow
    present_integral_sum = np.exp(-integral_sum)
    bond_price = x * np.mean(present_integral_sum)
    print('Bond price based on Vasiek Model and Monte-Carlo Simulation is: $', bond_price)

if __name__ == '__main__':
    vasicek_monte_carlo(1000,0.1,0.3,0.3,0.03,1)
