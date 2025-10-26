import matplotlib.pyplot as plt
import numpy as np
import os

def dS(beta, S, I, N):
    return -beta * S * I / N
def dI(beta, S, I, N, gamma):
    return beta * S * I / N - gamma * I
def dR(gamma, I):
    return gamma * I

def simulate_sir(S0, I0, R0, beta, gamma, N, T_max, auto_stop=False):
    S, I, R = S0, I0, R0
    S_list, I_list, R_list = [S], [I], [R]
    I_peak = I0
    I_peak_time = 0
    if auto_stop:
        T_max = 100000000
    for t in range(T_max):
        dS_dt = dS(beta, S, I, N)
        dI_dt = dI(beta, S, I, N, gamma)
        dR_dt = dR(gamma, I)

        S += dS_dt
        I += dI_dt
        R += dR_dt

        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        if I > I_peak:
            I_peak = I
            I_peak_time = t + 1
        if auto_stop and I < 1:
            T_max = t + 1
            break
    print(f"Peak Infected: {I_peak} at time {I_peak_time}")
    return S_list, I_list, R_list, T_max, I_peak, I_peak_time

def plot_sir(I_list, T_max, filename):
    plt.figure()
    plt.plot(range(T_max + 1), I_list, label='Infected')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_heatmap(data_grid, x_values, y_values, title, x_label, y_label, cbar_label, filename):
    plt.figure(figsize=(10, 8))
    extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    im = plt.imshow(data_grid, aspect='auto', origin='lower', 
                    extent=extent, cmap='viridis')
    
    plt.colorbar(im, label=cbar_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)

if __name__ == "__main__":
    N = 137000
    I0 = 1
    R0 = 0
    S0 = N - I0 - R0
    beta = 2
    gamma = 1
    T_max = 100

    S_list, I_list, R_list, _, _, _ = simulate_sir(S0, I0, R0, beta, gamma, N, T_max)
    plot_sir(I_list, T_max, 'problem_set_3/sir_simulation_1.png')

    S_list, I_list, R_list, T_max, _, _ = simulate_sir(S0, I0, R0, beta, gamma, N, T_max, auto_stop=True)
    plot_sir(I_list, T_max, 'problem_set_3/sir_simulation_2.png')

    # Heatmap generation
    betas = np.linspace(1.0, 3.0, 50)
    gammas = np.linspace(0.5, 1.5, 50)
    peak_times_grid = np.zeros((len(gammas), len(betas)))
    peak_values_grid = np.zeros((len(gammas), len(betas)))
    heatmap_T_max = 1000

    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            _, _, _, _, I_peak, I_peak_time = simulate_sir(
                S0, I0, R0, 
                beta=b, 
                gamma=g, 
                N=N, 
                T_max=heatmap_T_max, 
                auto_stop=True
            )
            
            peak_times_grid[i, j] = I_peak_time
            peak_values_grid[i, j] = I_peak

    # Plot peak infection heatmap
    plot_heatmap(
        data_grid=peak_times_grid,
        x_values=betas,
        y_values=gammas,
        title='Time of Peak Infection vs. Beta and Gamma',
        x_label='Infection Rate (beta)',
        y_label='Recovery Rate (gamma)',
        cbar_label='Time',
        filename='problem_set_3/peak_time_heatmap.png'
    )
    # Plot peak infection value heatmap
    plot_heatmap(
        data_grid=peak_values_grid,
        x_values=betas,
        y_values=gammas,
        title='Peak Number of Infected Individuals vs. Beta and Gamma',
        x_label='Infection Rate (beta)',
        y_label='Recovery Rate (gamma)',
        cbar_label='Peak Infected Population',
        filename='problem_set_3/peak_value_heatmap.png'
    )