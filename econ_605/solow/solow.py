import matplotlib.pyplot as plt
import numpy as np


# Define a function that calculates time paths for the Solow model, using a Cobb-Douglas production function
def solow_cobb_douglas(alpha, n, s, d, g_A, K_0, L_0, A_0, T):
    # If they're scalar, convert parameters into time series (useful for parameter breaks)
    alpha_s = np.ones(T + 1) * alpha
    n_s = np.ones(T + 1) * n
    s_s = s * np.ones(T + 1)
    d_s = np.ones(T + 1) * d
    g_A_s = np.ones(T + 1) * g_A

    # Set up time series
    k = np.ones(T + 1) * K_0 / (A_0 * L_0)
    L = np.ones(T + 1) * L_0
    A = np.ones(T + 1) * A_0

    # Loop through time periods
    for t in range(T):
        # k_{t + 1} = G(K_{t})
        k[t + 1] = s_s[t] * k[t]**(alpha_s[t]) + (1 - g_A_s[t] - n_s[t] - d_s[t]) * k[t]

        # L_{t + 1} = (1 + n) L_{t}
        L[t + 1] = (1 + n_s[t]) * L[t]

        # A_{t + 1} = (1 + g_A) A_{t}
        A[t + 1] = (1 + g_A_s[t]) * A[t]

    # Back out other series
    K = k * (A * L)
    Y = K**(alpha_s) * (A * L)**(1 - alpha_s)
    C = (1 - s_s) * Y
    R = (alpha_s) * ((A * L) / K)**(1 - alpha_s)
    w = (1 - alpha_s) * A**(1 - alpha_s) * (K/L)**(alpha_s)
    k_star = (s_s / (n_s + g_A_s + d_s))**(1/(1 - alpha_s))

    # Return those series
    return k, K, L, A, Y, C, R, w, k_star

# Set plot options
plt.rc('text', usetex=True)  # Use LaTeX to compile text, which looks way better but also takes longer
plt.rc('font', size=11, **{'family':'serif',
       'serif':['lmodern', 'Palatino Linotype', 'DejaVu Serif']})  # Font size and type

# Set number of time periods
T = 100

# Vector of saving rates over time (changes at T/2)
s = np.ones(T + 1) * 0.05
s[int(np.ceil(T/2)):] = 0.10

# Vector of population growth rate over time (changes at T/2)
n = np.ones(T + 1) * 0.025
n[int(np.ceil(T/2)):] = 0.05

# Vector of technology growth rate over time (changes at T/2)
g_A = np.ones(T + 1) * 0.10
g_A[int(np.ceil(T/2)):] = 0.03

# Initial series
k, K, L, A, Y, C, R, w, k_star =\
    solow_cobb_douglas(alpha=0.3, n=n[0], s=s[0], d=0.15, g_A=g_A[0], K_0=1.4, L_0=1.0, A_0=0.2, T=T)

# Changed series
k1, K1, L1, A1, Y1, C1, R1, w1, k_star1 =\
    solow_cobb_douglas(alpha=0.3, n=n[0], s=s[0], d=0.15, g_A=g_A, K_0=1.4, L_0=1.0, A_0=0.2, T=T)

# Set up plot
fig, ax = plt.subplots()
ax.set_title('Dynamics of the Solow model with technological progress')

# Plot changing steady state k
ax.plot(k_star1, linestyle='--', color='0.4', label='$k*$')

# Plot changing series
ax.plot(np.log(Y1), 'r-', alpha=0.9, lw=0.9, label='$\ln(Y)$')
ax.plot(np.log(K1), 'm-', alpha=0.9, lw=0.9, label='$\ln(K)$')
ax.plot(np.log(w1), 'b-', alpha=0.9, lw=0.9, label='$\ln(w)$')
ax.plot(np.log(R1), 'g-', alpha=0.9, lw=0.9, label='$\ln(R)$')

# Plot unchanged counterfactual series
ax.plot(np.log(Y), 'r--', alpha=0.9, lw=0.9)
ax.plot(np.log(K), 'm--', alpha=0.9, lw=0.9)
ax.plot(np.log(w), 'b--', alpha=0.9, lw=0.9)
ax.plot(np.log(R), 'g--', alpha=0.9, lw=0.9)

# Plot break point
ax.axvspan(xmin=(np.floor(T/2) * 2 - 2)/2, xmax=(np.ceil(T/2) * 2 - 1)/2, color='0.9', alpha=0.5, zorder=3)

# Set axis limits and labels
ax.set_xlim(0, T)
ax.set_xlabel('Period')

# Include a legend and display the plot
ax.legend()
plt.show()
