# Set up model parameters
alpha = .5
w = 1  # Normalization
beta_A = .25
beta_B = .75
L_H = 5
L_F = 45
K_H = 180
K_F = 20

# Calculate equilibrium interest rate
r = (
    (L_H + L_F) / (K_H + K_F)
    * ( alpha * beta_A + (1-alpha) * beta_B ) / ( alpha * (1-beta_A) + (1-alpha) * (1-beta_B) )
    )

# Calculate output prices
p_A = (r / beta_A)**beta_A * (1-beta_A)**(beta_A-1)
p_B = (r / beta_B)**beta_A * (1-beta_B)**(beta_B-1)

# Calculate output levels
Y_A = alpha * ( (L_H + L_F + r*(K_H + K_F)) / p_A )
Y_B = (1-alpha) * ( (L_H + L_F + r*(K_H + K_F)) / p_B )

# Calculate factor allocations
L_A = Y_A * ( r * (1-beta_A) / beta_A )**beta_A
L_B = L_H + L_F - L_A
K_A = Y_A * ( r * (1-beta_A) / beta_A )**(beta_A-1)
K_B = K_H + K_F - K_A


print('r =', r, '; L_A =', L_A, 'L_B =', L_B, 'K_A =', K_A, 'K_B =', K_B)
