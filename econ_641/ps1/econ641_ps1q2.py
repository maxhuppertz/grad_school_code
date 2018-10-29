# Set up model parameters
alpha = .5
w = 1
beta_A = .25
beta_B = .75
L_H = 5
L_F = 45
K_H = 180
K_F = 20

r = (
    (L_H + L_F) / (K_H + K_F)
    * ( alpha * beta_A + (1-alpha) * beta_B ) / ( alpha * (1-beta_A) + (1-alpha) * (1-beta_B) )
    )

print(r)
