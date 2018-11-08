import numpy as np

r_min = 0
r_max = 20
r_step = .01
r = np.array(np.linspace([x for x in range(r_min, r_max+1, r_step)]), ndmin=2)

s_min = 0
s_max = 25
s_step = 1
s = np.array(np.linspace([x for x in range(s_min, s_max+1, s_step)]), ndmin=2)

beta_min_denom = r.transpose() @ s
