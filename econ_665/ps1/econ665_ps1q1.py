import numpy as np

r_min = 0
r_max = 20
r_num = 211
r = np.array(np.linspace(r_min, r_max+1, r_num), ndmin=2)

s_min = 0
s_max = 25
s_num = s_max + 1
s = np.array(np.linspace(s_min, s_max, s_num), ndmin=2)

beta_min = ( (r.transpose() @ np.ones(r.shape)) @ np.exp( r.transpose() @ s ) ) / 1 + r.transpose() @ s
