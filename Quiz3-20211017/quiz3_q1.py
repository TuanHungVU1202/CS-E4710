import numpy as np
import math
discr = 1000
x = np.linspace(-1, 1, discr)
noise = np.zeros(discr)
total_noise = 0
for i in range(0, discr):
    Pr1 = (1/2) * x[i] + 1/2
    Pr0 = -(1/2) * x[i] + 1/2
    if Pr1 < Pr0:
        noise[i] = Pr1
    else:
        noise[i] = Pr0
    total_noise += noise[i]
Expectation_noise = total_noise/discr
print(Expectation_noise)