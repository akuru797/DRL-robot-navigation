import numpy as np
import sys

file = str(sys.argv[1])
print(file)
x = np.load(file)
print(x)
