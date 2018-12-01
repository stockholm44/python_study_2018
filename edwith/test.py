import numpy as np

a = np.arange(20,40).reshape(1,-1)
a
# a[a>22]
a
np.where(a>20)
np.where(a>25, 1, 0)

X = np.arange(5,12)
eval(str("X") + ">10")
eval
