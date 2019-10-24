#!/bin/python3

import Layers as lys
import numpy as np

x_1 = np.array([1, 4])
x_2 = np.array([2, 5])

lin_1 = lys.Linear(2, 5, 0.001)
relu = lys.Relu()
h1 = relu.forward(lin_1.forward(x_1))
lin_2  = lys.Linear(h1.shape[0], 2, 0.001)
sm_cel = lys.SM_CEL()

y = np.array([1, 0])
h2 = sm_cel.forward_SM(lin_2.forward(h1))

print(h2)



# FORWARD TESTING SUCCESSFUL

# Start Backward prop testing

g_1 = sm_cel.backward(h2, y)

g_2 = lin_2.backward(g_1, h1)

g_3 = relu.backward(h1)

g_4 = lin_1.backward(g_3, x_1)

print(g_4)

# BACKPROP SEEMS GOOD