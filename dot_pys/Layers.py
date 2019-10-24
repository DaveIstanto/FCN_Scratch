#!/bin/python3

# This script is for custom neural net layers

import numpy as np 
import math

# Section for affine transformation layers

## Class for linear layer
class Linear:
	
	# Constructor: Requires in-dimension and out-dim
	# Makes weight matrix and bias
	def __init__(self, in_dim, out_dim, lr):
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.limit = math.sqrt(6 / (in_dim + out_dim))
		self.wt = np.random.uniform(low = -1 * self.limit, high = self.limit, size = (out_dim, in_dim) )
		#self.wt = np.random.randn(out_dim, in_dim) * math.sqrt((2 / in_dim)) # Weight matrix using random values: square root of (2/ni). Where ni is number of input units for that layer.
		self.b = np.zeros(out_dim)# Bias vector (column vector)
		self.lr = lr # Learning rate

	# Forward function: 
	# input shape -> Column vector with in_dim dimensional (in_dim)
	# output shape -> Column vector with out_dim dimensional (out_dim)
	def forward(self, input_vector):
		return np.matmul(self.wt, input_vector) + self.b

	# Backward Function:
	# inputs -> gradient (g): from previous layer (column vector)
	# 			h(k-1): values from input of this layer
	# outputs -> gradient (g)
	def backward(self, g, h_in):
		re_g = g.reshape(len(g), 1) # reshape g for matrix multiplication
		re_h_in = h_in.reshape(1, len(h_in)) # reshape h_in

		del_b = g # For updating b
		del_wt = np.matmul(re_g, re_h_in) # For updating wt
		del_h_in = np.matmul(np.transpose(self.wt), re_g).reshape((len(h_in),)) # For derivative of del_h_in, becomes the new g

		# Set self biases and weights
		self.b = self.b - self.lr * del_b
		self.wt = self.wt - self.lr * del_wt
		
		return del_h_in

# Section for activation layers

## Class for ReLU (Rectified Linear Units)
class Relu:
	# Constructor: No requirement, as this class outputs the same dimension as input
	def __init__(self):
		pass

	# Forward function:
	# input shape -> Column vector (in_dim)
	# output shape -> Column vector (in_dim)
	def forward(self, input_vector):
		# Get element-wise max {0, input_vector_i}
		zero_vector = np.zeros((input_vector.shape[0])) # Make zero vector with same dim as input
		return np.maximum(zero_vector, input_vector)

	# Backward function
	# input -> Column vector, the values of post-activation
	def backward(self, g, post_act):
		# For each element in post_act, if element > 0, gradient is element, else, gradient is 0.0000001
		return g * np.where(post_act <= 0, 0.0000001, post_act)

## Class for sigmoid 
class Sigmoid:
	# Constructor : None
	def __init__(self):
		pass
	
	def overf_prot_forward(self, x):
		if x < 0:
			a = math.exp(x) 
			return a / (1 + a) 
		else:
			return 1 / (1 + math.exp(-1 * x))

	# Forward function:
	# input shape -> Column vector (in_dim)
	# output shape -> Column vector (in_dim)
	def forward(self, input_vector):
		return np.vectorize(self.overf_prot_forward)(input_vector)
	
	# Backward function: Get derivative
	# input: gradient vector from next layer, the output of the forward function (post activation vector)
	# output: gradient
	def backward(self, g, post_act):
		#return np.maximum(np.ones(post_act.shape[0]) * 1e-9, g * np.multiply(post_act, (1 - post_act)))
		return g * np.multiply(post_act, (1 - post_act))


## Class for MSE
class MSE:
	# Constructor: None
	def __init__(self):
		pass

	# Forward function:
	# input: prediction vector (yhat), actual value vector (y)
	# output: mean squared error (scalar)
	def forward(self, yhat, y):
		return np.sum(np.square(np.subtract(yhat, y))) / len(yhat)

	# Backward function:
	# input: prediction vector (yhat), actual value vector (y)
	# output: vector with size (yhat) (this is the gradient vector of y w.r.t yhat_i)
	def backward(self, yhat, y):
		first_term = np.multiply((2 / len(yhat)), yhat)
		second_term = np.multiply((-2 / len(yhat)), y)
		return first_term + second_term

	

## Class for Softmax and Cross entropy loss
class SM_CEL:

	# Constructor: None
	def __init__(self):
		pass

	# Forward Function:
	# input shape -> Column vector
	# output shape -> Scalar
	def forward_SM(self, input_vector):
		# Get sum of exponent for input vector (normalized version so no overflow)
		e = np.exp(input_vector - np.max(input_vector))
		yHat = np.divide(e, np.sum(e))
		
		return np.maximum(yHat, np.ones(yHat.shape[0]) * 1e-9)
		
	
	def forward_CEL(self, yHat, y):
		# Returns according to cross entropy loss
		return (-1 * np.sum(np.multiply(y, np.log(yHat))))
        
	# Backward function: Gets gradient of SM_CEL w.r.t pre-nonlinearity activation (Page 206)
	# input shape -> yHat (column vector), y (column vector, one hot encoded)
	# output shape -> column vector (derivatives of cross entropy loss w.r.t each member of pre-nonlinearity activation vector)
	def backward(self, yHat, y):
		return np.subtract(yHat, y)

