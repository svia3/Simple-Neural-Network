
"""

This is a simple neural network that was deduced from the youtube video: 
https://www.youtube.com/watch?v=kft1AJ9WVDk

I am using this to understand the basics for neural networks and will be building some projects
later on that will incorporate more complicated structures, as well as teach myself tensorflow.

"""


import numpy as np

class SimpleNetwork():

	def __init__(self):
		"""
		1. take the inputs from the training example and put them through formula to get output
		2. calculate error: difference between the hypothesis - expected 
		3. adjust weights according to 'serverity' of error
		"""

		self.training_inputs = np.array([[0,0,1],
							 			 [1,1,1],
							 		 	 [1,0,1],
							 			 [0,1,1]])

		self.training_outputs = np.array([[0,1,1,0]]).T

		np.random.seed(1)
		self.weights = 2 * np.random.random((3,1)) - 1	# - 1 to 1 

		print("Random Synaptic Weights:")
		print(self.weights)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient_descent(self, x):
		return x * (1 - x)
		
	def train_network(self):
		for i in range(500000):
			input_layer = self.training_inputs
			outputs = self.sigmoid(np.dot(input_layer, self.weights)) 
			cost = self.training_outputs - outputs
			adjustments = cost * self.gradient_descent(outputs)
			self.weights += np.dot(input_layer.T, adjustments)

		print("Synaptic Weights After Training:")
		print(self.weights)

		print("Outputs After Training:")
		print(outputs)


if __name__=='__main__':
	net = SimpleNetwork()
	net.train_network()