from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

class Brain(object):
	def __init__(self, learning_rate = 0.001, number_actions = 5):
		self.learning_rate = learning_rate

		# INPUT LAYER FORMED BY INPUT STATES
		states = Input(shape = (3,))

		# TWO HIDDEN LAYERS TOTALLY CONNECTED
		x = Dense(units = 64, activation = 'sigmoid')(states)
		x = Dropout(rate = 0.1)(x) #during training in each iteration 10% of neurons will be randomly turned off
		y = Dense(units = 32, activation = 'sigmoid')(x)
		y = Dropout(rate = 0.1)(y) #during training in each iteration 10% of neurons will be randomly turned off

		# OUTPUT LAYER TOTALLY CONNECTED TO THE LAST HIDDEN LAYER
		q_values = Dense(units = number_actions, activation = 'softmax')(y)

		# ENSAMBLAR LA ARQUITECTURA COMPLETA EN UN MODELO DE KERAS 
		self.model = Model(inputs = states, outputs = q_values)

		# COMPILING THE MODEL WITH THE MEAN SQUARE ERROR LOSS FUNCTION(MSE) AND THE OPTIMIZER (Adam)
		self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

