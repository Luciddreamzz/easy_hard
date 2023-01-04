import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input

# tf.config.optimizer.set_jit(True)


class Mlp(tf.keras.Model):

	def __init__(self, n_classes, activation, initializer, *layers_sizes):
		super(Mlp, self).__init__()
		if isinstance(layers_sizes, int):
			self.layers_sizes = (layers_sizes,)
		else:
			self.layers_sizes = layers_sizes
		self.dense_layers = [tf.keras.layers.Dense(
			units=u, activation=activation, kernel_initializer=initializer)
			for u in self.layers_sizes]
		self.out = tf.keras.layers.Dense(n_classes - 1, activation='sigmoid')
		self.n_classes = n_classes

	def call(self, inputs, **kwargs):
		x = inputs
		for layer in self.dense_layers:
			x = layer(x)
		return self.out(x)

	def predict_label(self, X):
		return np.round(self.predict(X))

	def get_config(self):
		return {"n_coarse_classes": self.n_coarse_classes, "layers_sizes": self.layers_sizes}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
