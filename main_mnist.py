from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import tensorflow.contrib.slim as slim
import os
import tempfile
import pickle

import warnings
import numpy as np
import tensorflow as tf
import mnist_data
from copy import deepcopy
from sacred import Experiment
from sklearn.model_selection import train_test_split
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver, MongoObserver

from models import Mlp
from data_classes import Circular, Dataset

 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

tf.config.optimizer.set_jit(True)


# defining the sacred experiment
ex = Experiment('easy_hard', save_git_info=False)

print('Check experiment')

# setting the path for the output files
file_system_path = os.path.abspath('../output')

mongodb_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'easy_hard'
ex.captured_out_filter = apply_backspaces_and_linefeeds

# adding observers to the experiment
# ex.observers.append(MongoObserver(url=mongodb_url, db_name=curr_db_name))

ex.observers.append(FileStorageObserver(file_system_path))
@ex.config  # config function is used to define the hyper parameters of the experiment
def config():		
    training = dict(
	    opt='Adam',
	    learning_rate=0.001, 
	    training_epochs=10,
	    batch_size = 20,
	    valid_size = 0.2)
		#batch_size=round(sum(class_['size'] for class_ in dataset['classes'])/100))
    learning_model = dict(
	    model_type='Mlp',
	    layers_sizes=[30, 30],
	    activation='linear',
	    loss='binary_crossentropy',
	    initialization='he_uniform')


@ex.capture
def load_model(learning_model):

	#n_classes = len(set(x['_class'] for x in dataset['classes']))
	n_classes = 11
	models_dict = {'Mlp': Mlp}
	if isinstance(learning_model['layers_sizes'], int):
		layers_sizes = (learning_model['layers_sizes'],)
	else:
		layers_sizes = learning_model['layers_sizes']
	print(layers_sizes)
	model = models_dict[learning_model['model_type']](
		n_classes,
		learning_model['activation'],
		learning_model['initialization'],
		*layers_sizes)

	return model


@ex.capture(prefix='train_data')
#return x_train, x_valid, x_test, y_train, y_valid, y_test


# returns a random batch of the training set
@ex.capture(prefix='training')
def random_batch(x, y, batch_size):
	idx = np.random.randint(len(x), size=batch_size)
	return x[idx], y[idx]


@ex.automain
def run(_run, training, learning_model):

	training_epochs = training['training_epochs']
	batch_size = training['batch_size']

	# loading data
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(True)

	# Prepare the training dataset.
	train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

	# Prepare the validation dataset.
	val_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
	val_dataset = val_dataset.batch(batch_size)

	# Prepare the test dataset.
	test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
	test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

	# loading the compiled model
	model = load_model()

	# instantiate optimizer
	optimizer = tf.keras.optimizers.get(training['opt'])
	optimizer.learning_rate.assign(training['learning_rate'])

	# getting model loss function
	loss_fn = tf.keras.losses.get(learning_model['loss'])

	# Prepare the metrics.
	train_acc_metric = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
	val_acc_metric = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
	test_acc_metric = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")

	print("Training the model...")

	weights_dict = {f'layer_{i}': {'weights': [], 'bias': []} for i in range(len(model.layers))}
	gradients_dict = {f'layer_{i}': {'weights': [], 'bias': []} for i in range(len(model.layers))}
	training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

	@tf.function(jit_compile=True)
	def train_step(x, y):
		with tf.GradientTape() as tape:
			logits = model(x, training=True)
			loss_value = loss_fn(y, logits)
		grads = tape.gradient(loss_value, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		train_acc_metric.update_state(y, logits)
		return loss_value, grads, model.trainable_variables

	@tf.function(jit_compile=True)
	def test_step(x, y, metric):
		val_logits = model(x, training=False)
		metric.update_state(y, val_logits)
		
	"""
	# save weights and biases before training
	for idx in range(len(model.layers)):
		weights_dict[f'layer_{idx}']['weights'].append(model.layers[idx].get_weights()[0].tolist())
		weights_dict[f'layer_{idx}']['bias'].append(model.layers[idx].get_weights()[1].tolist()) """

	for epoch in range(1, training_epochs + 1):
		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss, gradients, weights = train_step(x_batch_train, y_batch_train)

			# save loss in training history
			training_history['loss'].append(loss.numpy().mean())

			# updating gradients history
			for idx in range(len(model.layers)):
				gradients_dict[f'layer_{idx}']['weights'].append(gradients[idx+0].numpy())
				gradients_dict[f'layer_{idx}']['bias'].append(gradients[idx+1].numpy())

			# updating weights history
			for idx in range(len(model.layers)):
				weights_dict[f'layer_{idx}']['weights'].append(weights[idx+0].numpy())
				weights_dict[f'layer_{idx}']['bias'].append(weights[idx+1].numpy())

		# Evaluate accuracy at the end of each epoch
		train_acc = train_acc_metric.result()
		# add accuracy value to training history
		training_history['accuracy'].append(train_acc.numpy())
		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()

		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in val_dataset:
			test_step(x_batch_val, y_batch_val, val_acc_metric)

		# Evaluate val accuracy
		val_acc = val_acc_metric.result()
		# Add val accuracy to training history
		training_history['val_accuracy'].append(val_acc.numpy())
		# reset metric state
		val_acc_metric.reset_states()

		# Evaluate val loss
		y_pred = model(validation_data, training=False)
		val_loss = loss_fn(validation_labels, y_pred).numpy().mean()
		# Add val loss to training history
		training_history['val_loss'].append(val_loss)

	# Run a test loop at the end of training
	for x_batch_test, y_batch_test in test_dataset:
		test_step(x_batch_test, y_batch_test, test_acc_metric)
	# Evaluate test accuracy
	test_acc = test_acc_metric.result().numpy()
	test_acc_metric.reset_states()

	# Evaluate test loss
	y_pred = model(test_data, training=False)
	test_loss = loss_fn(test_labels, y_pred).numpy().mean()

	results = {
		'train_acc': training_history['accuracy'][-1].item(), 'train_loss': training_history['loss'][-1].item(),
		'val_acc': training_history['val_accuracy'][-1].item(), 'val_loss': training_history['val_loss'][-1].item(),
		'test_acc': test_acc.item(), 'test_loss': test_loss.item()
		}

	_run.info.update(results)

	print("Saving artifacts..."+str(test_acc))

	with tempfile.TemporaryDirectory() as tmpdir:

		train_history_path = os.path.join(tmpdir, 'training_history.pkl')
		weights_history_path = os.path.join(tmpdir, 'weights_history.pkl')
		gradients_history_path = os.path.join(tmpdir, 'gradients_history.pkl')

		with open(train_history_path, 'wb') as handle1, open(weights_history_path, 'wb') as handle2, \
			open(gradients_history_path, 'wb') as handle3:

			pickle.dump(training_history, handle1)
			pickle.dump(weights_dict, handle2)
			pickle.dump(gradients_dict, handle3)

		ex.add_artifact(train_history_path)
		ex.add_artifact(weights_history_path)
		ex.add_artifact(gradients_history_path)

	print("Artifacts saved...")

	return "success" if test_acc >= 1 else "fail"
