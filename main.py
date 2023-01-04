import os
import tempfile
import pickle

import warnings
import numpy as np

import tensorflow as tf
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


# noinspection PyUnusedLocal
@ex.config  # config function is used to define the hyper parameters of the experiment
def config():

	dataset = dict(
		classes=[
			           {'type': 'Circular', '_class': 0, 'data_dim': 5, 'ray': 1, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 1, 'data_dim': 5, 'ray': 2, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 0, 'data_dim': 5, 'ray': 3, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 1, 'data_dim': 5, 'ray': 4, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 0, 'data_dim': 5, 'ray': 5, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 1, 'data_dim': 5, 'ray': 6, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 0, 'data_dim': 5, 'ray': 7, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 1, 'data_dim': 5, 'ray': 8, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 0, 'data_dim': 5, 'ray': 9, 'noise': 0, 'size': 1250},
			           {'type': 'Circular', '_class': 1, 'data_dim': 5, 'ray': 10, 'noise': 0, 'size': 1250},

		           ],
		train_ratio=0.7,
		test_ratio=0.15,
		val_ratio=0.15,
		sampling_seed=np.random.randint(0, 2**31-1))
		
	training = dict(
		opt='Adam',
		learning_rate=0.001,
		training_epochs=1000,
		batch_size=round(sum(class_['size'] for class_ in dataset['classes'])/100))
	learning_model = dict(
		model_type='Mlp',
		layers_sizes=[30, 30],
		activation='linear',
		loss='binary_crossentropy',
		initialization='he_uniform')


@ex.capture
def load_model(learning_model, dataset):

	n_classes = len(set(x['_class'] for x in dataset['classes']))
	models_dict = {'Mlp': Mlp}
	if isinstance(learning_model['layers_sizes'], int):
		layers_sizes = (learning_model['layers_sizes'],)
	else:
		layers_sizes = learning_model['layers_sizes']

	model = models_dict[learning_model['model_type']](
		n_classes,
		learning_model['activation'],
		learning_model['initialization'],
		*layers_sizes)

	return model


@ex.capture(prefix='dataset')
def load_data(classes, train_ratio, val_ratio, test_ratio, sampling_seed, _seed):

	class_dict = {'Circular': Circular}

	# copying the list of classes configuration in the function scope
	classes_copy = deepcopy(classes)

	# creating the list containing the class objects
	class_list = []
	for idx, class_ in enumerate(classes_copy):
		class_type = class_.pop('type')
		class_instance = class_dict[class_type](**class_, seed=sampling_seed+idx)
		class_list.append(class_instance)

	# creating DataHolder object
	data_holder = Dataset(class_list)
 
	# generating data points for each fine class
	data_holder.generate_classes()

	# stacking data points belonging to each fine class in a single data set
	X, y = data_holder.output_data()

	# splitting data set in training, validation and test set
	x_train, x_test, y_train, y_test = \
		train_test_split(X, y, stratify=y, test_size=1-train_ratio, random_state=sampling_seed)

	x_valid, x_test, y_valid, y_test = \
		train_test_split(
			x_test, y_test, stratify=y_test,
			test_size=test_ratio/(test_ratio + val_ratio), random_state=sampling_seed)

	return x_train, x_valid, x_test, y_train, y_valid, y_test


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
	x_train, x_val, x_test, y_train, y_val, y_test = load_data()

	# Prepare the training dataset.
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

	# Prepare the validation dataset.
	val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
	val_dataset = val_dataset.batch(batch_size)

	# Prepare the test dataset.
	test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
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
		y_pred = model(x_val, training=False)
		val_loss = loss_fn(y_val, y_pred).numpy().mean()
		# Add val loss to training history
		training_history['val_loss'].append(val_loss)

	# Run a test loop at the end of training
	for x_batch_test, y_batch_test in test_dataset:
		test_step(x_batch_test, y_batch_test, test_acc_metric)
	# Evaluate test accuracy
	test_acc = test_acc_metric.result().numpy()
	test_acc_metric.reset_states()

	# Evaluate test loss
	y_pred = model(x_test, training=False)
	test_loss = loss_fn(y_test, y_pred).numpy().mean()

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
