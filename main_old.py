import os
import numpy as np
import tempfile
import json
import tensorflow as tf
import math

from copy import deepcopy
from sklearn.model_selection import train_test_split
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tensorflow import keras
from models import Mlp
from data_classes import Circular, Dataset
from utils import plot_decision_boundaries

print('All import OK')

# defining the sacred experiment
ex = Experiment(save_git_info=False)
ex = Experiment('easyhard',save_git_info=False)
print('Check experiment')

print(np.int)



# setting the path for the output files
file_system_path = os.path.abspath('../output')
onedrive_path = os.path.join(os.path.expanduser('~'), 'OneDrive - Università degli Studi di Catania/easy_hard/output')


mongodb_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'easy_hard'
ex.captured_out_filter = apply_backspaces_and_linefeeds

# adding observers to the experiment
#ex.observers.append(MongoObserver(url=mongodb_url, db_name=curr_db_name))
# ex.observers.append(FileStorageObserver(onedrive_path))
ex.observers.append(FileStorageObserver(file_system_path))


# noinspection PyUnusedLocal
@ex.config  # config function is used to define the hyper parameters of the experiment
def config():

	dataset = {
		'classes': [
			{'type': 'Circular', '_class': 0, 'data_dim': 2, 'ray': 1, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 1, 'data_dim': 2, 'ray': 2, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 0, 'data_dim': 2, 'ray': 3, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 1, 'data_dim': 2, 'ray': 4, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 0, 'data_dim': 2, 'ray': 5, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 1, 'data_dim': 2, 'ray': 6, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 0, 'data_dim': 2, 'ray': 7, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 1, 'data_dim': 2, 'ray': 8, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 0, 'data_dim': 2, 'ray': 9, 'noise': 0, 'size': 1250},
			{'type': 'Circular', '_class': 1, 'data_dim': 2, 'ray': 10, 'noise': 0, 'size': 1250},

		],
		'train_ratio': 0.7,
		'test_ratio': 0.15,
		'val_ratio': 0.15,
		'sampling_seed': np.random.randint(0, 2**31-1)}
	training = dict(
		opt='Adam',
		learning_rate=0.001,
		batch_size=round(sum(class_['size'] for class_ in dataset['classes'])/100),
		training_epochs=10
		# round(sum(class_['size'] for class_ in dataset['classes'])/100))
		# round(sum(class_['size'] for class_ in dataset['classes'])*dataset['train_ratio'])
	)
	learning_model = dict(
		model_type='Mlp',
		layers_sizes=100,
		activation='relu',
		loss='binary_crossentropy',
		initialization='he_uniform')


@ex.capture
def load_model(learning_model, dataset, training):

	n_classes = len(set(x['_class'] for x in dataset['classes']))
	models_dict = {'Mlp': Mlp}
	if isinstance(learning_model['layers_sizes'], int):
		layers_sizes = (learning_model['layers_sizes'],)
	else:
		layers_sizes = learning_model['layers_sizes']
	# instantiate optimizer
	optimizer = tf.keras.optimizers.get(training['opt'])
	optimizer.learning_rate.assign(training['learning_rate'])

	model = models_dict[learning_model['model_type']](
		n_classes,
		learning_model['activation'],
		learning_model['initialization'],
		*layers_sizes
		)

	model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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


# @ex.capture
# def log_metrics(_run, logs, model):

#	_run.log_scalar("loss", float(logs.get('loss')))
#	_run.log_scalar("accuracy", float(logs.get('accuracy')))
#	_run.log_scalar("val_loss", float(logs.get('val_loss')))
#	_run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
#	_run.log_scalar("weights_0", model.layers[0].get_weights()[0].tolist())
#	_run.log_scalar("bias_0", model.layers[0].get_weights()[1].tolist())
#	_run.log_scalar("weights_1", model.layers[1].get_weights()[0].tolist())
#	_run.log_scalar("bias_1", model.layers[1].get_weights()[1].tolist())
#	_run.result = float(logs.get('accuracy'))


class WeightsCallback(keras.callbacks.Callback):

	def __init__(self, model):
		super(WeightsCallback, self).__init__()
		self.model = model
		self.weight_dict = {f'layer_{i}': {'weights': [], 'bias': []} for i in range(len(self.model.layers))}

	def on_epoch_end(self, batch, logs={}):
		for idx in range(len(self.model.layers)):
			self.weight_dict[f'layer_{idx}']['weights'].append(self.model.layers[idx].get_weights()[0].tolist())
			self.weight_dict[f'layer_{idx}']['bias'].append(self.model.layers[idx].get_weights()[1].tolist())

	def on_train_end(self, logs={}):
		print("Saving weights history...")
		with tempfile.TemporaryDirectory() as tempdir:
			path = os.path.join(tempdir, 'weights_history.json')
			with open(path, 'w') as dump_file:
				json.dump(self.weight_dict, dump_file, indent=4)
			ex.add_artifact(path)


@ex.automain
def run(_run, training):

	training_epochs = training['training_epochs']
	batch_size = training['batch_size']

	# loading data
	x_train, x_val, x_test, y_train, y_val, y_test = load_data()

	# loading the compiled model
	model = load_model()

	# defining Callbacks
	# log_callback = callbacks.LambdaCallback(on_epoch_end=lambda _, logs: log_metrics(logs=logs, model=model))
	weights_callback = WeightsCallback(model)

	print("Training the model...")

	# fitting model to training data
	training_history = model.fit(
		x_train, y_train, epochs=training_epochs, batch_size=batch_size,
		validation_data=(x_val, y_val), callbacks=weights_callback, verbose=0)

	train_score = model.evaluate(x_train, y_train, verbose=1)
	val_score = model.evaluate(x_val, y_val, verbose=1)
	test_score = model.evaluate(x_test, y_test, verbose=1)
	results = {
		'train_acc': train_score[1], 'train_loss': train_score[0],
		'val_acc': val_score[1], 'val_loss': val_score[0],
		'test_acc': test_score[1], 'test_loss': test_score[0]
		}

	_run.info.update(results)

	with tempfile.TemporaryDirectory() as tmpdir:

		print("Saving training history...")

		file_path = os.path.join(tmpdir, 'training_history.json')

		with open(file_path, 'w') as handle:
			json.dump(training_history.history, handle, indent=4)

		ex.add_artifact(file_path)

		# print("Producing decision boundaries...")
		# fig = plot_decision_boundaries(x_test, y_test, model)
		# fig_path = os.path.join(tmpdir, 'decision_boundaries.png')
		# fig.savefig(fig_path)
		# ex.add_artifact(fig_path)
		# plt.close(fig)"""

	return "success" if model.evaluate(x_test, y_test)[1] >= 1 else "fail"
