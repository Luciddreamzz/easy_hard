import json
import numpy as np
from sacred import Experiment
from main import ex
from sklearn.model_selection import ParameterGrid
from pymongo import MongoClient

run = Experiment('multiple_run')

curr_db_name = 'easy_hard'
client = MongoClient('localhost', 27017)

db = client[curr_db_name]


# noinspection PyUnusedLocal
@run.config
def run_config():
	layers_sizes = np.arange(1, 101)
	repeats = 1


@run.automain
def run(layers_sizes, repeats):

	for update in layers_sizes:
		model_update = dict(
			layers_sizes=update
		)
		config_updates = dict(learning_model=model_update)
		ex.add_config(config_updates)

		for i in range(repeats):
			ex.run()


