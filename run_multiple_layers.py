from sacred import Experiment
from main_old import ex
from itertools import product
from pymongo import MongoClient

run = Experiment('multiple_run')

curr_db_name = 'easy_hard'
client = MongoClient('localhost', 27017)

db = client[curr_db_name]


# noinspection PyUnusedLocal
@run.config
def run_config():
	max_size = 25
	n_layers = 2
	repeats = 1


@run.automain
def run(max_size, n_layers, repeats):

	for item in filter(lambda x: is_sorted(x), product(*(range(1, max_size) for i in range(n_layers)))):
		model_update = dict(
			layers_sizes=item
		)
		config_updates = dict(learning_model=model_update)
		ex.add_config(config_updates)

		for i in range(repeats):
			ex.run()


def is_sorted(x):
	return list(x) == sorted(x)[::-1]



