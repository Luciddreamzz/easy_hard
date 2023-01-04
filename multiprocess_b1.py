import os
from concurrent import futures
from sacred.observers import FileStorageObserver, MongoObserver
from main_2 import ex
from itertools import product
from pymongo import MongoClient

file_system_path = os.path.abspath('../output')
mongodb_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'easy_hard'

# define hyperparameters
max_workers = 6
repeats = 1

configs = [[5, 5], [15, 10], [20, 15], [18, 18], [25, 20], [30, 30]]


def exp(config):
	if not ex.observers:
		# ex.observers.append(MongoObserver(url=mongodb_url, db_name=curr_db_name))
		ex.observers.append(FileStorageObserver(file_system_path))  # instead onde_drive_path

	model_update = dict(
		layers_sizes=config
	)
	training_update = dict(training_epochs=1000000, batch_size=1)
	config = dict(learning_model=model_update, training=training_update)
	run = ex.run(config_updates=config)
	return run.result


def is_sorted(x):
	return list(x) == sorted(x)[::-1]


if __name__ == "__main__":

	for _ in range(repeats):
		for chunk in [configs[i:i + max_workers] for i in range(0, len(configs), max_workers)]:
			with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
				tasks = [executor.submit(exp, config) for config in chunk]
