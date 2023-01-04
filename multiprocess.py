import os
from concurrent import futures
from sacred.observers import FileStorageObserver, MongoObserver
from main_2 import ex
from itertools import product
from pymongo import MongoClient
import time

file_system_path = os.path.abspath('../output')
mongodb_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'easy_hard'

# define hyperparameters
min_sizes = 1, 1
max_size = 100
n_layers = 2
max_workers = 6
repeats = 30
dim_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]



def exp(config):
	if not ex.observers:
		# ex.observers.append(MongoObserver(url=mongodb_url, db_name=curr_db_name))
		ex.observers.append(FileStorageObserver(file_system_path))  # instead onde_drive_path

	model_update = dict(
		layers_sizes=config
	)
	config = dict(learning_model=model_update)
	run = ex.run(config_updates=config)
	return run.result


def is_sorted(x):
	return list(x) == sorted(x)[::-1]

def multi_and(iterable, idx):
    if idx == len(iterable)-1:
        return iterable[idx]
    return iterable[idx] and multi_and(iterable, idx+1)


if __name__ == "__main__":

	t1 = time.time()
	for _ in range(repeats):
		if n_layers == 1:
			configs = list(range(1, max_size+1))
			configs = list(filter(lambda x: multi_and([x[i] in dim_list for i in range(n_layers)],0), configs))
		else:
			configs = list(filter(lambda x: is_sorted(x), (list(tup) for tup in product(*(range(s, max_size+1) for s, i
			                                                                              in zip(min_sizes, range(n_layers)))))))
			configs = list(filter(lambda x: multi_and([x[i] in dim_list for i in range(n_layers)],0), configs))


		for chunk in [configs[i:i + max_workers] for i in range(0, len(configs), max_workers)]:
			with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
				tasks = [executor.submit(exp, config) for config in chunk]

	t2 = time.time()
	print(f"Experiment took {(t2 - t1)/60} minutes")



#configs = list(filter(lambda x: list(x) == sorted(x)[::-1], (list(tup) for tup in product(*(range(s, max_size+1) for s, i in zip(min_sizes, range(n_layers)))))))
