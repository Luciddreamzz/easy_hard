import json
import os
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from main_old import ex
from sklearn.model_selection import ParameterGrid
from pymongo import MongoClient
from concurrent import futures
from itertools import chain, islice, repeat


file_system_path = os.path.abspath('../output')
onedrive_path = os.path.join(os.path.expanduser('~'), 'OneDrive - Università degli Studi di Catania/easy_hard/output')
mongodb_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'easy_hard'

client = MongoClient('localhost', 27017)
db = client[curr_db_name]


def exp(config_updates):
	if not ex.observers:
		ex.observers.append(MongoObserver(url=mongodb_url, db_name=curr_db_name))
		ex.observers.append(FileStorageObserver(onedrive_path))

	model_update = dict(
		layers_sizes=config_updates
	)
	config = dict(learning_model=model_update)
	run = ex.run(config_updates=config)
	return run.result


n_circumferences = 10
max_repeats = 4
n_layers = 2
if n_layers > 1:
	data_type = "object"
else:
	data_type = "number"


cursor = db.runs.aggregate([
	{"$match": {"status": "COMPLETED", "config.dataset.classes": {"$size": n_circumferences},
	"config.learning_model.layers_sizes": {"$type": data_type}}, },
	{"$group": {
		"_id": "$config.learning_model.layers_sizes", "count": {"$sum": 1},

		# "exp": {"$push": "$$ROOT"}
	}},
	{"$match": {"count": {"$lt": max_repeats}}}
])


def chunks(iterable, chunk_size=4):
	iterator = chain.from_iterable(repeat(n, max_repeats-n['count']) for n in iterable)
	for first in iterator:
		yield list(chain([first], islice(iterator, chunk_size - 1)))


chunk_gen = chunks(cursor)

if __name__ == "__main__":

	for chunk in chunk_gen:

		configs = [tuple(el['_id']['py/tuple']) for el in chunk]
		with futures.ProcessPoolExecutor(max_workers=4) as executor:
			tasks = [executor.submit(exp, config) for config in configs]