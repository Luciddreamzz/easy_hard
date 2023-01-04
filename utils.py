import numpy as np
import gzip
import json
from matplotlib import pyplot as plt


def plot_decision_boundaries(X, y, model, steps=1000, cmap='Paired'):
	"""
	Function to plot the decision boundary and data points of a model.
	Data points are colored based on their actual label.
	"""
	X_test, y_test_coarse = X, y

	cmap = plt.get_cmap(cmap)

	# Define region of interest by data limits
	xmin, xmax = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
	ymin, ymax = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

	# Create grid points
	x_span = np.linspace(xmin, xmax, steps)
	y_span = np.linspace(ymin, ymax, steps)
	xx, yy = np.meshgrid(x_span, y_span)

	# Make predictions across region of interest
	labels = model.predict_label(np.c_[xx.ravel(), yy.ravel()])

	# Plot decision boundary in region of interest
	z = labels.reshape(xx.shape)

	fig, ax = plt.subplots(figsize=(5, 5))
	ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
	ax.set_title("Decision boundaries")

	# Get predicted labels on training data and plot
	train_labels = model.predict(X_test)
	ax.scatter(X_test[:, 0], X_test[:, 1], s=15, c=y_test_coarse, cmap=cmap, lw=0)
	# plt.setp(ax1.get_xticklabels(), visible=False)
	# plt.setp(ax2.get_xticklabels(), visible=False)

	return fig


def save_json_gz(obj, filepath):

	json_str = json.dumps(obj)
	json_bytes = json_str.encode()
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)
