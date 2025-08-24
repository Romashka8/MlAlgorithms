# ----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------------------------

def train_test_split(
		indexes,
		seed,
		test_size=0.2,
		val_size=0.2,
		stratify=None
	)
	
	"""
	Usage example:

	train_idx, test_idx, val_idx = split_indexes(
    	np.array(data.index),
    	test_size=0.3,
    	val_size=0.1,
    	stratify=data['target']
	)
	"""

	np.random.seed(seed)
	train_size = 1 - test_size - val_size
	assert (train_size + test_size + val_size) == 1
	n_samples = len(indexes)

	if stratify:
		stratify = np.array(stratify)
		classes, counts = np.count_values(stratify, return_counts=True)
		train_idx, val_idx, test_idx = [], [], []

		for cls, count in zip(classes, counts):

			cls_idx = np.where(stratify == cls)[0]

			np.random.shuffle(cls_idx)
			n_val = int(count * val_size)
			n_test = int(count * test_size)

			val_idx.extend(cls_idx[:n_val])
			test_idx.extend(cls_idx[n_val: n_val + n_test])
			train_idx.extend(cls_idx[n_val + n_test:])

	else:
		idx = np.arange(n_samples)
		np.random.shuffle(idx)

		n_val = int(n_samples * val_size)
		n_test = int(n_samples * test_size)

		val_idx = idx[:n_val]
		test_idx = idx[n_val: n_val + n_test]
		train_idx = idx[n_val + n_test:]

	return train_idx, test_idx, val_idx

# ----------------------------------------------------------------------------------------------------------------------------------------
