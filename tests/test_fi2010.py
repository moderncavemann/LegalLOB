import numpy as np

from genlob.data.fi2010 import FI2010WindowDataset
from scripts.train_fi2010_deeplob import make_train_val_subsets


def test_fi2010_window_dataset_aligns_window_end_label():
    features = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    labels = np.ones((20, 5), dtype=np.int64)
    labels[:, 2] = np.arange(20) % 3 + 1

    ds = FI2010WindowDataset(features, labels, window_len=5, label_index=2, stride=2)

    assert len(ds) == 8
    x0, y0 = ds[0]
    x1, y1 = ds[1]

    assert tuple(x0.shape) == (5, 4)
    assert np.allclose(x0.numpy(), features[:5])
    assert int(y0) == int(labels[4, 2] - 1)
    assert np.allclose(x1.numpy(), features[2:7])
    assert int(y1) == int(labels[6, 2] - 1)


def test_fi2010_train_validation_split_purges_window_and_label_overlap():
    features = np.zeros((1000, 40), dtype=np.float32)
    labels = np.ones((1000, 5), dtype=np.int64)
    ds = FI2010WindowDataset(features, labels, window_len=100, label_index=4)
    train, val = make_train_val_subsets(ds, val_fraction=0.1, label_horizon=100)
    last_train_start = int(ds.starts[max(train.indices)])
    first_val_start = int(ds.starts[min(val.indices)])
    assert last_train_start + ds.window_len - 1 + 100 < first_val_start
