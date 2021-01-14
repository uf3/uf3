import numpy as np
from uf3.util import subsample


def test_fps():
    data = np.random.rand(50) * 100
    max_samples = 10
    indices = subsample.farthest_point_sampling(data,
                                                max_samples=max_samples)
    assert (len(indices) == max_samples) or (len(indices) == len(data))
    assert len(np.unique(indices)) == len(indices)
    # test min_diff
    data = np.arange(100)
    max_samples = 50
    min_diff = 10
    indices = subsample.farthest_point_sampling(data,
                                                max_samples=max_samples,
                                                min_diff=min_diff)
    assert len(indices) < max_samples
    # test max_samples
    max_samples = 200
    indices = subsample.farthest_point_sampling(data,
                                                max_samples=max_samples)
    assert len(indices) == len(data)
    indices = subsample.farthest_point_sampling(data)
    assert len(indices) == len(data)