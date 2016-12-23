from boltons.iterutils import chunked_iter
from os.path import join as ojoin
import h5py
import json


class Iterator(object):

    def __init__(self, N, batch_size, seed):
        self.N = N
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_generator = self._flow_index(N, batch_size, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        idxs = range(N)
        for batch_index, chunk in enumerate(chunked_iter(idxs, size=self.batch_size)):
            self.batch_index = batch_index
            yield batch_index, chunk

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class H5FeatureBatchGenerator(Iterator):
    """
    mini-batch generator from a already shuffle h5 file
    """

    def __init__(self, feature_path, split, batch_size=5, seed=None):
        self.feature_path = feature_path
        self.split = split
        self.batch_size = batch_size
        self.hf = ojoin(feature_path, 'feature.h5')
        config = json.load(
            open(ojoin(feature_path, 'feature_config.json'), 'r'))
        self.__dict__.update(config)
        self.N = self.get_nsample(split)
        super(H5FeatureBatchGenerator, self).__init__(self.N, batch_size, seed)

    def get_nsample(self, split):
        if getattr(self, 'nsample_{}'.format(split)) is None:
            nsample = getattr(self, '{}_nb_sample'.format(split))
        else:
            nsample = getattr(self, 'nsample_{}'.format(split))
        return nsample

    def get(self, key, chunk):
        with h5py.File(self.hf) as hf:
            arr = hf.get(key)[chunk]
        return arr

    def next(self):
        batch_idx, chunk = next(self.index_generator)
        batch_x = self.get('X_{}'.format(self.split), chunk)
        batch_y = self.get('y_{}'.format(self.split), chunk)
        return batch_x, batch_y
