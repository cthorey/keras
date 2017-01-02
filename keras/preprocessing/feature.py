from boltons.iterutils import chunked_iter
from os.path import join as ojoin
import h5py
import json

from .image import Iterator


class H5FeatureIterator(Iterator):
    """
    mini-batch generator from a already shuffle h5 file.

    Note if you want shuffle, you have to do it when 
    you dump the feature. h5 format does not allow to slice 
    not contigus slice.
    """

    def __init__(self, feature_path,
                 split,
                 batch_size=5,
                 seed=None):
        self.feature_path = feature_path
        self.split = split
        self.batch_size = batch_size
        self.hf = ojoin(feature_path, 'feature.h5')
        config = json.load(
            open(ojoin(feature_path, 'feature_config.json'), 'r'))
        self.__dict__.update(config)
        self.nb_sample = self.get_nsample(split)
        # DONT CHANGE shuffle - reason in the docstring
        super(H5FeatureIterator, self).__init__(
            self.nb_sample, batch_size=batch_size, shuffle=False, seed=seed)

    def get_nsample(self, split):
        if getattr(self, 'nsample_{}'.format(split)) is None:
            nsample = getattr(self, '{}_nb_sample'.format(split))
        else:
            nsample = getattr(self, 'nsample_{}'.format(split))
        return nsample

    def get(self, key, chunk):
        chunk = list(chunk)  # make sure this is a list
        with h5py.File(self.hf) as hf:
            arr = hf.get(key)[chunk]
        return arr

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = self.get('X_{}'.format(self.split), index_array)
        batch_y = self.get('y_{}'.format(self.split), index_array)
        return batch_x, batch_y
