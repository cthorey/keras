'''

This  module   contains  code  that  extent   the  keras.preprocessing
module. In particular, it proposes two classes that allow to generate
min-batch of image + bboxes. 

```
data_gen=ImageBBoxDataGenerator(nbbox=5,normalize_bbox=True,...)
generator=data_gen.flow_from_directory('data/train',target_size=(224,244))

for X_batch,y_batch in generator:
    do stuff
```

I  just   implemented  the   flow  from   directory  method   for  the
moment. X_batch has the shape  (nbatch,targetsize) and y_batch has the
shape  (nbatch,nbbox*5).   The  5  corresponds  to   x0,y0,x1,y1  +  a
confidence score. 
'''


from keras.preprocessing.image import *
from keras.models import Model as KerasModel
from tqdm import *
import numpy as np
import os
from keras.preprocessing.image import *
from PIL import Image, ImageDraw
import itertools
import random
import pandas as pd


def bbox_to_array(bbox, x, dim_ordering='tf'):
    '''
    Create an array representation for the bbox so that we can
    apply the the image transformation to the box as well. 
    '''
    # create a zeros matrix same size of the image
    if dim_ordering == 'tf':
        shape = x.shape[:-1] + (1,)
    elif dim_ordering == 'th':
        shape = (1,) + x.shape[:-1]
    else:
        raise ValueError('dim ordering')
    z = np.zeros(shape)
    # identify the bbox points coord in that image (!x,y inverted here: matrix
    # repres)
    if dim_ordering == 'tf':
        points = [[bbox[f + 1], bbox[f], 0]
                  for f in range(len(bbox)) if f % 2 == 0]
    elif dim_ordering == 'th':
        points = [[0, bbox[f + 1], bbox[f]]
                  for f in range(len(bbox)) if f % 2 == 0]
    else:
        raise ValueError('dim ordering')

    if len(bbox) > 0:
        # transform them into indices
        idx_points = reduce(lambda x, y: [[int(x[i])] + [int(y[i])]
                                          for i in range(3)], points)
        # Set this point equal to something
        z[idx_points] = 1
    return z


def array_to_bbox(x, dim_ordering='tf'):
    '''
    Inverse transformation than bbox_to_array. 
    '''
    bbox = np.argwhere(x != 0)
    if dim_ordering == 'tf':
        bbox = [list(bb)[:-1][::-1] for bb in list(bbox)]
    elif dim_ordering == 'th':
        bbox = [list(bb)[1:][::-1] for bb in list(bbox)]
    else:
        raise ValueError('dim_ordering')
    if len(bbox) > 0:
        bbox = reduce(lambda x, y: x + y, bbox)
    return bbox


def check_format_bbox(func):
    '''
    Check that the return bboxes by the random transform
    are composed of 4 coordinates. To remove this funciton,
    we need first to find a more clever approach to taking
    only the values != zero in the transformed matrix.
    '''
    def func_wrapper(*args, **kwargs):
        res = 0
        i = 0
        while res == 0:
            x_img, x_bbox = func(*args, **kwargs)
            if len(x_bbox) == 0:
                res = 1
            else:
                check = [len(array_to_bbox(bbox)) for bbox in x_bbox]
                res = int(np.all(np.array(check) == 4))
            i += 1
            if i > 50:
                raise Exception('Sorry, we try hard but did not get anything')

        return x_img, x_bbox
    return func_wrapper


class ImageBBoxDataGenerator(ImageDataGenerator):

    '''
    Generate minibatches with real-time data augmentation. Apply the same transformation
    to the image and the bunding box.


    # Arguments

        nbbox: Number of bboxes to output.
        normalize_bbox: Normalizing the bbox so that the coordiantes are between 0,1
        sort_by_size: Sort the bboxes by size, larger to smaller. 
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
            '''

    def __init__(self, nbbox=5, normalize_bbox=True, sort_by_size=True, *args, **kwargs):
        super(ImageBBoxDataGenerator, self).__init__(*args, **kwargs)
        self.normalize_bbox = normalize_bbox
        self.nbbox = nbbox
        self.sort_by_size = sort_by_size

    def arr2nlist(self, arr):
        return [list(l) for l in list(arr)]

    def xy2wh(self, coord):
        x0, y0, x1, y1 = coord
        return [x0, y0, x1 - x0, y1 - y0]

    def wh2xy(self, coord):
        x0, y0, w, h = coord
        return [x0, y0, x0 + w, y0 + h]

    def norm_bbox(self, bbox, imgw, imgh):
        '''
        (xo,yo,x1,y1) - (xo,yo,w,h) with all
        normalize
        '''
        x0, y0, w, h = bbox
        w = w / float(imgw)
        h = h / float(imgh)
        x0 = x0 / float(imgw)
        y0 = y0 / float(imgh)
        return [x0, y0, w, h]

    def standardize_bbox(self, bboxes, imgsize):
        '''
        input is a pandas dataframe
        '''
        imgw, imgh = imgsize
        # Put it in the format x0,y0,w,h
        bboxes = [self.xy2wh(coord) for coord in bboxes]
        if self.sort_by_size:
            bboxes = sorted(bboxes, key=lambda bbox: bbox[
                            2] * bbox[3], reverse=True)
        if self.normalize_bbox:
            bboxes = [self.norm_bbox(bbox, imgw, imgh)
                      for bbox in bboxes]
        # add 1 for the confidence
        bboxes = np.array([np.array(f + [1]) for f in bboxes])
        # fill in the array if size<nboxes
        nbox = bboxes.shape[0]
        if nbox == 0:
            bboxes = np.zeros((self.nbbox, 5))
        elif nbox < self.nbbox:
            zeros = np.zeros((self.nbbox - nbox, 5))
            bboxes = np.vstack((bboxes, zeros))
        else:
            bboxes = bboxes[:self.nbbox, :]

        # flatten the array to have a nbox*5
        bboxes = bboxes.flatten()
        return bboxes

    def standardize_img(self, x):
        return self.standardize(x)

    @check_format_bbox
    def random_transform(self, x_img, x_bbox):
        '''
        Apply random transform to both the image and the bbox
        x_img : array for one image dim (W,H,C)
        x_bbox : List of array [(W,H,1)]
        '''
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        if self.rotation_range > 10:
            raise ValueError(
                'Rotation larger than 10 is not supported at the moment')

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * x_img.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * x_img.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x_img.shape[img_row_index], x_img.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x_img = apply_transform(x_img, transform_matrix, img_channel_index,
                                fill_mode=self.fill_mode, cval=self.cval)
        xbbox = []
        for bbox in x_bbox:
            xbbox.append(apply_transform(bbox, transform_matrix, img_channel_index,
                                         fill_mode=self.fill_mode, cval=self.cval))
        x_bbox = xbbox

        if self.channel_shift_range != 0:
            x_img = random_channel_shift(
                x_img, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x_img = flip_axis(x_img, img_col_index)
                xbbox = []
                for bbox in x_bbox:
                    xbbox.append(flip_axis(bbox, img_col_index))
                x_bbox = xbbox

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x_img = flip_axis(x_img, img_row_index)
                xbbox = []
                for bbox in x_bbox:
                    xbbox.append(flip_axis(bbox, img_row_index))
                x_bbox = xbbox

        return x_img, x_bbox

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_every=None,
                            save_prefix='',
                            save_format='jpeg'):
        return ImageBBoxDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            dim_ordering=self.dim_ordering,
            nbbox=self.nbbox,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            save_every=save_every)


class ImageBBoxDirectoryIterator(Iterator):
    '''
    DirectoryIterator in the same idea than the one of the keras.
    However, the y is a set of bounding boxes

    Arguments:
        directory: path to the target directory. It should the images and a csv file 
        where each row is set of bbox coordiante (x0,y0,x1,y1)
        nbbox : Number of bbox to output for one image. Set to 5 by default.
        target_size: tuple of integers, default: (256, 256). The dimensions to which 
        all images found will be resized.
        color_mode: one of "grayscale", "rbg". Default: "rgb". Whether the images will
        be converted to have 1 or 3 color channels.
        batch_size: size of the batches of data (default: 32).
        shuffle: whether to shuffle the data (default: True)
        seed: optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None). This allows you to optimally specify a 
        directory to which to save the augmented pictures being generated (useful 
        for visualizing what you are doing).
        save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).
        save_format: one of "png", "jpeg" (only relevant if save_to_dir is set). Default: "jpeg".
        save_every is a probability to not save every images.
    '''

    def __init__(self, directory, data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 nbbox=5,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg', save_every=None):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.save_every = save_every
        self.directory = directory
        self.nbbox = nbbox
        self.data_generator = data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = ['png', 'jpg', 'jpeg']

        # Ensure that each picture comes with bbox
        self.filenames = list(set([os.path.splitext(f)[0]
                                   for f in os.listdir(self.directory)]))
        assert len(self.filenames) == len(os.listdir(self.directory)) / 2
        self.nb_sample = len(self.filenames)
        print('Found {} images'.format(self.nb_sample))

        super(ImageBBoxDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)

    def read_csv(self, filename):
        data = pd.read_csv(filename, index_col=0).values
        data = self.data_generator.arr2nlist(data)
        return data

    def resize_bb(self, coord, a, b):
        '''
        resize the bbox with the images
        a = w_newimage/w_original_image
        b = h_newimage/h_original_image
        '''
        x0, y0, x1, y1 = coord
        return x0 * a, y0 * b, x1 * a, y1 * b

    def rescale_bbox(self, bbox, imgsize):
        w, h = imgsize
        x0, y0, x1, y1 = bbox
        return x0 * w, y0 * h, x1 * w, y1 * h

    def arr_to_bbox(self, arr, imgsize=None, scale=False):
        s = arr.shape[0]
        arr = arr.reshape((s / 5, 5))[:, :4]
        bboxes = self.data_generator.arr2nlist(arr)
        bboxes = [self.data_generator.wh2xy(bbox) for bbox in bboxes]
        if scale:
            bboxes = [self.rescale_bbox(bbox, imgsize) for bbox in bboxes]
        return bboxes

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = np.zeros((current_batch_size, self.nbbox * 5))
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img_name = '{}.jpg'.format(fname)
            img = load_img(os.path.join(self.directory, img_name),
                           grayscale=grayscale)
            imgw, imgh = img.size

            # process img
            img = img.resize(self.target_size)
            x_img = img_to_array(img, dim_ordering=self.dim_ordering)

            # process bbox
            csv_name = '{}.csv'.format(fname)
            bboxes = self.read_csv(os.path.join(
                self.directory, csv_name))
            # resize them
            a = self.target_size[0] / float(imgw)
            b = self.target_size[1] / float(imgh)
            bboxes = [self.resize_bb(bbox, a, b) for bbox in bboxes]
            x_bbox = [bbox_to_array(
                bbox, x_img, dim_ordering=self.dim_ordering) for bbox in bboxes]

            # remove random transform for the moment
            try:
                x_img, x_bbox = self.data_generator.random_transform(
                    x_img, x_bbox)
            except:
                pass
            x_img = self.data_generator.standardize_img(x_img)

            x_bbox = [array_to_bbox(bbox) for bbox in x_bbox]
            x_bbox = self.data_generator.standardize_bbox(
                x_bbox, self.target_size)
            batch_y[i] = x_bbox
            batch_x[i] = x_img

            # optionally save augmented images to disk for debugging
            # purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                if self.save_every is not None:
                    p = np.random.rand()
                    if p > self.save_every:
                        break
                try:
                    arr_bx = np.copy(batch_x[i])
                    arr_by = np.copy(batch_y[i])

                    img = array_to_img(arr_bx, self.dim_ordering, scale=True)
                    draw = ImageDraw.Draw(img)
                    bboxes = self.arr_to_bbox(arr_by,
                                              imgsize=self.target_size,
                                              scale=self.data_generator.normalize_bbox)
                    for bbox in bboxes:
                        draw.rectangle(bbox)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(
                                                                          1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
                except:
                    pass

        return batch_x, batch_y
