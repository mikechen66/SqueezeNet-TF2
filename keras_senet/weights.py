#!/usr/bin/env python
# coding: utf-8

# weights.py

"""
There are the weights used im the SqueezeNet model variants including the scenatios of "inlcude_top=True"
and "inlcude_top=Flase". I would like to thank qubvel for his generous contribution.

"""

from keras.utils.data_utils import get_file


def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


WEIGHTS_COLLECTION = [

    # SE models
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000.h5',
        'name': 'seresnet50_imagenet_1000.h5',
        'md5': 'ff0ce1ed5accaad05d113ecef2d29149',
    },

    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000_no_top.h5',
        'name': 'seresnet50_imagenet_1000_no_top.h5',
        'md5': '043777781b0d5ca756474d60bf115ef1',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000.h5',
        'name': 'seresnet101_imagenet_1000.h5',
        'md5': '5c31adee48c82a66a32dee3d442f5be8',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000_no_top.h5',
        'name': 'seresnet101_imagenet_1000_no_top.h5',
        'md5': '1c373b0c196918713da86951d1239007',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000.h5',
        'name': 'seresnet152_imagenet_1000.h5',
        'md5': '96fc14e3a939d4627b0174a0e80c7371',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000_no_top.h5',
        'name': 'seresnet152_imagenet_1000_no_top.h5',
        'md5': 'f58d4c1a511c7445ab9a2c2b83ee4e7b',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000.h5',
        'name': 'seresnext50_imagenet_1000.h5',
        'md5': '5310dcd58ed573aecdab99f8df1121d5',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000_no_top.h5',
        'name': 'seresnext50_imagenet_1000_no_top.h5',
        'md5': 'b0f23d2e1cd406d67335fb92d85cc279',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000.h5',
        'name': 'seresnext101_imagenet_1000.h5',
        'md5': 'be5b26b697a0f7f11efaa1bb6272fc84',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000_no_top.h5',
        'name': 'seresnext101_imagenet_1000_no_top.h5',
        'md5': 'e48708cbe40071cc3356016c37f6c9c7',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000.h5',
        'name': 'senet154_imagenet_1000.h5',
        'md5': 'c8eac0e1940ea4d8a2e0b2eb0cdf4e75',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000_no_top.h5',
        'name': 'senet154_imagenet_1000_no_top.h5',
        'md5': 'd854ff2cd7e6a87b05a8124cd283e0f2',
    },
]