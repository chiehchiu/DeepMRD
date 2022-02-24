import inspect
import math
import random

import mmcv
import numpy as np

from ..builder import PIPELINES
import torchvision.transforms as transforms
from PIL import Image

@PIPELINES.register_module()
class XrayTrain(object):

    def __init__(self, transCrop, scale=(0.09, 1)):
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transformSequence = transforms.Compose([
            transforms.RandomResizedCrop(transCrop, scale=scale),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.6, 1.4, 0, 0),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([0.1, 3.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, results):

        img = results['img']
        filename = results['filename']
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        try:
            img = self.transformSequence(img)
        except:
            print("Cannot transform images: {}".format(filename))

        results['img'] = img

        return results


@PIPELINES.register_module()
class XrayTest(object):

    def __init__(self, transResize, transCrop):
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transformSequence = transforms.Compose([
            transforms.Resize((transResize, transResize)),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, results):

        img = results['img']
        filename = results['filename']
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        try:
            img = self.transformSequence(img)
        except:
            print("Cannot transform images: {}".format(filename))

        results['img'] = img

        return results
