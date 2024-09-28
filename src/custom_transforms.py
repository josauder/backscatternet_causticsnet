from __future__ import division
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''
#torchvision.transforms.functional.rgb_to_grayscale

#        return fn_idx, b, c, s, h
#        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
#            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

#        return img

class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, images):
        fn_idx = torch.randperm(4)

        brightness_factor = float(torch.empty(1).uniform_(-self.brightness, self.brightness))
        contrast_factor = float(torch.empty(1).uniform_(-self.contrast, self.contrast))
        saturation_factor = float(torch.empty(1).uniform_(-self.saturation, self.saturation))
        hue_factor = float(torch.empty(1).uniform_(-self.hue, self.hue))
        
        output_images = []
        for img in images:

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, 1+brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, 1+contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, 1+saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
            output_images.append(img)
        return output_images

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images

    def invert(self, img):
        img = torch.clone(img.detach())
        for t, m, s in zip(img, self.mean, self.std):
             t.mul_(s).add_(m)
        return img

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors

    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5. Assumes the intrinsics are not modified (that the focal point is in the center!)"""
    def __call__(self, images):
        if random.random() < 0.5:
            return [F.hflip(im) for im in images]
        return images

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics