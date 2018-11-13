from scipy import ndimage
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import os
import numpy as np

def load_images(path, op, ed):
    files =  [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
    files = list(filter(lambda path: os.path.exists(path), files))
    L = len(files)
    frames = [ndimage.imread(file) for file in files]
    return frames
    
def bbc(vidframes, padding, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.zeros((3,padding,112,112))

    croptransform = transforms.CenterCrop((112, 112))

    if(augmentation):
        crop = StatefulRandomCrop((122, 122), (112, 112))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            crop,
            flip
        ])

    for i in range(len(vidframes)):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((122, 122)),
            transforms.CenterCrop((122, 122)),
            croptransform,
            transforms.ToTensor(),
            transforms.Normalize([0,0,0],[1,1,1]),
        ])(vidframes[i])

        temporalvolume[:,i] = result
    '''
    for i in range(len(vidframes), padding):
        temporalvolume[0][i] = temporalvolume[0][len(vidframes)-1]
    '''
    return temporalvolume
