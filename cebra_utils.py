import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from PIL import Image
import cv2
import os
import torch
import torch.nn.functional as F
import itertools
import random
from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin

class ChangeOrderLayer(nn.Module):
    def __init__(self, first_dim = -2, second_dim = 1):
        super().__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim
    def forward(self, x):
        return x.movedim(self.first_dim, self.second_dim).squeeze() # Permute dimensions 1 and 2

@cebra.models.register("convolutional-model-offset11")
class ConvulotionalModel1(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ChangeOrderLayer(),
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(1024, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(2, 3)
    

@cebra.models.register("convolutional-model-30frame")
class ConvulotionalModel30Frame(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ChangeOrderLayer(1,1),
            nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(1024, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
        def get_offset(self):
            return cebra.data.Offset(2, 3)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

def process_brain(brain_seq):
    brain_seq = np.array(brain_seq)
    flat_seq = np.array([(brain_frame.flatten()) for brain_frame in brain_seq])
    return flat_seq.astype(float)


## Takes a sliding window of data and then returns a list of windows
# data: list of data
# window_size: size of window
# returns: list of windows
def bin_data(data, window_size):
    output = []
    for i in range(len(data) - window_size + 1):
        output.append(data[i:i+window_size])
    return output


## Loads data from a folder of TIF files
# filepath: path to folder
# processor: function to process each image
# max: max images to load as a proportion of array size
# min: min images to load as a proportion of array size
# returns: list of processed images, list of filenames
def import_data(filepath, processor, min = 0, max = 1):
    output_data = []
    output_name = []
    path_list = os.listdir(filepath)
    path_list.sort()
    random.Random(4).shuffle(path_list)
    min_index = int(min * len(path_list))
    max_index = int(max * len(path_list))
    for file in itertools.islice(path_list, min_index, max_index):
     filename = os.fsdecode(file)
     if filename.endswith(".tif"):
         out = cv2.imreadmulti(filepath + '/' + filename)[1]
         output_data.append(processor(out))
         output_name.append(filename.split('.')[0])
     elif filename.endswith(".npy"):
         output_data.append(processor(np.load(filepath + '/' + filename)))
         output_name.append(filename.split('.')[0])
     else:
         continue
    return output_data, output_name

def flatten_data(data):
    return np.concatenate(data, axis=0)

def pad_data(data, pre, post):
    pre_padding = np.repeat(np.expand_dims(data[0], axis = 0), pre, axis=0)
    post_padding = np.repeat(np.expand_dims(data[-1], axis = 0), post, axis=0)
    return np.concatenate((pre_padding, data, post_padding), axis=0)

def generate_CEBRA_embeddings(model, data, session_id, offset = (2,3)):
    data_torch = torch.empty(0,5,128,128).to('cuda')
    padded = pad_data(data, offset[0], offset[1])
    for i, frame in enumerate(data):
        frame = torch.from_numpy(np.array(padded[i: i + offset[0] + offset[1]])).float().unsqueeze(0).to('cuda')
        data_torch = torch.cat((data_torch, frame), dim = 0)
    data_torch = data_torch.swapdims(-2, 1)
    embedding = model.to('cuda')[session_id](data_torch).detach().cpu().numpy().squeeze()
    return embedding

def load_model(model_path):
    #find available device
    saved_solver = torch.load(model_path)
    model = saved_solver.model
    return model

def reshape_frames(frames, shape_ref):
    shape_list = [np.shape(x)[0] for x in shape_ref]
    gen_video_list = []
    index = 0
    for shape in shape_list:
        gen_video_list.append((frames[index : index + shape]))
        index += shape
    return gen_video_list

#choose a random window of set size from the data deterministically based on seed
def choose_random_window( window_size, seed, data):
    random.seed(seed)
    start = random.randint(0, len(data) - window_size)
    return data[start:start+window_size]

def normalize_array(in_array):
    return np.array([x / np.linalg.norm(x) for x in in_array])