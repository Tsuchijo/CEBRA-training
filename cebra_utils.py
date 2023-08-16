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


@cebra.models.register("Mesogan-Encoder")
class AttnLSTMEncoder(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True, ch=64, lstm_hidden=512, attn_kernel=3):
        super().__init__(
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

        self.downsample = nn.Sequential(
            # (1, 128, 128) -> (64, 128, 128)
            nn.Conv2d(1, ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            # (64, 128, 128) -> (128, 64, 64)
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            # (128, 64, 64) -> (256, 32, 32)
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
            # (256, 32, 32) -> (512, 16, 16)
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),
            # (512, 16, 16) -> (512, 8, 8)
            nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),
        )

        if attn_kernel == 1: padding = 0
        elif attn_kernel == 3: padding = 1

        self.attention = nn.Conv2d(ch * 8 + lstm_hidden, 1, kernel_size=attn_kernel, stride=1, padding=padding, bias=False)

        self.net_h0 = nn.Sequential(
            # (512, 8, 8) -> (lstm_hidden, 4, 4)
            nn.Conv2d(ch * 8, lstm_hidden, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(lstm_hidden),
            nn.ReLU(inplace=True),
            # (512, 4, 4) -> (512, 1, 1)
            nn.AvgPool2d(4)
        )

        self.net_c0 = nn.Sequential(
            # (512, 8, 8) -> (512, 4, 4)
            nn.Conv2d(ch * 8, lstm_hidden, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(lstm_hidden),
            nn.ReLU(inplace=True),
            # (512, 4, 4) -> (512, 1, 1)
            nn.AvgPool2d(4)
        )
        self.lstm = nn.LSTM(input_size=ch * 8, hidden_size=lstm_hidden, batch_first=True)

    def forward(self, x):
        # swap dimensions 1 and 2
        x = x.movedim(1, 2)
        x = x[:, :, None, : , :]
        n, t = x.size(0), x.size(1)

        # x: (n * t, 1, 128, 128) -> (n * t, 512, 8, 8)
        x = x.view(n * t, x.size(2), x.size(3), x.size(4))
        h = self.downsample(x)

        # (n * t, 512, 8, 8) -> (n, t, 512, 8, 8)
        h_expand = h.view(n, t, h.size(1), h.size(2), h.size(3))
        # (n, t, 512, 8, 8) -> (n, 512, 8, 8)
        h_mean = torch.mean(h_expand, dim=1)

        # (n, 512, 8, 8) -> (n, 512, 1, 1)
        h0 = self.net_h0(h_mean)
        c0 = self.net_c0(h_mean)

        # at each time-step, compute an attention map
        attn_maps = []
        outputs = []
        hidden = (h0.view(1, h0.size(0), h0.size(1)), c0.view(1, c0.size(0), c0.size(1)))

        # attention lstm
        for i in range(t):
            current_h = h_expand[:, i, :, :, :]

            # (1, n, lstm_hidden) -> (n, lstm_hidden, 1, 1) -> (n, lstm_hidden, 8, 8)
            previous_lstm_h = hidden[0].view(hidden[0].size(1), hidden[0].size(2), 1, 1).expand((-1, -1, current_h.size(2), current_h.size(3)))
            # (n, 512+lstm_hidden, 8, 8)
            attn_input = torch.cat((current_h, previous_lstm_h), dim=1)

            # (n, 512+lstm_hidden, 8, 8) -> (n, 1, 8, 8)
            attn_map = self.attention(attn_input)
            attn_map = attn_map.view(n, 1,  8 * 8)
            attn_map = torch.nn.functional.softmax(attn_map, dim=2)
            attn_map = attn_map.view(n, 1, 8, 8)

            # (n, 1, 8, 8) * (n, 512, 8, 8) -> (n, 512) -> (n, one time step, 512)
            attn_applied = torch.sum(attn_map * current_h, dim=(2, 3))
            attn_applied = attn_applied.view(attn_applied.size(0), 1, attn_applied.size(1))

            output, hidden = self.lstm(attn_applied, hidden)

            attn_maps.append(attn_map)
            outputs.append(output)

        # (n, t, 1, 8, 8)
        attn_maps = torch.stack(attn_maps, dim=1)
        # (n, t, 1, 512)
        outputs = torch.stack(outputs, dim=1)
        # take mean of outputs over time 
        # (n, t, 1, 512) -> (n, 512)
        outputs = torch.mean(torch.squeeze(outputs), dim=1)
        # normalize outputs
        outputs = F.normalize(outputs, p=2, dim=1)
        # hidden: ((1, n, 512), (1, n, 512))
        return outputs
    def get_offset(self):
            return cebra.data.Offset(5, 5)

def process_brain(brain_seq):
    brain_seq = np.array(brain_seq)
    # if brain seq > 128 x 128, downsample
    if brain_seq.shape[1] > 128:
        # downsample to 128 x 128
        brain_seq = np.array([cv2.resize(brain_frame, (128, 128)) for brain_frame in brain_seq])

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
    print (np.array(data).shape)
    data = np.array(data)
    t, x, y = data.shape
    padded = np.zeros((t + pre + post, x, y))
    padded[pre:pre + t, :, :] = data
    padded[:pre, :, :] = data[0:1, :, :]
    padded[pre + t:, :, :] = data[-1:, :, :]
    return padded

def generate_CEBRA_embeddings(model, data, session_id, offset = (2,3)):
    data_torch = torch.empty(0,offset[0] + offset[1],128,128).to('cuda')
    padded = pad_data(np.squeeze(data), offset[0], offset[1])
    print (padded.shape)
    for i, frame in enumerate(data):
        frame = torch.from_numpy(np.array(padded[i: i + offset[0] + offset[1]])).float().unsqueeze(0).to('cuda')
        data_torch = torch.cat((data_torch, frame), dim = 0)
    data_torch = data_torch.swapdims(-2, 1)
    # batch process data to save memory
    output = None
    model = model[session_id].eval().to('cuda')
    for i in range(0, len(data_torch), 100):
        embedding = model(data_torch[i:i+100]).detach().cpu().numpy().squeeze()
        if i == 0:
            output = embedding
        else:
            output = np.concatenate((output, embedding), axis = 0)
    return output

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

def choose_first_second( window_size, data):
    return data[0:0+window_size]

def normalize_array(in_array):
    return np.array([x / np.linalg.norm(x) for x in in_array])