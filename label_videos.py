#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import cv2
import os
import random
import itertools

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
         print(filename)
         out = cv2.imreadmulti(filepath + '/' + filename)[1]
         output_data.append(processor(out))
         output_name.append(filename.split('.')[0])
     elif filename.endswith(".npy"):
         output_data.append(processor(np.load(filepath + '/' + filename)))
         output_name.append(filename.split('.')[0])
     else:
         continue
    return output_data, output_name

data_directory = '/mnt/teams/Tsuchitori/MV1_run_30hz_30frame_brain2behav_DFF_new/'
behavior_data_paths = [  data_directory + 'camera1/' + \
                     file for file in os.listdir(data_directory + 'brain/')]

data_path = behavior_data_paths[0]

# Load data
behavior_data, behavior_names = import_data(data_path, lambda x : x, 0, 1)
labels = dict()
print('running')

key_behavior = {
    0: 'still',
    1: 'sniffing',
    2: 'walking',
    3: 'chewing',
}

for video, name in zip(behavior_data, behavior_names):
    # Open the video file
    if name.split('_')[0] == 'nomove':
        labels[name] = 'still'
        continue
    while True:
        exit = False
        for frame in video:
            # Display the frame to the user
            cv2.imshow('Video', frame)
            key = cv2.waitKey(12) & 0xFF # Wait 12ms for user input
            # Exit the loop and close the window when 'q' is pressed
            if key == ord('q'):
                exit = True
                break
            # if a number is pressed save that number to the labels
            elif key >= ord('0') and key <= ord('9'):
                labels[name] =  key_behavior[key - ord('0')]
                exit = True
                break

        # Restart the video when the loop ends
        if exit:
            break


    cv2.destroyAllWindows()
# Save the labels to a pickle file
import pickle
file = data_path.split('/')[-1]
with open(file + 'labels.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

