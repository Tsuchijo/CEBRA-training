#!/home/murph_4090ws/miniconda3/envs/cebra-env/bin/python
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

key_behavior = {
    0: 'still',
    1: 'sniffing',
    2: 'walking',
    3: 'grooming',
}

for data_path in behavior_data_paths:
    # Load data
    behavior_data, behavior_names = import_data(data_path, lambda x : x, 0, 0.2)
    labels = dict()
    print('labelling: ' + str(len(behavior_data)) + ' videos')

    # create one hot encoding dictionary for each behavior
    one_hot = dict()
    for i, behavior in enumerate(key_behavior.values()):
        one_hot[behavior] = np.zeros(len(key_behavior))
        one_hot[behavior][i] = 1

    for video, name in zip(behavior_data, behavior_names):
        # Open the video file
        if name.split('_')[0] == 'nomove':
            labels[name] = one_hot['still']
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
                    labels[name] =  one_hot[key_behavior[key - ord('0')]]
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

