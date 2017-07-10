# -*- coding: utf-8 -*-
"""
Reuben Farrugia

This script will be used to restore old passport images provided by the 
National Archive. This script will load the image to be restored and restore
it using our dictionary based face inpainting algorithm
"""
from os import listdir
import scipy.misc


print('---------------------------------------------------------------')

# Define the face id to be considered
face_id = 1;

img_folder = 'imgs/malta_archive/'
lms_folder = 'lms/malta_archive/'

# Get a list of images to be restored
from os.path import isfile, join
img_list = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]

# Define the fule to be processed
img_filename = img_folder + img_list[face_id]

# Print a message to show which image is being loaded
print('The image %s is being loaded...' %(img_filename))

# Load the face to be inpainted
Iface = scipy.misc.imread(img_filename)

# Deruve the reference model
ref_model_filename = 'model/refModel21.csv'

# Derive the filename of the corresponding landmark file
lm_filename = lms_folder + img_list[face_id][1:-4] + '.mat'

print(lm_filename)
