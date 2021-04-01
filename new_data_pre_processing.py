from PIL import Image
import glob
import cv2
import numpy as np
import os
import pydicom
# import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.misc import imsave
import torch.nn.functional as F
import torch
import multiprocessing
from PIL import Image
def normal(patient_folder, result_path):
    img_sum = 0
    i=0
    j=1

    for filename in glob.glob(patient_folder+'/*.png'):
        print(filename)

        img=Image.open(filename)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img.save(filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype('float64')
        img_sum += img
        i += 1
        j =j+1

    img_sum = img_sum / i
    cv2.imwrite(result_path, img_sum)

if __name__ == '__main__':
    # Load the dicom folder
    save_axis = 'y' # orientation of the slices
    need_resample = True # Need to do resample or not
    INPUT_FOLDER_1 = './CT/Covid'
    INPUT_FOLDER_2 = './CT/Healthy'
    INPUT_FOLDER_3 = './CT/Others'


    RESULT_FOLDER_1 = './RESULT_1'
    RESULT_FOLDER_2 = './RESULT_2'
    RESULT_FOLDER_3 = './RESULT_3'

    patients1 = os.listdir(INPUT_FOLDER_1)
    patients2 = os.listdir(INPUT_FOLDER_2)
    patients3 = os.listdir(INPUT_FOLDER_3)
    patients1.sort()
    patients2.sort()
    patients3.sort()
    '''
    for patient in patients1:
        print(patient)




        # Save X
        result_path = os.path.join(RESULT_FOLDER_1, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)

        normal(os.path.join(INPUT_FOLDER_1, patient), os.path.join(result_path, 'normal.png'))'''
    for patient in patients2:
        print(patient)
        # Save y
        result_path = os.path.join(RESULT_FOLDER_2, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)

        normal(os.path.join(INPUT_FOLDER_2, patient), os.path.join(result_path, 'normal.png'))
    for patient in patients3:
        print(patient)
        # Save z
        result_path = os.path.join(RESULT_FOLDER_3, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)

        normal(os.path.join(INPUT_FOLDER_3, patient), os.path.join(result_path, 'normal.png'))
