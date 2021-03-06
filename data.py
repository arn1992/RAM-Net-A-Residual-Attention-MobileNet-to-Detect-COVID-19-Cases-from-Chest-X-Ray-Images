import numpy as np
import pandas as pd
import os
import random
from shutil import copyfile
import pydicom as dicom
import cv2

# set parameters here
savepath = 'D:/Covid19/data'
seed = 0
np.random.seed(seed) # Reset the seed so all runs are the same.
random.seed(seed)
MAXVAL = 255  # Range [0 255]

# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = 'D:/Covid19/covid-chest-xray/images'
cohen_csvpath = 'D:/Covid19/covid-chest-xray/metadata.csv'

# path to covid-14 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = 'D:/Covid19/covid-chest-xray/images'
fig1_csvpath = 'D:/Covid19/covid-chest-xray/metadata.csv'

# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
rsna_datapath = 'D:/Covid19/rsna-pneumonia-detection-challenge/rsna-pneumonia-detection-challenge'
# get all the normal from here
rsna_csvname = 'D:/Covid19/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
# get all the 1s from here since 1 indicate pneumonia
# found that images that aren't pneunmonia and also not normal are classified as 0s
rsna_csvname2 = 'D:/Covid19/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
rsna_imgpath = 'D:/Covid19/rsna-pneumonia-detection-challenge/stage_2_train_images'

# parameters for COVIDx dataset
train = []
test = []
test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}


mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Klebsiella'] = 'pneumonia'
mapping['Chlamydophila'] = 'pneumonia'
mapping['Legionella'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'

# train/test split
split = 0.1

# to avoid duplicates
patient_imgpath = {}


# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
#idx_pa = csv["view"] == "PA"  # Keep only the PA view
views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
cohen_idx_keep = cohen_csv.view.isin(views)
cohen_csv = cohen_csv[cohen_idx_keep]

fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
#fig1_idx_keep = fig1_csv.view.isin(views)
#fig1_csv = fig1_csv[fig1_idx_keep]

# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
for index, row in cohen_csv.iterrows():
    #
    f = row['finding'].split(',')[0]  # take the first finding, for the case of COVID-19, ARDS
    if f in mapping:  #
        count[mapping[f]] += 1
        entry = [str(row['patientid']), row['filename'], mapping[f], row['view']]
        filename_label[mapping[f]].append(entry)

for index, row in fig1_csv.iterrows():
    #print(row[index])
    if not str(row['finding']) == 'nan':
        f = row['finding'].split(',')[0]  # take the first finding
        if f in mapping:  #
            #print(row['patientid'])
            count[mapping[f]] += 1
            if os.path.exists(os.path.join(fig1_imgpath, str(row['patientid']) + '.jpg')):
                entry = [str(row['patientid']), str(row['patientid']) + '.jpg', mapping[f]]
            elif os.path.exists(os.path.join(fig1_imgpath, str(row['patientid']) + '.png')):
                entry = [str(row['patientid']), str(row['patientid']) + '.png', mapping[f]]
            filename_label[mapping[f]].append(entry)

print('Data distribution from covid-chestxray-dataset:')
print(count)

# add covid-chestxray-dataset into COVIDx dataset
# since covid-chestxray-dataset doesn't have test dataset
# split into train/test by patientid
# for COVIDx:
# patient 8 is used as non-COVID19 viral test
# patient 31 is used as bacterial test
# patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr[:,0]))
    # num_test = max(1, round(split*num_diff_patients))
    # select num_test number of random patients
    if key == 'pneumonia':
        test_patients = ['8', '31']
    elif key == 'COVID-19':
        test_patients = ['19', '20', '36', '42', '86',
                         '94', '97', '117', '132',
                         '138', '144', '150', '163', '169'] # random.sample(list(arr[:,0]), num_test)
    else:
        test_patients = []
    print('Key: ', key)
    print('Test patients: ', test_patients)
    # go through all the patients
    for patient in arr:
        if patient[0] not in patient_imgpath:
            patient_imgpath[patient[0]] = [patient[1]]
        else:
            if patient[1] not in patient_imgpath[patient[0]]:
                patient_imgpath[patient[0]].append(patient[1])
            else:
                continue  # skip since image has already been written
        if patient[0] in test_patients:
            copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
            test.append(patient)
            test_count[patient[2]] += 1
        else:
            if 'COVID' in patient[0]:
                copyfile(os.path.join(fig1_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
            else:
                copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
            train.append(patient)
            train_count[patient[2]] += 1

print('test count: ', test_count)
print('train count: ', train_count)

# add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
csv_normal = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname), nrows=None)
csv_pneu = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname2), nrows=None)
patients = {'normal': [], 'pneumonia': []}

for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])

for index, row in csv_pneu.iterrows():
    if int(row['Target']) == 1:
        patients['pneumonia'].append(row['patientId'])

for key in patients.keys():
    print(patients[key])
    arr = np.array(patients[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr))
    # num_test = max(1, round(split*num_diff_patients))
    test_patients = np.load('rsna_test_patients_{}.npy'.format(key))  # random.sample(list(arr), num_test), download the .npy files from the repo.
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
    for patient in arr:
        if patient not in patient_imgpath:
            patient_imgpath[patient] = [patient]
        else:
            continue  # skip since image has already been written

        ds = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
        pixel_array_numpy = ds.pixel_array
        imgname = patient + '.png'
        if patient in test_patients:
            cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
            test.append([patient, imgname, key])
            test_count[key] += 1
        else:
            cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
            train.append([patient, imgname, key])
            train_count[key] += 1

print('test count: ', test_count)
print('train count: ', train_count)

# final stats
print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))

# export to train and test csv
# format as patientid, filename, label, separated by a space
train_file = open("train_split_v3.txt","a")
for sample in train:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()

test_file = open("test_split_v3.txt", "a")
for sample in test:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)

test_file.close()

