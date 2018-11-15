import dicom
import os
import pandas as pd


data_dir = './data/kaggle/train'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('./data/kaggle/train/stage1_train_labels.csv' , index_col = 0)

print labels_df.head()

for patient in patients [:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir+patient
    




