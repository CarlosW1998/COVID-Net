#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import glob
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import numpy as np
import pandas as pd


# In[2]:


"""Helper functions"""
def _extract_data(df_row):
    return df_row['Anon MRN'], df_row['Anon TCIA Study Date'], df_row['Anon Exam Description'], df_row['Anon Study UID']


def load_ricord_metadata(ricord_meta_file):
    df = pd.read_excel(ricord_meta_file, sheet_name='CR Pos - TCIA Submission')
    ricord_metadata = []
    for index, row in df.iterrows():
        ricord_metadata.append(_extract_data(row))
    return ricord_metadata

def make_ricord_dict(ricord_data_set_file):
    """Loads bboxes from the given text file"""
    ricord_dict = {}
    with open(ricord_data_set_file, 'r') as f:
        for line in f.readlines():
            # Values after file name are crop dimensions
            if(len(line.split()) > 1):
                fname, xmin, ymin, xmax, ymax = line.rstrip('\n').split()
                bbox = tuple(int(c) for c in (xmin, ymin, xmax, ymax))
                ricord_dict[fname] = bbox
            else:
                fname = line.rstrip('\n')
                ricord_dict[fname] = None
                
    return ricord_dict


# In[ ]:


"""
RICORD data requires some preprocessing before splitting into test/train. 
Some images contain padding and some images are unusable. 
ricord_data_set.txt contains the name of usable images along with bounding box dimensions if needed. 

This cell crops the DICOM according to dimensions in ricord_data_set.txt and saves the image 
as png format in out_dir.

DICOM_images and "MIDRC-RICORD-1c Clinical Data Jan 13 2021 .xlsx" need to be downloaded from
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281 before running this cell
"""

ricord_dir = './manifest-1610656454899/MIDRC-RICORD-1C'
ricord_meta_file = './MIDRC-RICORD-1c Clinical Data Jan 13 2021 .xlsx'

out_dir = 'ricord_images'
ricord_set_file = 'ricord_data_set.txt'

os.makedirs(out_dir, exist_ok=True)

ricord_dict = make_ricord_dict(ricord_set_file)

metadata = load_ricord_metadata(ricord_meta_file)
file_count = 0
for mrn, date, desc, uid in metadata:
    date = date.strftime('%m-%d-%Y')
    uid = uid[-5:]
    study_dir = os.path.join(ricord_dir, 'MIDRC-RICORD-1C-{}'.format(mrn), '*-{}'.format(uid))
    dcm_files = sorted(glob.glob(os.path.join(study_dir, '*', '*.dcm')))
    for i, dcm_file in enumerate(dcm_files):
        # Create output path and check if image is to be included
        out_fname = 'MIDRC-RICORD-1C-{}-{}-{}.png'.format(mrn, uid, i)
        out_path = os.path.join(out_dir, out_fname)

        if out_fname not in ricord_dict:
            continue

        # Load DICOM image
        ds = pydicom.dcmread(dcm_file)

        # Verify orientation
        if ds.ViewPosition != 'AP' and ds.ViewPosition != 'PA':
            print('Image from MRN-{} Date-{} UID-{} in position {}'.format(mrn, date, uid, ds.ViewPosition))
            continue

        # Apply transformations if required
        if ds.pixel_array.dtype != np.uint8:
            # Apply LUT transforms
            arr = apply_modality_lut(ds.pixel_array, ds)
            if arr.dtype == np.float64 and ds.RescaleSlope == 1 and ds.RescaleIntercept == 0:
                arr = arr.astype(np.uint16)
            arr = apply_voi_lut(arr, ds)
            arr = arr.astype(np.float64)

            # Normalize to [0, 1]
            arr = (arr - arr.min())/arr.ptp()

            # Invert MONOCHROME1 images
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                arr = 1. - arr

            # Convert to uint8
            image = np.uint8(255.*arr)
        else:
            # Invert MONOCHROME1 images
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                image = 255 - ds.pixel_array
            else:
                image = ds.pixel_array

        # Crop if necessary
        bbox = ricord_dict[out_fname]
        if bbox is not None:
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Save image
        cv2.imwrite(out_path, image)
        file_count += 1
print('Created {} files'.format(file_count))

