#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:13:48 2019

@author: craigposkanzer
"""

import numpy as np
import nibabel as nib 

categories= (['body', 'face', 'scene', 'object'])
sublist = (['sub-01','sub-02','sub-03','sub-04','sub-05','sub-09','sub-10',
            'sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'])
runs = (['localizer_run_2','localizer_run_3','localizer_run_4'])
data_dir = '/gsfs0/data/poskanzc/MVPN/data/'

for sub in sublist:
    for cat in categories:
        if cat == 'face':
            rois = (['FFA', 'OFA' , 'STS', 'GM'])
        if cat == 'body':
            rois = (['EBA', 'FBA' , 'STS', 'GM']) 
        if cat == 'object':
            rois = (['MFA', 'MTG', 'GM'])
        if cat == 'scene':
            rois = (['PPA', 'RSP' , 'TOS', 'GM'])
        
        for roi in rois:
            if roi == 'GM':
                num_vox = 53539
            else:
                num_vox = 80
            for run in runs:
                print(sub)
                print(roi)
                print('vox:', num_vox)
                mask_data = nib.load(data_dir + sub + '/' + cat + '_ROIs' + '/' + run + '/' + roi + '_functional.nii.gz' )
                mask_data_affine = mask_data.affine
                mask_data = mask_data.get_data()
                mask_shape = np.shape(mask_data)
                time_length = mask_shape[3]
                
                ROI_idx = np.nonzero(mask_data)
                ROI_array = np.zeros([time_length, num_vox])
                for i in range(num_vox*time_length):
                    x_ROI = ROI_idx[0][i]
                    y_ROI = ROI_idx[1][i]
                    z_ROI = ROI_idx[2][i]
                
                    t = ROI_idx[3][i]
                    v = i // time_length
                    
                    ROI_array[t,v] = mask_data[x_ROI,y_ROI,z_ROI,t]
                if np.shape(ROI_array)[1]>80:
                    np.save(data_dir + sub + '/' + cat + '_ROIs' + '/' + run + '/' + roi + '_vox.npy', ROI_array)
                else:
                    np.save(data_dir + sub + '/' + cat + '_ROIs' + '/' + run + '/' + roi + '_80vox.npy', ROI_array)    
                            
          