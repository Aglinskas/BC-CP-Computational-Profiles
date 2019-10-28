#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:47:48 2019

@author: craigposkanzer
"""

#from mvpa2.tutorial_suite import *
from nilearn.input_data import NiftiMasker
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from sklearn import mixture
import datetime
import os 

root= '/Users/craigposkanzer/Documents/MVPN-Data'
sub='sub-01'
mask_filename = '/Users/craigposkanzer/Documents/MVPN-Data/brain_mask_bool.nii.gz'
fmri_filename = '/Users/craigposkanzer/Documents/MVPN-Data/5layer/5layer_images/lin_sub-01.nii'
f_temp = ['Data/1layer/{}_lin_img.nii','Data/1layer/{}_nl_img.nii','Data/5dense/lin_{}.nii','Data/5dense/{}_5dense_nl_img.nii','Data/5layer/lin_{}.nii','Data/5layer/{}_5layer_nl_img.nii']
mfn = root+'Data/brain_mask_bool.nii'

#turn the image into an array
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
#ds = masker.fit_transform(fmri_filename)
#ds = fmri_dataset(fn,mask=mfn)

# Grab the first scan
fn = root+f_temp[0].format(sub) # Filename

#ds = fmri_dataset(fn,mask=mfn) # Dataset

ds = masker.fit_transform(fn)
oDS = copy.deepcopy(ds)

# Append the other scanss
for i in np.arange(1,len(f_temp)):
    fn = root+f_temp[i].format(sub)
    #ads = fmri_dataset(fn,mask=mfn)
    ads = masker.fit_transform(fn)
    ds = np.vstack((ds,ads))
    
print('Datasets stacked')

data = np.array(copy.deepcopy(ds))
data=data.transpose()
#X : array-like, shape (n_samples, n_features)
model = mixture.BayesianGaussianMixture(max_iter=100000,
                                      n_components=100,covariance_type='diag',
                                      init_params='kmeans',
                                      weight_concentration_prior_type='dirichlet_process')
print('model specified')
#%% Run the model
t_start = datetime.datetime.now()
print('Running model')
BNP = model.fit(data)
print('time elapsed')
print(datetime.datetime.now()-t_start)


print(BNP.converged_)
C = BNP.predict(data) # 

#%%
print('num clusters: {}'.format(len(set(C))))#
#%% Arrange Clusters to make a bit more sense
Co=copy.deepcopy(C)
cmap0=np.unique(C)
linear=np.arange(len(np.unique(C)))
sz = np.array([sum(C==cmap0[i]) for i in np.arange(len(np.unique(C)))])

order = np.argsort(sz)
order = list(order)
order.reverse()
order = np.array(order)

for i in range(len(order)):
    Co[C==cmap0[order][i]]=len(order)-i

# Beat the array into the right shape
Co = Co.reshape((1,Co.shape[0])) # For nifti Masking
Co = [float(i) for i in Co[0]]
Co = np.array(Co)
#%% Save the NiFTi with cluster assignments
ofn= os.path.join(root,'Cluster_results','{}-cid6-test.nii.gz'.format(sub))
#oDS = copy.deepcopy(ds)
#oDS.samples=Co
#map2nifti(oDS).to_filename(ofn)
nifti = masker.inverse_transform(Co)
print(nifti)
nifti.to_filename(ofn)

#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
#%%
C = BNP.predict(data) # 
plt.show()
print(set(C))
#print(BNP.means_)
#print(BNP.precisions_)
print(BNP.converged_)
#BNP.weight_concentration_
print('Converged: {}'.format(BNP.converged_))
print('num clusters: {}'.format(len(set(C))))
print('means')
print(BNP.means_[np.unique(C)].transpose())
print('variance')
print(1 / BNP.precisions_[np.unique(C)].transpose())
