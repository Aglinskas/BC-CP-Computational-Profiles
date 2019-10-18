from mvpa2.tutorial_suite import *
import numpy as np

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from sklearn import mixture
import datetime

root= '/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/'
sub='sub-01'

f_temp = ['Data/1layer/{}_lin_img.nii','Data/1layer/{}_nl_img.nii','Data/5dense/lin_{}.nii','Data/5dense/{}_5dense_nl_img.nii','Data/5layer/lin_{}.nii','Data/5layer/{}_5layer_nl_img.nii']

fn = root+f_temp[1].format(sub)
mfn = root+'Data/brain_mask_bool.nii'
ds = fmri_dataset(fn,mask=mfn)

data = copy.deepcopy(ds.samples[0]) 

data=data.reshape(-1,1) # Numpy is being a little bitch about vector shapes
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
print('num clusters: {}'.format(len(set(C))))
C = BNP.predict(data) # 

#%%

ofn= root+'/Results/test3.nii'
oDS = copy.deepcopy(ds)
oDS.samples[0]=C
map2nifti(oDS).to_filename(ofn)

#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
#%%
C = BNP.predict(dt) # 
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