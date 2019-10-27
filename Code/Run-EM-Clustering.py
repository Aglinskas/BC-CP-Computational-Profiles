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
mfn = root+'Data/brain_mask_bool.nii'


# Grab the first scan
fn = root+f_temp[0].format(sub) # Filename
ds = fmri_dataset(fn,mask=mfn) # Dataset
oDS = copy.deepcopy(ds)

# Append the other scanss
for i in np.arange(1,len(f_temp)):
    fn = root+f_temp[i].format(sub)
    ads = fmri_dataset(fn,mask=mfn)
    ds = vstack((ds,ads))
print('Datasets stacked')
    
# Harry potter levels of transmutation
data = np.array(copy.deepcopy(ds.samples))
data=data.transpose()

    #Parameters (for BNP)
    #----------
    #X : array-like, shape (n_samples, n_features)
    #    List of n_features-dimensional data points. Each row
    #    corresponds to a single data point.
#data=data.reshape(-1,1) # Numpy is being a little bitch about vector shapes
#niter=100000 # takes ~30min, converges
model = mixture.BayesianGaussianMixture(max_iter=100000,
                                      n_components=100,covariance_type='full',
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
C = BNP.predict(data) 
print('num clusters: {}'.format(len(set(C))))#

#%%
# Arrange Clusters to make a bit more sense
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
    
#%% Save the NiFTi with cluster assignments
ofn= os.path.join(root,'Results','{}-cid6-2.nii'.format(sub))
#oDS = copy.deepcopy(ds)
oDS.samples=Co
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