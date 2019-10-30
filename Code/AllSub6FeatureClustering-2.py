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


root= '/gsfs0/data/poskanzc/MVPN/analysis/net_results/subject_images/'
code_root = '/gsfs0/data/poskanzc/BC-CP-Computational-Profiles/'


#code_root = '/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/'
#root = '/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Data/subject_images/'

sub=['sub-01','sub-02','sub-03','sub-04','sub-05','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
#sub='sub-01'
#mask_filename = '/Users/craigposkanzer/Documents/MVPN-Data/brain_mask_bool.nii.gz'
#fmri_filename = '/Users/craigposkanzer/Documents/MVPN-Data/5layer/5layer_images/lin_sub-01.nii'
f_temp = ['{}_lin_img.nii.gz','{}_nl_img.nii.gz','dense_images/{}_5dense_lin_img.nii.gz','dense_images/{}_5dense_nl_img.nii.gz','5layer_images/{}_5layer_lin_img.nii.gz','5layer_images/{}_5layer_nl_img.nii.gz']
mfn = root+'brain_mask_bool.nii.gz'


masker = NiftiMasker(mask_img=mfn, standardize=True)

def get_ds(s):
    ''' Returns feat by voxels matrix for input subject '''
    # Grab first scan
    fn = root+f_temp[0].format(s) # Filename
    ds = masker.fit_transform(fn)

    # Append the other scanss
    for i in np.arange(1,len(f_temp)):
        fn = root+f_temp[i].format(s)
        #ads = fmri_dataset(fn,mask=mfn)
        ads = masker.fit_transform(fn)
        ds = np.vstack((ds,ads))
        
    return ds.transpose()

oDS = get_ds(sub[0])

# First subject DS
ds = get_ds(sub[0])
# Append other subject DS

for s in range(1,len(sub)):
    ds = np.vstack((ds,get_ds(sub[s])))


print('Datasets stacked')

data = np.array(copy.deepcopy(ds))
#data=data.transpose()
#X : array-like, shape (n_samples, n_features)
model = mixture.BayesianGaussianMixture(max_iter=100000,
                                      n_components=100,covariance_type='full',
                                      tol=0.0001,
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
#%% Map Back out to individual subjects

#C = [np.random.randint(500) for _ in range(C.shape[1])]
dishout = np.reshape(C,(len(sub),oDS.shape[0]))
analysis_name = 'subs-concat-1'
for i in range(len(sub)):
    ofn = os.path.join(code_root,'Results','sub{}-{}'.format(i,analysis_name)+'.nii')
    Co = dishout[i,:]
    # Beat the array into the right shape
    Co = Co.reshape((1,Co.shape[0])) # For nifti Masking
    Co = [float(i) for i in Co[0]]
    Co = np.array(Co)
    
    nifti = masker.inverse_transform(Co)
    print(nifti)
    nifti.to_filename(ofn)

print('ALL DONE')