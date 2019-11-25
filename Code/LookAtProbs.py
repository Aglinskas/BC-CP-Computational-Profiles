#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:45:50 2019

@author: aidasaglinskas
"""  
from datetime import datetime
import pickle 
fn = '/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Results/model_version_57.pickle'
model = pickle.load(open(fn,'rb'))

from matplotlib import pyplot as plt
plt.plot(np.sort(model.weights_,ascending=False))

# Get the Data from the BNP scripts
# until this line 
data = np.array(copy.deepcopy(ds))


now = datetime.now
t0=now()
P = model.predict_proba(data)
C = model.predict(data)
print('time elapsed')
print(now()-t0)

plt.plot(P[601,:])