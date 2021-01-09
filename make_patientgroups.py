#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a dictionary of random patients for different numbers of patients
@author: ehereman
"""


di= dict.fromkeys([1,2,5,10])
for i in [1,2,5,10]:
    randper=np.random.permutation(np.arange(files_folds['train_sub'].shape[1]))
    nbgroup= int(math.floor(41/i))
    groups=[randper[j*i:(j+1)*(i)] for j in range(nbgroup)]
    if len(randper)> nbgroup*i:
        groups[-1]=np.concatenate((groups[-1],randper[nbgroup*i:]))
    di[i]=groups
 np.save('patient_groups',di)