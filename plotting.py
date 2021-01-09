#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:22:14 2020

@author: ehereman
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_linearmap/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_samec34network2/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogmapped_samec34network2/'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_3_diffnet/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_map2pat/'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_EOGtrain2pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_map41pat_deeper/'
direct1='/volume1/scratch/ehereman/results_featuremapping/spectromap_eogtoc34_diffnet_map41pat/'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_map2pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_diffnetwork_only1pat/'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_3_diffnet/'


#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_map1pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnet_map10pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_map10pat'
#direct1='/volume1/scratch/ehereman/results_transferlearning/transferlearning_c34toeog_EOGtrain5pat2'
direct1='/volume1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_diffnetwork2_eog5pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network2_map2pat/group15'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_diffnetTL_map1pat/group30'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network3_map1pat/group2'
direct1='/volume1/scratch/ehereman/results_transferlearning/transferlearning_c34toeog_EOGtrain1pat4/group23'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network_totalmass_map10pat/n19'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network5_map10pat/group1'
direct1='/volume1/scratch/ehereman/results_transferlearning/transferlearning_c34toeog_totalmass3_EOGtrain10pat/n9'
direct1='/volume1/scratch/ehereman/results_transferlearning/transferlearning_c34toeog_EOGtrain1pat4_subjnorm/group32'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_eogtoc34_samec34network6_map1pat/group6'
direct1='/volume1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_diffnetwork32_eog2pat/group1'
#direct1='/volume1/scratch/ehereman/results_featuremapping/featmatch_eogtoc34_samenetwork2_eog2pat/group1'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_adaptc34network2_subjnorm_map10pat/group1'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_adaptc34network12_subjnorm_map10pat/group2'
#direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_samec34network3_subjnorm_map10pat/group2'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_samec34network7_subjnorm_map10pat/group0'
direct1='/volume1/scratch/ehereman/results_featuremapping/posttrainingmap_cycle_samec34network1_unmatch_subjnorm_map10pat/group1'

#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_adaptednetwork12_map10pat/group0'
#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_samenetwork_subjnorm'
#direct1='/volume1/scratch/ehereman/results_featuremapping/trainclasslayer_eogorig_adaptednetwork2_map10pat'

#direct1='/volume1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork3_eog10pat/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork3/'
#direct1='/volume1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_samenetwork4_eog10pat/group3'
#direct1='/volume1/scratch/ehereman/results_featuremapping/fmandclasstraining_eogtoc34_diffnetwork6_eog10pat/group0'

file1= 'eval_result_log.txt'
file2= 'test_result_log.txt'
acc_list=[]

for file in [file1,file2]:
    file1= open(os.path.join(direct1, file), "r")
    endfile = False
    acc=[]
    while not endfile:
        a=(file1.readline())
        a=a.split()
        if len(a)==0:
            endfile=True
        else:
            a= float(a[-2])
            acc.append(a)
            
    #acc = np.convolve(acc, np.ones((20,))/20, mode='valid')

    acc_list.append(np.array(acc))
#fig=plt.figure()

#labels=['shared weights', 'different weights', 'class weighted loss'] #
labels = ['Validation set','Test set']
colors=['k','r','c','m','b','g']#'k',
#plt.xlabel('Train step (10^3)')
#plt.figure()
plt.ylabel('Accuracy')
for i in range(len(acc_list)):
    accs= np.array(acc_list[i])
    plt.plot(np.array(range(len(accs)))*200, accs, label= labels[i],c=colors[i], alpha=.4) #*0.5 , linestyle='--'
a=acc_list[0]
pt=max(enumerate(a),key=itemgetter(1))
mx=pt[1]
plt.scatter(pt[0]*200, pt[1],marker='o',color='k')
plt.legend()
plt.title('')
#plt.savefig('regr_ss1-3')
plt.show()

