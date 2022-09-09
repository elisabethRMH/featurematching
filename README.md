# Feature Matching for sleep staging networks

This is the code for the paper "Feature matching as improved transfer learning technique for wearable EEG" [3] available on https://www.sciencedirect.com/science/article/pii/S1746809422004839 (older preprint version on arXiv: https://arxiv.org/abs/2201.00644).

## How to use feature matching

Step 1: Download data. Datasets on which this method were tested include the cEEGrid dataset [1], the MASS dataset (available at http://ceams-carsm.ca/mass/) [2] and the SleepUZL dataset [3]. Of these datasets, only the MASS dataset is publicly available.

Step 2: Pre-training is performed using SeqSleepNet and the ARNN model from https://github.com/pquochuy/SeqSleepNet [4]. 

Step 3: Use the training code: train_fmandclassmodelSqSlNet_ceegrid.py (for SeqSleepNet) or train_fmandclassmodel_ceegrid.py (for ARNN). 
Any other model architecture can be plugged in easily.

Step 4: Use the testing code: test_fmandclassmodelSqSlNet_ceegrid.py (for SeqSleepNet) or test_fmandclassmodel_ceegrid.py (for ARNN)

## Contact
For any questions:

Elisabeth Heremans

Contact details: elisabeth.heremans@kuleuven.be

## References

[1] A. Sterr, J. K. Ebajemito, K. B. Mikkelsen, M. A. Bonmati-Carrion, N. Santhi, C. della Monica, L. Grainger, G. Atzori, V. Revell, S. Debener, D.-J. Dijk, and M. De Vos, “Sleep EEG Derived From Behind-the-Ear Electrodes (cEEGrid) Com- pared to Standard Polysomnography: A Proof of Concept Study,” Frontiers in Human Neuro- science, vol. 12, p. 452, 11 2018. [Online]. Available: https://www.frontiersin.org/article/10.3389/fnhum.2018.00452/full.

[2] O'Reilly, C., Gosselin, N., Carrier, J. and Nielsen, T. (2014), Montreal Archive of Sleep Studies: an open-access resource for instrument benchmarking and exploratory research. J Sleep Res, 23: 628-635. https://doi.org/10.1111/jsr.12169.

[3] E. R. M. Heremans et al., “Feature matching as improved transfer learning technique for wearable EEG,” Biomed. Signal Process. Control, vol. 78, p. 104009, Sep. 2022, doi: 10.1016/J.BSPC.2022.104009.

[4] Phan, H., Andreotti, F., Cooray, N., Chen, O. Y., & De Vos, M. (2019). SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging. IEEE transactions on neural systems and rehabilitation engineering : a publication of the IEEE Engineering in Medicine and Biology Society, 27(3), 400–410. https://doi.org/10.1109/TNSRE.2019.2896659

