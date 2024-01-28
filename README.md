

# Fetal-SynthSeg
Domain shift has always been a crucial topic for medical image segmentation. It is a common issue that deep segmentation networks often exhibit performance problems when applied to data that deviates from what the model is trained on. For computer-assisted studies of fetal brain development and pathology, it is crucial that accurate and consistent segmentation schemes be used for data acquired in very different settings. In this project, we developed a Multi-centric developing brain MRI segmentation framework by adapting SynthSeg[https://github.com/BBillot/SynthSeg] so that it could be used on datasets with large domain shifts. Our modifications are mainly focused on training data preprocessing, data augmentation, and model inference. Our results show that the models we trained are able to generalize to subjects with a variety of developmental ages, image attributes (size, quality, etc), modality, pathology, and background noises. 

## Code 
The code to run the adapted version of training/validation/testing is located [here](https://github.com/ZiyaoShang/SynthSeg_fetal/tree/master/scripts/fetal_scripts). \
Project-specific pre/postprocessing helper functions are located [here](https://github.com/ZiyaoShang/SynthSeg_fetal/blob/master/scripts/fetal_scripts/helpers.py).

## Citation

This is based on the original Synthseg framework [https://github.com/BBillot/SynthSeg] for my semester project on fetal brain segmentation. More instructions can also be found in the original repo.

**Robust machine learning segmentation for large-scale analysisof heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]

**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559) | [bibtex](bibtex.bib) ]

