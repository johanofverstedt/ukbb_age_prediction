# UK Biobank Age Prediction (TS-SSP)
Code for the UK Biobank age prediction method "tissue-specific standardized supervoxel-based predictions" (TS-SSP)

biological_age_tools.py: Biological age code for debiasing the predicted ages, computing age gaps, etc is found.
generic_feature_extraction.py: Code for extracting features from tissue-specific standardized supervoxels.
feature_extraction_program.py: A python program that takes a supervoxel image, folders of fat fraction/jacobian determinant images, a list of subject ids and extracts an npz file of tissue-specific supervoxel features from each subject images.
ts_ssp.py: The tissue-specific standardized supervoxel-based prediction machine learning code. Given a set of features extracted from images using the feature extraction program, we can predict some parameter such as morphological age using the machine learning pipeline in this file. Follows the basic interface of sklearn with a fit(x, y), predict(x) pair of methods.

