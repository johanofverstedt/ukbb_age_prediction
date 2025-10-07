
#
# Program for extracting supervoxel-features from a number of subjects in the UK Biobank.
# The program requires a subject id file and the stiched and registered MR neck-to-knee images
# to be structured in a specific way:
# The deformed ff (fat fraction) images are to be placed in a folder with the names: ff_{subject_id}.nii.gz
# The jacdet (Jacobian Determinant) images are to be placed in a folder with the names: jd_{subject_id}.nii.gz
# The subject ids to compute the features for is to be listed in a text file, one subject id per line (with line break '\n')
#
# Author: Johan Ofverstedt
#

import numpy as np
import SimpleITK as sitk
import os
import time
import argparse

import feature_extraction as fe

def load_supervoxel_file(path):
    supervoxel_image = sitk.ReadImage(path, sitk.sitkInt32)
    arr = sitk.GetArrayFromImage(supervoxel_image)
    supervoxel_indices = np.unique(arr)
    supervoxel_indices = [supervoxel_indices[i] for i in range(len(supervoxel_indices)) if supervoxel_indices[i] > 0]
    return supervoxel_image, arr, supervoxel_indices
    
def supervoxel_feature_extraction_from_files(supervoxel_path, ff_folder, jacdet_folder, subject_ids, out_folder, feature_set_name, single_tissue = False, ext = '.nii.gz', save_period=100):
    # load the supervoxels once and for all
    supervoxel_image, supervoxel_array, supervoxel_indices = load_supervoxel_file(supervoxel_path)
    
    all_features = []
    bbox_cache = None
    start_index = 0
    
    output_path = f'{out_folder}/{feature_set_name}.npz'
    os.makedirs(out_folder, exist_ok=True)
    
    if os.path.exists(output_path):
        in_features = np.load(output_path, allow_pickle=True)['arr_0']
        start_index = in_features.shape[0] # skip the first subjects we have already processed
        all_features.append(in_features)
        
    for i in range(start_index, len(subject_ids)):
        t1 = time.time()
        subj = subject_ids[i]
        
        ff_path = ff_folder + f'ff_{subj}{ext}'
        jacdet_path = jacdet_folder + f'jd_{subj}{ext}'
        
        ff_image = sitk.ReadImage(ff_image, sitk.sitkFloat32)
        jacdet_image = sitk.ReadImage(jacdet_path, sitk.sitkFloat32)
        ff_array = sitk.GetArrayFromImage(ff_image)
        jacdet_array = sitk.GetArrayFromImage(jacet_image)
        
        # compute the voxel size in mL
        voxel_size = np.prod(np.array(ff_image.GetSpacing())) / 1000.0
                
        if single_tissue:
            features_i, bbox_cache = fe.extract_single_tissue_features_from_numpy_arrays(supervoxel_array, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=voxel_size, bbox_cache=bbox_cache, return_bbox_cache=True)
        else:
            features_i, bbox_cache = fe.extract_tissue_specific_features_from_numpy_arrays(supervoxel_array, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=voxel_size, bbox_cache=bbox_cache, return_bbox_cache=True)
        
        all_features.append(features_i)

        t2 = time.time()
        print(f'Processed subject {i+1} with id {subj}. Time elapsed: {t2-t1}s.')
        if (i - start_index ) % save_period) == save_period-1:
            print(f'Saving features ({i} processed)')
            cat_features = np.concatenate(all_features, axis=0)
            np.savez_compressed(output_path, cat_features, allow_pickle=True)
            
    cat_features = np.concatenate(all_features, axis=0)   
    print(f'--- DONE --- Saving features ({cat_features.shape[0]} processed)')
    np.savez_compressed(output_path, cat_features, allow_pickle=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('supervoxel_path', help='The full path to the supervoxel image file')
    parser.add_argument('ff_folder', help='The folder containing the ff (fat fraction) image files with the file pattern ff_subj.nii.gz')
    parser.add_argument('jacdet_folder', help='The folder containing the jacdet (Jacobian Determinants) image files with the file pattern jd_subj.nii.gz')
    parser.add_argument('subject_id_path', help='The full path to a text file containing one subject id on each line.')
    parser.add_argument('single_tissue', action='store_true', help='Do feature extraction for a single tissue group')
    parser.add_argument('ext', default='.nii.gz', help='File extension of the image files')
 
    args = parser.parse_args()
    
    if not os.path.exists(args.supervoxel_path):
        print('Supervoxel image file not found. Aborting...')
        return
    
    if not os.path.exists(args.subject_id_path):
        print('Subject id file not found. Aborting...')
        return
    
    with open(args.subject_id_path) as file:
        subject_ids = [line.rstrip() for line in file]

    print(f'Number of subjects to extract features for: {len(subject_ids)}')
    
    supervoxel_feature_extraction_from_files(args.supervoxel_path, args.ff_folder, args.jacdet_folder, subject_ids, single_tissue = args.single_tissue, ext=args.ext)
   

