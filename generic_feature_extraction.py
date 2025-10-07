
#
# Implementation of feature extraction for Tissue-specific Standardized Supervoxel Prediction (TS-SSP)
#
# Author: Johan Ofverstedt
#

import numpy as np

#
# To use this code, load the images (supervoxel/SLIC image, the deformed fat fraction image, jacobian determinant image)
# read the voxel size/volume, and convert the images into numpy arrays and call the functions in this module.
# This code assumes that all the images are the same array size as the supervoxel image, which
# it should be if the images are registered to a template image.
#
# Both a 3-tissue group configuration or the 1-tissue group configuration is provided in this module.
# Since all tissue group assignments are binary, here we do not use the weighted mean/sd computations
# from the paper for now.
#
# Fat fractions are assumed to be in the range [0, 1]. If values are in a different range, renormalize
# the images into this range before calling the functions in this module.
#
# Merge the resulting feature arrays into a single feature array for the whole dataset
# for further processing with the ts_ssp module.
#

# helper function to obtain a bounding box for a given mask
def get_bbox_3d(mask):
    mask_0 = np.any(mask, axis=(1, 2))
    if np.any(mask_0) == False:
        # the mask is empty
        return (None, None)

    mask_1 = np.any(mask, axis=(0, 2))
    mask_2 = np.any(mask, axis=(0, 1))
        
    min_0, max_0 = np.where(mask_0)[0][[0, -1]]
    min_1, max_1 = np.where(mask_1)[0][[0, -1]]
    min_2, max_2 = np.where(mask_2)[0][[0, -1]]
    
    return (min_0, min_1, min_2), (max_0+1, max_1+1, max_2+1)

def weighted_mean_and_sd(wmask, value, return_total_weight=False):
    # This function computes the weighted mean and (unbiased) standard deviation
    # given an array of weights (wmask) and an array of values.
    # Optional: return the total weight computed in the function as a third output
    
    wmask = wmask.astype(np.float32)
    val_sum = np.sum(wmask * value)
    weight_sum = np.sum(wmask)
    if weight_sum < 1e-15:
        if return_total_weight:
            return np.nan, np.nan, weight_sum
        else:
            return np.nan, np.nan
    wmean = val_sum / weight_sum
    
    weight_2_sum = np.sum(wmask**2)
    if np.abs(weight_sum - 1.0) < 1e-7 and np.abs(weight_2_sum - 1.0) < 1e-7:
        # if both the weight sum and squared weight sum is close to 1, the result is nan
        wsd = np.nan
    else:
        sd_denom = 1.0 / (weight_sum - weight_2_sum/weight_sum)
        wsd = np.sqrt(sd_denom * np.sum(wmask*((value-wmean)**2)))
    if return_total_weight:
        return wmean, wsd, weight_sum
    else:
        return wmean, wsd

def extract_tissue_specific_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, tissue_masks, single_voxel_volume=1.0, bbox_cache=None, return_bbox_cache=True):
    N_tissue_groups = len(tissue_masks)
    
    if bbox_cache is None:
        bbox_cache_output = []
    else:
        bbox_cache_output = bbox_cache
        
    features = []
    for i, ind in enumerate(supervoxel_indices):              
        if bbox_cache is None:
            supervoxel_mask_i = supervoxel_image==ind
        
            bb_mn, bb_mx = get_bbox_3d(supervoxel_mask_i)
            bbox_cache_output.append((bb_mn, bb_mx))

            if bb_mn is not None:
                supervoxel_mask_i = supervoxel_mask_i[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
        else:
            bb_mn, bb_mx = bbox_cache[i]
    
            # extract the supervoxel mask within the bounding box only
            if bb_mn is not None:
                supervoxel_mask_i = supervoxel_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]==ind
        
        if bb_mn is None:
            features.append([[np.nan, 0.0, np.nan, 0.0, np.nan, np.nan] for _ in range(N_tissue_groups)])
            continue
                          
        # get all subimages (ff, jacdet and tissue masks) within the supervoxel bounding box
               
        deformed_ff_image_i = deformed_ff_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
        jacdet_image_i      = jacdet_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]

        feat_local = []
        for j in range(N_tissue_groups):
            tissue_mask_i = tissue_masks[j][bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
       
            # get the tissue masks within the current supervoxel
        
            tissue_mask_i = supervoxel_mask_i.astype(np.float32) * tissue_mask_i
       
            # Compute FF features and voxel count
            feature_mean_ff, feature_sd_ff, feature_count = weighted_mean_and_sd(tissue_mask_i, deformed_ff_image_i, True)
        
            # Compute JacDet features
            feature_mean_jacdet, feature_sd_jacdet = weighted_mean_and_sd(tissue_mask_i, jacdet_image_i, False)

            # Compute Total Volume features
            if feature_count < 1e-7:
                feature_total_volume = 0.0
            else:        
                feature_total_volume = feature_mean_jacdet * feature_count * single_voxel_volume
            
            feat_local.append([feature_mean_ff, feature_total_volume, feature_mean_jacdet, feature_count, feature_sd_ff, feature_sd_jacdet])
        
        features.append(feat_local)
       
    features = np.array(features, np.float32)
    # features should now be an array of size (#supervoxels, #tissue groups, #features)
     
    # Concatenate the features along the tissue group axis
    features = np.expand_dims(features, 0)
     
    # features is now an array with the dimensions: (subject, supervoxel, tissue group, feature)
    # features from different subjects may now be concatenated along axis 0 to form a cohort feature-array
    
    if return_bbox_cache:
        return features, bbox_cache_output
    else:
        return features
        
def extract_single_tissue_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=1.0, bbox_cache=None, return_bbox_cache=True):

    if bbox_cache is None:
        bbox_cache_output = []
    else:
        bbox_cache_output = bbox_cache

    features = []
        
    for i, ind in enumerate(supervoxel_indices):
        if bbox_cache is None:
            supervoxel_mask_i = supervoxel_image==ind
        
            bb_mn, bb_mx = get_bbox_3d(supervoxel_mask_i)
            bbox_cache_output.append((bb_mn, bb_mx))

            if bb_mn is not None:
                supervoxel_mask_i = supervoxel_mask_i[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
        else:
            bb_mn, bb_mx = bbox_cache[i]
    
            # extract the supervoxel mask within the bounding box only
            if bb_mn is not None:
                supervoxel_mask_i = supervoxel_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]==ind
        
        deformed_ff_image_i = deformed_ff_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
        jacdet_image_i      = jacdet_image[bb_mn[0]:bb_mx[0], bb_mn[1]:bb_mx[1], bb_mn[2]:bb_mx[2]]
       
        masked_ff_image = deformed_ff_image_i[supervoxel_mask_i]
        masked_jacdet_image = jacdet_image_i[supervoxel_mask_i]
        
        feature_mean_ff = np.mean(masked_ff_image)
        feature_sd_ff = np.std(masked_ff_image, ddof=1)
               
        feature_mean_jacdet = np.mean(masked_jacdet_image)
        feature_sd_jacdet = np.std(masked_jacdet_image, ddof=1)

        features.append([[feature_mean_ff, feature_mean_jacdet, feature_sd_ff, feature_sd_jacdet]])
        
    features = np.array(features)
     
    # Introduce a new subject dimension
    features = np.expand_dims(features, 0)
     
    # features is now an array with the dimensions: (subject, supervoxel, tissue group which is a singleton dimension, feature)
    # features from different subjects may now be concatenated along axis 0 to form a cohort feature-array
    
    if return_bbox_cache:
        return features, bbox_cache_output
    else:
        return features
        
def test_feature_extraction(seed=1000):
    # test the feature extraction code with randomly generated images
    
    np.random.seed(seed)
    
    supervoxel_image = np.zeros((4, 4, 4), dtype=np.int32)
    supervoxel_image[0:2, 0:2, 0:2] = 1
    supervoxel_image[2:4, 0:2, 0:2] = 2
    supervoxel_image[0:2, 2:4, 0:2] = 3
    supervoxel_image[2:4, 2:4, 0:2] = 4
    supervoxel_image[0:2, 0:2, 2:4] = 5
    supervoxel_image[2:4, 0:2, 2:4] = 6
    supervoxel_image[0:2, 2:4, 2:4] = 7
    supervoxel_image[2:4, 2:4, 2:4] = 8

    # test the bounding box function
    bb_mn, bb_mx = get_bbox_3d(supervoxel_image==4)
    assert(bb_mn[0] == 2)
    assert(bb_mn[1] == 2)
    assert(bb_mn[2] == 0)
    assert(bb_mx[0] == 4)
    assert(bb_mx[1] == 4)
    assert(bb_mx[2] == 2)

    bb_mn, bb_mx = get_bbox_3d(supervoxel_image==7)
    assert(bb_mn[0] == 0)
    assert(bb_mn[1] == 2)
    assert(bb_mn[2] == 2)
    assert(bb_mx[0] == 2)
    assert(bb_mx[1] == 4)
    assert(bb_mx[2] == 4)
            
    deformed_ff_image = np.random.rand(4, 4, 4)
    jacdet_image = np.power(2, 2*np.random.rand(4, 4, 4)-1)
    
    supervoxel_indices = np.unique(supervoxel_image)
    print('Supervoxel_indices: ', supervoxel_indices)

    features, bbox_cache = extract_tissue_specific_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=2.0)
    features1b, _ = extract_tissue_specific_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=2.0, bbox_cache=bbox_cache)
    features2, bbox_cache2 = extract_single_tissue_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=2.0)
    features2b, _ = extract_single_tissue_features_from_numpy_arrays(supervoxel_image, supervoxel_indices, deformed_ff_image, jacdet_image, single_voxel_volume=2.0, bbox_cache=bbox_cache2)
    
    # check that we get the same feature values back if we use cached bounding boxes or not
    assert(np.abs(features[np.invert(np.isnan(features))]-features1b[np.invert(np.isnan(features1b))]).sum() < 1e-15)
    assert(np.abs(features2[np.invert(np.isnan(features2))]-features2b[np.invert(np.isnan(features2b))]).sum() < 1e-15)
    
    print('--- SUPERVOXELS ---')
    print(supervoxel_image)
    print('--- FF ---')
    print(deformed_ff_image)
    print('--- JacDet ---')
    print(jacdet_image)
    print('--- FEATURES ---')
    print('Features shape (correct result (1, 8, 3, 6)): ', features.shape)
    print(features)       
    
    # test tissue-specific feature extraction
    assert(features.shape[0] == 1) # 1 subjects
    assert(features.shape[1] == 8) # 8 supervoxels
    assert(features.shape[2] == 3) # 3 tissue groups per supervoxel
    assert(features.shape[3] == 6) # 6 features per supervoxel
    for i in range(features.shape[1]):
        for j in range(features.shape[2]):
            cnt = features[0, i, j, 3]
            assert(not np.isnan(features[0, i, j, 3]))
            assert(cnt >= 0)
            assert(cnt <= 8)
            
            mean_ff = features[0, i, j, 0]
            totalvolume = features[0, i, j, 1]
            mean_jacdet = features[0, i, j, 2]

            if cnt == 0:
                assert(np.isnan(features[0, i, j, 0]))
                assert(not np.isnan(features[0, i, j, 1]))
                assert(np.isnan(features[0, i, j, 2]))
                assert(not np.isnan(features[0, i, j, 3]))
                assert(np.isnan(features[0, i, j, 4]))
                assert(np.isnan(features[0, i, j, 5]))
                
                assert(totalvolume < 1e-15)
            elif cnt == 1:
                assert(not np.isnan(features[0, i, j, 0]))
                assert(not np.isnan(features[0, i, j, 1]))
                assert(not np.isnan(features[0, i, j, 2]))
                assert(np.isnan(features[0, i, j, 4]))
                assert(np.isnan(features[0, i, j, 5]))
                
                assert(mean_ff >= 0.0)
                assert(mean_ff <= 1.0)
                assert(mean_jacdet >= 0.5)
                assert(mean_jacdet <= 2.0)
            else:
                assert(not np.isnan(features[0, i, j, 0]))
                assert(not np.isnan(features[0, i, j, 1]))
                assert(not np.isnan(features[0, i, j, 2]))
                assert(not np.isnan(features[0, i, j, 4]))
                assert(not np.isnan(features[0, i, j, 5]))

                assert(mean_ff >= 0.0)
                assert(mean_ff <= 1.0)
                assert(mean_jacdet >= 0.5)
                assert(mean_jacdet <= 2.0)

            # minimum possible volume is 0 and maximum volume possible is 8*2*2=32
            print(totalvolume, mean_jacdet, cnt)
            assert(totalvolume >= 0.0)
            assert(totalvolume - 1e-15 <= 32)

    # single tissue tests
    assert(features2.shape[0] == 1) # 1 subjects
    assert(features2.shape[1] == 8) # 8 supervoxels
    assert(features2.shape[2] == 1) # 1 tissue group per supervoxel
    assert(features2.shape[3] == 4) # 4 features per supervoxel
    for i in range(features2.shape[1]):
        for j in range(features2.shape[2]):
            mean_ff = features2[0, i, j, 0]
            mean_jacdet = features2[0, i, j, 1]
            sd_ff = features2[0, i, j, 2]
            sd_jacdet = features2[0, i, j, 3]
            
            assert(not np.isnan(mean_ff))
            assert(not np.isnan(mean_jacdet))
            assert(not np.isnan(sd_ff))
            assert(not np.isnan(sd_jacdet))
            
            assert(mean_jacdet >= 0.5)
            assert(mean_jacdet <= 2.0)
            assert(mean_ff >= 0.0)
            assert(mean_ff <= 1.0)
            
            assert(sd_ff >= 0.0)
            assert(sd_jacdet >= 0.0)

    print('All tests passed!')
        
if __name__ == '__main__':
    for s in range(1000):
        test_feature_extraction(s)
   
        
    
            
        
    
    
