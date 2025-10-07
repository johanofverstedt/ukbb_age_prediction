
#
# Implementation of Tissue-specific Standardized Supervoxel Prediction
#
# Author: Johan Ofverstedt
#

from threading import local
import sklearn.cross_decomposition
import sklearn.metrics
import sklearn.neural_network
import numpy as np
import scipy
import scipy.stats
import sklearn
import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import sklearn.neighbors
import sklearn.decomposition
import sklearn.feature_selection
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import joblib
import time
import os

CODE_VERSION = 1

###
### --- Tissue-specific Standardized Supervoxel Prediction Main Class ---
###

# Utilities for saving and loading a model

def load_model(path):
    return joblib.load(path)

def save_model(path, model):
    joblib.dump(model, path)

# Helper functions for component ranking and selection

def abs_pearson_score(X, y):
    if y.ndim == 1:
        return np.abs(sklearn.feature_selection.r_regression(X, y))
    else:
        for i in range(y.shape[1]):
            r = np.abs(sklearn.feature_selection.r_regression(X, y[..., i]))
            if i == 0:
                result = r
            else:
                result += r
        return result / y.shape[1]   

def ranked_filtering(xs, y, n):
    if n is None or n >= xs.shape[0]:
        n = 'all'
    feat_sel = sklearn.feature_selection.SelectKBest(score_func = abs_pearson_score, k = n)
    feat_sel.fit(xs, y)
    return feat_sel.get_support()

class TS_SSP:
    def __init__(self, n_comp, n_comp2, reg, c = 3.0, nan_threshold=0.1, model_name='unnamed_model'):
        # n_comp is the number of PCA components used in the PCA.
        # n_comp2 is the number of PCA components selected out of the n_comp PCA compoments.
        # reg is the L2 regularization parameter. If set to None or 0, no regularization is used.
        # c is the clipping threshold at which the features are truncated (in standard deviations), default: 3.0.
        # nan_threshold controls how many subjects in the training set is allowed to be missing a feature
        #   until it is removed entirely.
        # model_name is a string that can be used to identify the model context later, and is saved along with the model.
        self.n_comp = n_comp
        self.n_comp2 = n_comp2
        self.reg = reg
        self.c = c
        self.nan_threshold = nan_threshold

        self.model_name = model_name
        self.code_version = CODE_VERSION

        self.orig_feat_num = None
        self.orig_feat_shape = None
        self.n_training_samples = None
        self.meta_data = None

    def fit(self, x, y):
        # Fit the regression from a feature vector x of shape (samples, features_1, ..., features_n)
        # to a vector (samples,) of y values
        self.orig_feat_num = np.prod(x.shape[1:])
        self.orig_feat_shape = x.shape[1:]
        self.n_training_samples = x.shape[0]

        x = x.reshape((x.shape[0], -1))

        # Find the features that should be masked out (True means the feature is kept, False discarded)
        # A nan_threshold value of 0.9 means that if less than 90% of the samples are missing that feature,
        # the feature is retained.
        num_of_missing_features = np.sum(np.isnan(x).astype(np.float32), axis=0, keepdims=False)
        self.mask = num_of_missing_features < self.nan_threshold*x.shape[0]

        # Mask out the missing features
        x = x[:, self.mask]

        self.x_mean = np.nanmean(x, axis=0, keepdims=True)
        self.x_std = np.clip(np.nanstd(x, axis=0, ddof=1, keepdims=True), 1e-7, None)
        self.y_mean = np.nanmean(y, axis=0, keepdims=True)
        self.y_std = np.clip(np.nanstd(y, axis=0, ddof=1, keepdims=True), 1e-7, None)

        # Apply z-score normalization
        x_z = (x-self.x_mean) / self.x_std
        y_z = (y-self.y_mean) / self.y_std

        # Replace missing features (represented as NaN) with 0.0
        x_z = np.nan_to_num(x_z, copy=False, nan=0.0)
        
        # Apply clipping of the features
        if self.c is not None:
            x_z = np.clip(x_z, -self.c, self.c)

        n_comp = self.n_comp
        if n_comp is not None:
            n_comp = np.minimum(np.minimum(int(0.9 * x.shape[0]), int(0.9 * x[0, ...].size)), n_comp)
        n_comp2 = np.minimum(n_comp, self.n_comp2)

        print(f'--- Fitting PCA (components: {n_comp}) ---')
        if n_comp is not None:
            t1 = time.time()
            self.pca = sklearn.decomposition.PCA(n_components=n_comp, copy=False, whiten=False)
            x_z = self.pca.fit_transform(x_z)
            t2 = time.time()
            print(f'PCA time elapsed: {t2-t1:0.2f}s')

            print(f'--- PCA component filtering (components: {n_comp2}) ---')
            t1 = time.time()
            self.pca_filt = ranked_filtering(x_z, y_z, n_comp2) 
            t2 = time.time()
            print(f'PCA filtering time elapsed: {t2-t1:0.2f}s')

            x_z = x_z[:, self.pca_filt]
            print(f'Components filtered in: {np.sum(self.pca_filt.astype(np.int64))}')
        else:
            self.pca = None
            self.pca_filt = None
            
        if self.reg is None or self.reg <= 0.0:
            print(f'--- Fitting REGRESSION ---')
            self.model = sklearn.linear_model.LinearRegression()
        else:
            print(f'--- Fitting REGRESSION (regularization: {self.reg}) ---')
            self.model = sklearn.linear_model.Ridge(alpha=self.reg)
        
        t1 = time.time()
        self.model.fit(x_z, y_z)
        t2 = time.time()
        print(f'Model fitting time elapsed: {t2-t1:0.2f}s')
    
    def predict(self, x):
        # Predict the value from a feature vector x of shape (samples, features_1, ..., features_n)

        feat_num = np.prod(self.orig_feat_shape)

        # Reshape the x features to the (samples, features) shape
        x = x.reshape((x.shape[0], feat_num))

        # Mask out the missing features
        x = x[:, self.mask]

        # Apply z-score normalization
        x_z = (x-self.x_mean) / self.x_std
        
        # Replace NaN with 0.0
        x_z = np.nan_to_num(x_z, copy=False, nan=0.0)
        
        # Apply clipping of the features
        if self.c is not None:
            x_z = np.clip(x_z, -self.c, self.c)
        
        if self.pca is not None:
            x_z = self.pca.transform(x_z)
        
        if self.pca_filt is not None:
            x_z = x_z[:, self.pca_filt]

        y_z = self.model.predict(x_z)

        # Undo z-score normalization
        y_z = y_z * self.y_std + self.y_mean

        return y_z        

    def saliency(self, zscores=True, normalize=True, normalization_percentile = 1.0):
        # Saliency (sensitivity) analysis providing a (features_1, ..., features_n)
        # vector of saliency values as an explanation of the regression model

        feat_num = np.prod(self.orig_feat_shape)

        coef = np.squeeze(self.model.coef_)
        if self.pca is not None:
            if self.pca_filt is not None:
                comp = self.pca.components_[self.pca_filt, :]
            else:
                comp = self.pca.components_[:, :]

            if np.sum(self.pca_filt.astype(np.int64)) == 1:
                comp = np.squeeze(comp)
                coef = coef * comp
            else:
                coef = np.dot(coef, comp)

        sal = np.zeros(feat_num)
        sal[self.mask] = coef

        # Undo the z-score normalization to map 
        if not zscores:
            x_std_unmasked = np.ones(feat_num)# self.x_std[self.mask]
            x_std_unmasked[self.mask] = np.squeeze(self.x_std)
            sal = sal * (np.squeeze(self.y_std) / np.squeeze(x_std_unmasked))
        
        # normalize saliency by dividing by the max absolute value
        # mapping the saliencies to the interval -1 to +1
        if normalize:
            if normalization_percentile >= 1.0:
                sal = sal / np.amax(np.abs(sal))
            else:
                sal = np.clip(sal / np.percentile(np.abs(sal), q=100*normalization_percentile), -1.0, 1.0)

        return sal.reshape(self.orig_feat_shape)

    def save(self, path):
        save_model(path, self)

    # Meta data is meant to be descriptive information containing information about the model
    # Example: (FEATURES, TISSUE_GROUPS)
    def get_meta_data(self):
        return self.meta_data

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, model_name):
        self.model_name = model_name

###
### --- Visualization ---
###

# Utilities for loading supervoxels using SimpleITK and mapping the saliency maps back to the spatial representation

def load_supervoxel_3D_data(supervoxel_file):
    supervoxel_sitk = sitk.ReadImage(supervoxel_file)
    supervoxel_arr = sitk.GetArrayFromImage(supervoxel_sitk)
    labels = np.sort(np.unique(supervoxel_arr))
    return supervoxel_arr, supervoxel_sitk, labels

def replace_labels_with_values_3d(supervoxel_arr, labels, values):
    # Prepend 0 to the values to represent the background (supervoxel 0)
    values_with_background = np.insert(values, 0, 0)   

    # Will find highest value in slic_arr to define the size of the map
    max_label = supervoxel_arr.max() 

    # Create the map for all supervoxels
    number_supervoxels_map = np.zeros(max_label+1, dtype=values_with_background.dtype)
    
    # Assign the values (with background) to the corresponding labels
    number_supervoxels_map[labels] = values_with_background
    number_supervoxels_map[0] = 0  # Ensure background label is correctly set to 0

    # Map the values based on the labels in slic_arr
    output_arr = number_supervoxels_map[supervoxel_arr]
    
    return output_arr

# Matplotlib colormap good for visualization of saliency maps

def get_saliency_cmap():
    cmap = np.asarray([
        [128, 0, 0], # Maroon
        [255, 0, 0], # Red
        [255, 165, 0], # Orange
        [255, 255, 255], # White
        [0, 255, 255], # Cyan
        [0, 0, 255], # Blue
        [0, 0, 128] # Deep Blue
    ])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('saliency', cmap[-1::-1, :]/255.0)
    return cmap

# Functions for generation of a saliency map
# where saliency gives a numeric value per supervoxel (and a tissue group)

def map_saliency_to_supervoxel_image_3d(saliency, supervoxels, supervoxel_labels):
    # Takes as input a 2D numpy array with shape (features, tissue groups)
    # a supervoxel integer-valued numpy array with BG=0.
    output_arr = replace_labels_with_values_3d(supervoxels, supervoxel_labels, saliency)

    body_arr = (supervoxels>0).astype(np.float32)
    
    return output_arr, body_arr

def map_saliency_to_supervoxel_image_3d_with_tissue_groups(saliency, supervoxels, supervoxel_labels, tissue_group_masks):
    # Takes as input a 2D numpy array with shape (features, tissue groups)
    # a supervoxel integer-valued numpy array with BG=0.
    # A number of tissue_group_mask arrays are provided with the same shape as supervoxels
    # where 0 means that a voxel does not belong to the tissue group, and 1 means that it does.
    # Here we assume that any given voxel does only belong to a single or no tissue-group
    # (partial belongingness would need to be handled differently in the future)

    output_arr = None
    for i in range(len(tissue_group_masks)):
        output_arr_i = replace_labels_with_values_3d(supervoxels, supervoxel_labels, saliency[:, i])
        output_arr_i = output_arr_i * tissue_group_masks[i]

        if output_arr is None:
            output_arr = output_arr_i
            body_arr = (supervoxels>0).astype(np.float32)
        else:
            output_arr = output_arr + output_arr_i
            body_arr = np.maximum(body_arr, (supervoxels>0).astype(np.float32))
    
    return output_arr, body_arr

def visualize_saliency_slice_with_colormap(saliency_volume, reference_image, body_volume, slice_idx, axis, spacing, output_path, fig_size=5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    saliency_slice = np.take(saliency_volume, slice_idx, axis=axis)
    reference_slice = np.take(reference_image, slice_idx, axis=axis)
    body_slice = np.take(body_volume, slice_idx, axis=axis)
    spacing_remaining = [spacing[i] for i in range(len(spacing)) if i != axis]
    aspect_ratio = spacing_remaining[1]/spacing_remaining[0]

    background = body_slice < 0.5
    saliency_slice = np.ma.masked_where(background, saliency_slice)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.set_size_inches(fig_size, fig_size*aspect_ratio)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ref_min = np.amin(reference_image)
    ref_max = np.amax(reference_image)
    ax1.imshow(reference_slice, cmap='gray', aspect=aspect_ratio, vmin=ref_min, vmax=ref_max)
    ax2.imshow(1.0-body_slice, cmap='gray', aspect=aspect_ratio, vmin=0, vmax=1)
    ax2.imshow(saliency_slice, cmap=get_saliency_cmap(), aspect=aspect_ratio, vmin=-1.0, vmax=1.0, alpha=1.0)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(output_path, pad_inches=0.0, bbox_inches='tight')
    plt.close()

def visualize_tissue_groups_slice_with_colormap(tissue_image, reference_image ,body_volume, slice_idx, axis, spacing, output_path, fig_size=5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tissue_slice = np.take(tissue_image, slice_idx, axis=axis)
    reference_slice = np.take(reference_image, slice_idx, axis=axis)
    body_slice = np.take(body_volume, slice_idx, axis=axis)
    spacing_remaining = [spacing[i] for i in range(len(spacing)) if i != axis]
    aspect_ratio = spacing_remaining[1]/spacing_remaining[0]

    background = body_slice < 0.5
    tissue_slice = np.ma.masked_where(background, tissue_slice)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.set_size_inches(fig_size, fig_size*aspect_ratio)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ref_min = np.amin(reference_image)
    ref_max = np.amax(reference_image)
    ax1.imshow(reference_slice, cmap='gray', aspect=aspect_ratio, vmin=ref_min, vmax=ref_max)
    ax2.imshow(1.0-body_slice, cmap='gray', aspect=aspect_ratio, vmin=0, vmax=1)
    ax2.imshow(tissue_slice, cmap='tab10', aspect=aspect_ratio, vmin=0.0, vmax=1.0, alpha=1.0)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(output_path, pad_inches=0.0, bbox_inches='tight')
    plt.close()

def visualize_regions_slice_with_colormap(region_image, reference_image, body_volume, slice_idx, axis, spacing, output_path, fig_size=5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    region_slice = np.take(region_image, slice_idx, axis=axis)
    reference_slice = np.take(reference_image, slice_idx, axis=axis)
    body_slice = np.take(body_volume, slice_idx, axis=axis)
    spacing_remaining = [spacing[i] for i in range(len(spacing)) if i != axis]
    aspect_ratio = spacing_remaining[1]/spacing_remaining[0]

    background = body_slice < 0.5
    region_slice = np.ma.masked_where(background, region_slice)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.set_size_inches(fig_size, fig_size*aspect_ratio)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ref_min = np.amin(reference_image)
    ref_max = np.amax(reference_image)
    ax1.imshow(reference_slice, cmap='gray', aspect=aspect_ratio, vmin=ref_min, vmax=ref_max)
    ax2.imshow(1.0-body_slice, cmap='gray', aspect=aspect_ratio, vmin=0, vmax=1)
    ax2.imshow(region_slice, cmap='plasma', aspect=aspect_ratio, vmin=0.0, vmax=1.0, alpha=1.0)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(output_path, pad_inches=0.0, bbox_inches='tight')
    plt.close()

def make_vertical_colorbar(sz_v, sz_h, output_path='./colorbar_vertical.png'):
    fig = plt.figure(figsize=(sz_h, sz_v))
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    import matplotlib.colorbar
    import matplotlib.colors

    cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=get_saliency_cmap(), norm =matplotlib.colors.Normalize(-1.0, 1.0), ticks=[-1.0, 0.0, 1.0])

    plt.savefig(output_path, bbox_inches='tight')

def make_horizontal_colorbar(sz_v, sz_h, output_path='./colorbar_horizontal.png'):
    fig = plt.figure(figsize=(sz_h, sz_v))
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    import matplotlib.colorbar
    import matplotlib.colors

    cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=get_saliency_cmap(), norm =matplotlib.colors.Normalize(-1.0, 1.0), ticks=[-1.0, 0.0, 1.0])

    plt.savefig(output_path, bbox_inches='tight')

    
    

