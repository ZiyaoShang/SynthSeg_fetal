
import nibabel as nib
import glob
import os
import numpy as np
from ext.lab2im.edit_volumes import align_volume_to_ref, crop_volume_around_region, pad_volume, resample_volume, \
    resample_volume_like, crop_volume, crop_volume_with_idx
from ext.lab2im.utils import get_volume_info, save_volume, get_list_labels, list_images_in_folder
from scipy.signal import convolve
from skimage import measure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def extract( 
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg",
    img_path = None,
    labels_all = None,
    inner_labels = None,
    n_clusters=3,
    save_plot=False):
    # [total volume, total surface area, surface to volumn, [6] relative volume of each structure, ... [6] relative
    # number of feayures must be 21
    print("extract()")
    seg_list = sorted(glob.glob(seg_path + '/*'))

    img_list = []
    if img_path is not None:
        img_list = sorted(glob.glob(img_path + '/*'))
        assert len(img_list) == len(seg_list)
    
    if labels_all is None:
        labels_all = get_list_labels(labels_dir=seg_path)
    assert inner_labels is not None
        

    all_features = np.zeros(21, dtype=float)
    for i in range(len(seg_list)):
        print("processing: " + str(seg_list[i].split('/')[-1].split('_')[0]))
        seg_vol, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i], return_volume=True, aff_ref=None, max_channels=10)

        if img_path is not None:
            img_vol, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True, aff_ref=None,max_channels=10)
            assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0] 
            assert (np.all(img_shp == seg_shp))
            assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), str(img_aff) + str(seg_aff)
            assert (img_n_dims == img_n_dims)
            assert (img_n_channels == seg_n_channels)
            assert (np.all(img_im_res == seg_im_res))
            assert np.sum(img_vol > 0) == np.sum(img_vol != 0)
            assert isinstance(img_vol[0, 0, 0], np.float64)
            assert isinstance(img_im_res[0], np.float32)

        assert isinstance(seg_vol[0, 0, 0], np.float64)
        assert isinstance(seg_im_res[0], np.float32)

        inner_labels = np.array(inner_labels).astype(np.float64)
        mask_whole = seg_vol == inner_labels[0]
        for lbl in inner_labels[1:]:
            mask_whole = np.logical_or(mask_whole, seg_vol == lbl)

        total_volume = float(np.sum(mask_whole))
        verts, faces, _, _ = measure.marching_cubes(mask_whole, method='lewiner') # TODO: is this valid? parameters?
        total_surface_area = float(measure.mesh_surface_area(verts, faces))
        assert (total_surface_area>0) and (total_volume>0)
        surface_to_volumn = total_surface_area / total_volume
        # print(total_surface_area)
        # print(total_volume)

        # all except bg and csf
        struct_vols = []
        struct_areas = []
        struct_surface_to_volumn = []
        for lb in inner_labels:
            mask_struct = seg_vol == lb
            verts, faces, _, _ = measure.marching_cubes(mask_struct, method='lewiner')
            struct_area = float(measure.mesh_surface_area(verts, faces))
            struct_vol = float(np.sum(mask_struct))
            struct_areas.append(struct_area)
            struct_vols.append(struct_vol)
            assert (struct_area>0) and (struct_vol>0)
            struct_surface_to_volumn.append(struct_area/struct_vol)
        struct_vols = np.array(struct_vols)
        struct_areas = np.array(struct_areas)
        struct_surface_to_volumn = np.array(struct_surface_to_volumn)
        struct_vols = struct_vols / np.sum(struct_vols)
        struct_areas = struct_areas / np.sum(struct_areas)

        # surface alternative: surface count
        # to be on the surface, a voxel must: 1, have at least one surrounding pixel that is the bg.
        # # 2, not be labelled as bg
        # convolved = convolve(mask, np.ones((3, 3, 3)), mode='valid')
        # convolved = np.pad(convolved, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        # surface_count = np.sum(np.logical_and(convolved < 27, mask))
        # print(surface_count)

        vol_feature = np.concatenate((np.array([total_volume]), np.array([total_surface_area]),np.array([surface_to_volumn]), struct_vols, struct_areas, struct_surface_to_volumn))
        all_features = np.vstack([all_features, vol_feature])

    raw_features = all_features[1:]
    # standardize feature sets
    normalized_data = MinMaxScaler().fit_transform(raw_features)
    normalized_data[:,0:12] *= 2
    print("np.std(normalized_data, axis=0)")
    print(np.std(normalized_data, axis=0))


    # dimentionality reduction with PCA
    pca = PCA(n_components=5)
    pca.fit(normalized_data)
    lowdim_features = pca.transform(normalized_data)
    print("np.std(lowdim_features, axis=0)")
    print(np.std(lowdim_features, axis=0))
    print("pca.components_")
    print(pca.components_) 
    print("pca.explained_variance_ratio_")
    print(pca.explained_variance_ratio_)

    # k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
    kmeans.fit(lowdim_features)
    clusters = kmeans.labels_
    # centers = kmeans.cluster_centers_
    print("clusters=")
    print(clusters)

    if save_plot:
        first_two = lowdim_features[:,2:]
        colors = {0: 'red', 1: 'green', 2: 'blue'}
        color_map = np.array([colors[cls] for cls in list(kmeans.labels_)])
        plt.figure(figsize=(8, 6))
        plt.scatter(first_two[:, 0], first_two[:, 1], edgecolors='w', linewidth=0.5, c=color_map, alpha=0.6)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter Plot of Data Points')
        plt.grid(True)
        plt.savefig('/home/zshang/SP/data/ZURICH/all_extra_label_bgsubd_centered_seg_train/img.png')

    # assign weights according to assigned cluster
    weight_per_cluster = 1.0 / n_clusters
    weights = []
    counts = np.bincount(clusters)
    for ind in range(len(clusters)):
        weights.append(weight_per_cluster / counts[clusters[ind]])

    print(weights)
    print(np.sum(weights))

    return weights 



# 1, the absolute brain volumn/surface may not be that useful because we always train on high-res
# extract(seg_path='/home/zshang/SP/data/ZURICH/all_extra_label_bgsubd_centered_seg_train', labels_all=np.array([0,10,11,12,13,1,2,3,4,5,6,7]), inner_labels=[2,3,4,5,6,7], n_clusters=3)