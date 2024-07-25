
import nibabel as nib
import glob
import os
import numpy as np
# from ext.lab2im.edit_volumes import align_volume_to_ref, crop_volume_around_region, pad_volume, resample_volume, \
#     resample_volume_like, crop_volume, crop_volume_with_idx
from ext.lab2im.utils import get_volume_info, save_volume, get_list_labels, list_images_in_folder
from scipy.signal import convolve
from skimage import measure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min



def extract( 
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg",
    img_path = None,
    labels_all = None,
    inner_labels = None,
    n_clusters=3,
    clustering_method='gmm',
    n_components=5,
    save_plot=False,
    gt_classes_2=None,
    gt_classes_3=None,
    accord_2=False,
    accord_3=False,
    accord_23=False,
    accord_exp=False,
    load = False,
    saved_features_path=None,
    fig_dir=None):
    # [total volume, total surface area, surface to volumn, [6] relative volume of each structure, ... [6] relative
    # number of feayures must be 21
    print("extract()")
    if not load:
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
        np.save(saved_features_path,raw_features)
    else:
        raw_features = np.load(saved_features_path)

    assert raw_features.shape == (len(glob.glob(seg_path + '/*')), 21)


    # standardize feature sets
    # print(raw_features.shape)
    normalized_data = MinMaxScaler().fit_transform(raw_features)
    assert normalized_data.shape == (len(glob.glob(seg_path + '/*')), 21)

    assert accord_2 + accord_3 + accord_23 + accord_exp == 1

    if accord_2:
        to_choose = np.argsort(classify_feat_importance_2(X=normalized_data,y=gt_classes_2))[-5:]
        print("top_indexes_2")
        print(to_choose)
        normalized_data[:, to_choose] *= 2

    if accord_3:
        to_choose = np.argsort(classify_feat_importance_3(X=normalized_data,y=gt_classes_3))[-5:]
        print("top_indexes_3")
        print(to_choose)
        normalized_data[:, to_choose] *= 2    
    
    if accord_23:
        to_choose2 = np.argsort(classify_feat_importance_2(X=normalized_data,y=gt_classes_2))[-5:]
        to_choose3 = np.argsort(classify_feat_importance_3(X=normalized_data,y=gt_classes_3))[-5:]
        to_choose = np.unique(np.append(to_choose2, to_choose3))
        print("top_indexes_23")
        print(to_choose)
        normalized_data[:, to_choose] *= 2

    if accord_exp:
        # pass
        # normalized_data[:, :12] *= 2
        normalized_data[:, [0,1,2,3,4,5,9,10,11,15,16,17]] *= 2
        # normalized_data[:, [19, 9, 18, 1, 0]] *= 2
        # normalized_data[:, [18, 17, 16, 12, 5]] *= 2
        # normalized_data[:, [0, 1, 5, 9, 12, 16, 17, 18, 19]] *= 2


    # standardize feature sets 
    # print("np.std(normalized_data, axis=0)")
    # print(np.std(normalized_data, axis=0))


    # dimentionality reduction with PCA
    pca = PCA(n_components=n_components)
    pca.fit(normalized_data)
    lowdim_features = pca.transform(normalized_data)
    assert lowdim_features.shape == (len(glob.glob(seg_path + '/*')), n_components)
    # print("np.std(lowdim_features, axis=0)")
    # print(np.std(lowdim_features, axis=0))
    # print("pca.components_")
    # print(pca.components_)
    print("pca.explained_variance_ratio_")
    print(pca.explained_variance_ratio_)

    # k-means clustering
    if clustering_method=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
        kmeans.fit(lowdim_features)
        clusters = kmeans.labels_
        assert np.all(clusters == kmeans.fit_predict(lowdim_features))
        centers = kmeans.cluster_centers_
        assert centers.shape[1] == lowdim_features.shape[1]
        closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        print("closest subject index for each cluster centroid (note that these indices start from zero)" +str(closest_inds))
        closest_inds = np.array(closest_inds)
        print(np.array(sorted(glob.glob(seg_path + '/*')))[closest_inds])
        
        print("clusters=")
        print(clusters)
        print("kmeans.inertia_")
        print(kmeans.inertia_)

    if clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(lowdim_features)
        clusters = gmm.predict(lowdim_features)
        centers = gmm.means_
        assert centers.shape[1] == lowdim_features.shape[1]
        closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        print("closest subject index for each cluster centroid (note that these indices start from zero)" +str(closest_inds))
        print("clusters=")
        print(clusters)

    if save_plot:
        if gt_classes_2 is not None:
            first_two = lowdim_features[:,:2]
            colors = {True: 'red', False: 'green'}
            color_map = np.array([colors[cls] for cls in list(gt_classes_2)])
            plt.figure(figsize=(8, 6))
            for s in range(first_two.shape[0]):
                plt.scatter(first_two[s, 0], first_two[s, 1], edgecolors='w', linewidth=0.5, c=color_map[s], s=0.1, alpha=0.6)
                plt.annotate(str(s+1), (first_two[s, 0], first_two[s, 1]), fontsize=6, ha='center', c=color_map[s])

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Scatter Plot of Data Points according to pathology')
            plt.grid(False)
            plt.savefig(fig_dir + 'img_path_acexp_01234591011151617_kmeans.png')

        if gt_classes_3 is not None:
            first_two = lowdim_features[:,:2]
            mx = np.max(gt_classes_3)
            mn = np.min(gt_classes_3)
            col_age = (gt_classes_3 - mn) / (mx - mn)
            # colors = [(1,0,0,), (0,0,1)]
            color_map = np.array([(cls, 0, 1-cls) for cls in list(col_age)])
            plt.figure(figsize=(8, 6))
            for s in range(first_two.shape[0]):
                plt.scatter(first_two[s, 0], first_two[s, 1], edgecolors='w', linewidth=0.5, c=color_map[s], s=1, alpha=0.6)
                plt.annotate(str(int(gt_classes_3[s])), (first_two[s, 0], first_two[s, 1]), fontsize=6, ha='center', c=color_map[s])

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Scatter Plot of Data Points according to age')
            plt.grid(False)
            plt.savefig(fig_dir + 'img_age_acexp_01234591011151617_kmeans.png')

        first_two = lowdim_features[:,:2]
        colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4:'yellow', 5:'pink', 6:'black', 7:'gray', 8:'orange', 9:'brown'}
        color_map = np.array([colors[cls] for cls in list(clusters)])
        plt.figure(figsize=(8, 6))
        plt.scatter(first_two[:, 0], first_two[:, 1], edgecolors='w', linewidth=0.5, c=color_map, alpha=0.6)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter Plot of Data Points')
        plt.grid(True)
        plt.savefig(fig_dir + 'img_10_cluster_acexp_01234591011151617_kmeans.png')

       
        # first_three = lowdim_features[:, :3]
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # sc = ax.scatter(first_three[:, 0], first_three[:, 1], first_three[:, 2], edgecolors='w', linewidth=0.5, c=color_map, alpha=0.6)

        # ax.set_xlabel('Feature 1')
        # ax.set_ylabel('Feature 2')
        # ax.set_zlabel('Feature 3')
        # ax.set_title('3D Scatter Plot of Data Points')
        # ax.grid(True)

        # plt.show()
        # plt.savefig(fig_dir + 'img_cluster_acexp_3d.png')

    # assign weights according to assigned cluster
    weight_per_cluster = 1.0 / n_clusters
    weights = []
    counts = np.bincount(clusters)
    print("number of templates in each cluster: ")
    print(counts)
    for ind in range(len(clusters)):
        weights.append(weight_per_cluster / counts[clusters[ind]])

    # print(weights)
    print(np.sum(weights))

    return weights 

def classify_feat_importance_2(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf = RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_leaf=10, min_samples_split=10, random_state=42, max_features='sqrt')
    rf.fit(X, y)
    y_pred = rf.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}')
    print("feature_importances, the higher the more important")
    print(rf.feature_importances_)
    return rf.feature_importances_


def classify_feat_importance_3(X, y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    rf = RandomForestRegressor(n_estimators=30, max_depth=5, min_samples_leaf=10, min_samples_split=10, random_state=42, max_features='sqrt')
    rf.fit(X, y)
    y_pred = rf.predict(X)

    accuracy = mean_squared_error(y, y_pred)
    print(f'Accuracy: {accuracy}')
    print("feature_importances, the higher the more important")
    print(rf.feature_importances_)
    return rf.feature_importances_

# 1, the absolute brain volumn/surface may not be that useful because we always train on high-res

extract(save_plot=False, seg_path='/Users/ziyaoshang/Desktop/FeTA_synthetic', labels_all=np.array([0,1,2,3,4,5,6,7]), inner_labels=[2,3,4,5,6,7], n_clusters=3, clustering_method='gmm', gt_classes_2=None, gt_classes_3=None, n_components=3, accord_2=False, accord_3=False, accord_23=False, accord_exp=True, load=True, saved_features_path='/Users/ziyaoshang/Desktop/trash/synth/features.npy', fig_dir='/Users/ziyaoshang/Desktop/trash/synth/')

# extract(save_plot=False, seg_path='/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/all_extra_label_centered_seg_train', labels_all=np.array([0,1,2,3,4,5,6,7]), inner_labels=[2,3,4,5,6,7], n_clusters=3, clustering_method='kmeans', gt_classes_2=pathology, gt_classes_3=age, n_components=5, accord_2=False, accord_3=False, accord_23=False, accord_exp=True, load=True, saved_features_path='/Users/ziyaoshang/Desktop/trash/tempp/features.npy', fig_dir='/Users/ziyaoshang/Desktop/trash/tempp/')

