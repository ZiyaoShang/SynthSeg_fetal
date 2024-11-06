
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def process_feratures( 
    raw_features=None,
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
    fig_dir=None):

    # print(raw_features.shape) must be [n_samples, n_features]
    normalized_data = MinMaxScaler().fit_transform(raw_features)
    assert normalized_data.shape == (raw_features.shape[0], 21)

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
        print("normalized_data[:, [0,1,2,3,4,5,9,10,11,15,16,17]] *= 2")
        normalized_data[:, [0,1,2,3,4,5,9,10,11,15,16,17]] *= 2
        # normalized_data[:, [19, 9, 18, 1, 0]] *= 2
        # normalized_data[:, [18, 17, 16, 12, 5]] *= 2
        # normalized_data[:, [0, 1, 5, 9, 12, 16, 17, 18, 19]] *= 2


    # standardize feature sets 
    # print("np.std(normalized_data, axis=0)")
    # print(np.std(normalized_data, axis=0))


    # dimentionality reduction with PCA 
    assert normalized_data.shape == (raw_features.shape[0], 21)
    pca = PCA(n_components=n_components)
    pca.fit(normalized_data) # (n_samples, n_features)
    lowdim_features = pca.transform(normalized_data) # (n_samples, n_features)
    assert lowdim_features.shape == (raw_features.shape[0], n_components)
    # print("np.std(lowdim_features, axis=0)")
    # print(np.std(lowdim_features, axis=0))
    # print("pca.components_")
    # print(pca.components_)
    print("pca.explained_variance_ratio_")
    print(pca.explained_variance_ratio_)

    # k-means clustering
    if clustering_method=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
        kmeans.fit(lowdim_features) #(n_samples, n_features)
        clusters = kmeans.labels_
        centers = kmeans.cluster_centers_

        assert np.all(clusters == kmeans.fit_predict(lowdim_features))
        assert np.all(np.abs(centers - kmeans.cluster_centers_) < 0.0001)
        assert centers.shape[1] == lowdim_features.shape[1]

        # closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        # print("closest subject index for each cluster centroid (note that these indices start from zero)" + str(closest_inds))
        # closest_inds = np.array(closest_inds)
        # print(np.array(sorted(glob.glob(seg_path + '/*')))[closest_inds])
        # print("clusters=")
        # print(clusters)
        print("kmeans.inertia_")
        print(kmeans.inertia_)

    if clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(lowdim_features) # (n_samples, n_features)
        clusters = gmm.predict(lowdim_features) # (n_samples, n_features)
        centers = gmm.means_

        assert centers.shape[1] == lowdim_features.shape[1]
        # closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        # print("closest subject index for each cluster centroid (note that these indices start from zero)" +str(closest_inds))
        # print("clusters=")
        # print(clusters)
    
    assert len(clusters) == lowdim_features.shape[0]

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
        plt.savefig(fig_dir + 'img_10_cluster_3_compo_acexp_01234591011151617_gmm.png')

    return clusters


def extract_pret( 
    seg_path = None,
    inner_labels = None,
    n_clusters_dhcp=10, 
    n_clusters_zurich=10,
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
    path_to_save_weights=None,
    fig_dir=None,
    first_dhcp_ind=0):
    # [total volume, total surface area, surface to volumn, [6] relative volume of each structure, ... [6] relative
    # number of feayures must be 21
    print("extract()")
    seg_list_all = np.array(sorted(glob.glob(seg_path + '/*')))

    if not load:

        assert inner_labels is not None

        all_features = np.zeros(21, dtype=float)
        for i in range(len(seg_list_all)):
            print("processing: " + str(seg_list_all[i].split('/')[-1].split('_')[0]))
            seg_vol, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list_all[i], return_volume=True, aff_ref=None, max_channels=10)

            assert isinstance(seg_vol[0, 0, 0], np.float64)
            assert isinstance(seg_im_res[0], np.float32)

            inner_labels = np.array(inner_labels).astype(np.float64)
            mask_whole = seg_vol == inner_labels[0]
            for lbl in inner_labels[1:]:
                mask_whole = np.logical_or(mask_whole, seg_vol == lbl)

            total_volume = float(np.sum(mask_whole))
            verts, faces, _, _ = measure.marching_cubes(mask_whole, method='lewiner') 

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

            assert (np.all(struct_vols<1) and np.all(struct_areas<1)) 
            assert len(np.array([total_volume])) == 1
            assert len(np.array([total_surface_area])) == 1
            assert len(np.array([surface_to_volumn])) == 1
            assert len(struct_vols) == 6
            assert len(struct_areas) == 6
            assert len(struct_surface_to_volumn) == 6

            vol_feature = np.concatenate((np.array([total_volume]), np.array([total_surface_area]),np.array([surface_to_volumn]), struct_vols, struct_areas, struct_surface_to_volumn))
            all_features = np.vstack([all_features, vol_feature])
            # print(vol_feature)

        raw_features = all_features[1:]
        print(f"saving raw features to {saved_features_path}...")
        np.save(saved_features_path,raw_features)
    else:
        print("loading raw features...")
        raw_features = np.load(saved_features_path)

    assert raw_features.shape == (len(glob.glob(seg_path + '/*')), 21), "feature matrix has incorrect shape"
    # [(s_x/s_total)/(s_y/s_total)] / [(a_x/a_total)/(a_y/a_total)] = (s_x/a_x)/(s_y/a_y)
    test_assert = raw_features[:,9:14] / raw_features[:,10:15] / raw_features[:,3:8] * raw_features[:,4:9] - raw_features[:,15:20] / raw_features[:,16:]
    assert np.sum(test_assert) < 1e-12, "feature matrix is incorrectly calculated"

    # generate mask for synth templates and dhcp templates: mask_synth, mask_dhcp, where path=synth, and neuro=dhcp
    mask_zurich = np.ones(len(seg_list_all)).astype(bool)
    mask_zurich[first_dhcp_ind:] = False 
    mask_dhcp = np.array([~m for m in mask_zurich])

    assert np.all(np.array([(("_rec-" in temppath) or ("_json_" in temppath)) for temppath in seg_list_all[mask_zurich]]))
    assert np.all(np.array(["_dseg_FeTA_labels.nii.gz" in temppath for temppath in seg_list_all[mask_dhcp]]))
    assert np.sum(mask_zurich) + np.sum(mask_dhcp) == len(raw_features)
    assert np.sum(mask_zurich) == 160
    assert np.sum(mask_dhcp) == 425 - 160


    print("processing dhcp")
    clusters_dhcp = process_feratures(    
        raw_features=raw_features[mask_dhcp, :],
        n_clusters=n_clusters_dhcp,
        clustering_method=clustering_method,
        n_components=n_components,
        save_plot=save_plot,
        gt_classes_2=gt_classes_2,
        gt_classes_3=gt_classes_3,
        accord_2=accord_2,
        accord_3=accord_3,
        accord_23=accord_23,
        accord_exp=accord_exp,
        fig_dir=fig_dir)
    
    print("processing synth")
    clusters_synth = process_feratures(    
        raw_features=raw_features[mask_zurich, :],
        n_clusters=n_clusters_zurich,
        clustering_method=clustering_method,
        n_components=n_components,
        save_plot=save_plot,
        gt_classes_2=gt_classes_2,
        gt_classes_3= gt_classes_3,
        accord_2=accord_2,
        accord_3=accord_3,
        accord_23=accord_23,
        accord_exp=accord_exp,
        fig_dir=fig_dir)

    # assign weights according to assigned cluster
    weight_per_cluster_dhcp = 0.5 / n_clusters_dhcp
    weight_per_cluster_synth = 0.5 / n_clusters_zurich
    counts_dhcp = np.bincount(clusters_dhcp)
    counts_synth = np.bincount(clusters_synth)
    print("number of templates in each cluster (dhcp): ")
    print(counts_dhcp)
    print("number of templates in each cluster (synth): ")
    print(counts_synth)

    weights = []
    # weight assigning example:
    # mask_path = [f,f,t,t,f,f,t,t,f,t]
    # mask_neuro= [t,t,f,f,t,t,f,f,t,f]
    # ind_path  = [0,0,0,1,2,2,2,3,4,4]
    # ind_neuro = [0,1,2,2,2,3,4,4,4,5]

    for ind in range(len(seg_list_all)):
        if not mask_zurich[ind]:
            ind_dhcp = np.sum(mask_dhcp[:ind])
            weights.append(weight_per_cluster_dhcp / counts_dhcp[clusters_dhcp[ind_dhcp]])
        else:
            ind_synth = np.sum(mask_zurich[:ind])
            weights.append(weight_per_cluster_synth / counts_synth[clusters_synth[ind_synth]])

    # validate weights: 
    print("testing correctness of weights...")
    for _ in range(1000):
        ass_ind = np.random.randint(0, len(seg_list_all))
        ass_weight = weights[ass_ind]
        if mask_dhcp[ass_ind]:
            ass_cluster = clusters_dhcp[np.sum(mask_dhcp[:ass_ind])]
            ass_count = np.sum(clusters_dhcp == ass_cluster)
            assert ass_weight == (0.5/n_clusters_dhcp/ass_count), ass_weight - (0.5/n_clusters_dhcp/ass_count)
        else:
            ass_cluster = clusters_synth[np.sum(mask_zurich[:ass_ind])]
            ass_count = np.sum(clusters_synth == ass_cluster)
            assert ass_weight == (0.5/n_clusters_zurich/ass_count), ass_weight - (0.5/n_clusters_zurich/ass_count)
    
    assert (np.sum(weights) >= 0.995) and (np.sum(weights) <= 1.00001), np.sum(weights)
    assert (np.sum(np.array(weights)[mask_zurich]) <= 0.50001) and (np.sum(np.array(weights)[mask_zurich]) >= 0.4975)
    print(weights)
    np.save(path_to_save_weights, weights)
    print(f"weights saved to: {path_to_save_weights}")

    return weights 

extract_pret(save_plot=False, seg_path='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/synth1v1+dhcp', inner_labels=[2,3,4,5,6,7], n_clusters_dhcp=8, n_clusters_zurich=6, clustering_method='gmm', gt_classes_2=None, gt_classes_3=None, n_components=3, accord_2=False, accord_3=False, accord_23=False, accord_exp=True, load=True, saved_features_path='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/weights_features/synth1v1+dhcp_sep_clex1/features_synth1v1+dhcp_sep_clex1.npy', fig_dir=None, path_to_save_weights='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/weights_features/synth1v1+dhcp_sep_clex1/weights_synth1v1+dhcp_sep_clex1.npy', first_dhcp_ind=160)


# extract_pret(save_plot=False, seg_path='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/zurich+dhcp_train', inner_labels=[2,3,4,5,6,7], n_clusters_dhcp=8, n_clusters_zurich=4, clustering_method='gmm', gt_classes_2=None, gt_classes_3=None, n_components=3, accord_2=False, accord_3=False, accord_23=False, accord_exp=True, load=True, saved_features_path='/Users/ziyaoshang/Desktop/fa2023/SP/temp/features_comp.npy', fig_dir=None, path_to_save_weights='/Users/ziyaoshang/Desktop/fa2023/SP/temp/weights_comp.npy', first_dhcp_ind=80)
# arr1 = np.load("/Users/ziyaoshang/Desktop/fa2023/SP/temp/weights_zurich+dhcp_train_clst.npy")
# arr2 = np.load("/Users/ziyaoshang/Desktop/fa2023/SP/temp/weights_comp.npy")
# print(np.sum(np.abs(arr1 - arr2)))

# print('done')
# "weights_zurich+dhcp_train_clst.npy"

