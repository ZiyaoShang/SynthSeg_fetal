import glob
import os
import numpy as np
from ext.lab2im.edit_volumes import resample_volume_like
from ext.lab2im.utils import get_volume_info, save_volume
from SynthSeg.evaluate import evaluation


def evaluate_own(gt_dir='/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg',
                 seg_dir="/Users/ziyaoshang/Desktop/fa2023/SP/results/zurich/unify_voxel_size_e10/final_seg"):

    evaluation(gt_dir=gt_dir,
               seg_dir=seg_dir,
               label_list=[0, 1, 2, 3, 4, 5, 6, 7],
               mask_dir=None,
               compute_score_whole_structure=False,
               path_dice=os.path.join(seg_dir, 'dice.npy'),
               path_hausdorff=os.path.join(seg_dir, 'hausdorff.npy'),
               path_hausdorff_99=os.path.join(seg_dir, 'hausdorff_99.npy'),
               path_hausdorff_95=os.path.join(seg_dir, 'hausdorff_95.npy'),
               path_mean_distance=os.path.join(seg_dir, 'mean_distance.npy'),
               crop_margin_around_gt=10,
               list_incorrect_labels=None,
               list_correct_labels=None,
               use_nearest_label=False,
               recompute=True,
               verbose=True)

def resample_seg_according_to_base_vol(to_resample = '/Users/ziyaoshang/Desktop/fa2023/SP/results/zurich/unify_voxel_size_e10/raw_seg',
                                       to_resample_file_extension = '*_synthseg.nii.gz',
                                       base='/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg',
                                       base_file_extension = '*_dseg.nii.gz',
                                       save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/results/zurich/unify_voxel_size_e10/final_seg"):

    to_resample_files = sorted(glob.glob(os.path.join(to_resample, '**', to_resample_file_extension), recursive=True))
    base_files = sorted(glob.glob(os.path.join(base, '**', base_file_extension), recursive=True))
    assert len(to_resample_files) == len(base_files), str((len(to_resample_files), len(base_files)))

    for i in range(len(to_resample_files)):
        im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(to_resample_files[i], return_volume=True, aff_ref=None, max_channels=10)
        b_im, b_shp, b_aff, b_n_dims, b_n_channels, b_h, b_im_res = get_volume_info(base_files[i], return_volume=True, aff_ref=None, max_channels=10)

        volume2 = resample_volume_like(vol_ref=b_im, aff_ref=b_aff, vol_flo=im, aff_flo=aff, interpolation='nearest')

        print(i)
        file_name = os.path.join(save_path, to_resample_files[i].split('/')[-1])
        save_volume(volume=volume2, aff=b_aff, header=b_h, path=file_name, res=b_im_res, dtype="float64", n_dims=b_n_dims)


def integrated_inference_resample_evaluate(
                                        testing_mri='/home/zshang/SP/data/CHUV/less_bg/img',
                                        model_weights="/home/zshang/SP/data/ZURICH/experiments/model/unify_voxel_size/dice_010.h5",
                                        raw_segs='/Users/ziyaoshang/Desktop/fa2023/SP/results/zurich/unify_voxel_size_e10/raw_seg',
                                        raw_segs_file_ext='*_synthseg.nii.gz',
                                        vols_with_correct_aff='/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg',
                                        vols_with_correct_aff_file_extension='*_dseg.nii.gz',
                                        gt_dir_for_eval='/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg',
                                        final_save_dir="/Users/ziyaoshang/Desktop/fa2023/SP/results/zurich/unify_voxel_size_e10/final_seg"):

    """
    last three steps + evaluation (best for synthseg) (for CHUV, do the first two steps, then directly evaluate with the original fetal_scripts
    predict with all evaluations enabled)
    :param raw_segs:
    :param
    """


    print("Predicting......")
    from SynthSeg.predict import predict

    path_images = testing_mri
    # path to the output segmentation
    path_segm = raw_segs
    # we can also provide paths for optional files containing the probability map for all predicted labels
    path_posteriors = None
    # and for a csv file that will contain the volumes of each segmented structure
    path_vol = None

    # of course we need to provide the path to the trained model (here we use the main synthseg model).
    path_model = model_weights
    # but we also need to provide the path to the segmentation labels used during training
    path_segmentation_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # optionally we can give a numpy array with the names corresponding to the structures in path_segmentation_labels
    path_segmentation_names = None
    cropping = None
    target_res = None
    path_resampled = None
    flip = False
    n_neutral_labels = 8
    sigma_smoothing = 0.5
    topology_classes = None
    keep_biggest_component = True

    # Regarding the architecture of the network, we must provide the predict function with the same parameters as during
    # training.
    n_levels = 5
    nb_conv_per_level = 2
    conv_size = 3
    unet_feat_count = 24
    activation = 'elu'
    feat_multiplier = 2

    gt_folder = None
    compute_distances = None

    # All right, we're ready to make predictions !!
    predict(path_images,
            path_segm,
            path_model,
            path_segmentation_labels,
            n_neutral_labels=n_neutral_labels,
            path_posteriors=path_posteriors,
            path_resampled=path_resampled,
            path_volumes=path_vol,
            names_segmentation=path_segmentation_names,
            cropping=cropping,
            target_res=target_res,
            flip=flip,
            topology_classes=topology_classes,
            sigma_smoothing=sigma_smoothing,
            keep_biggest_component=keep_biggest_component,
            n_levels=n_levels,
            nb_conv_per_level=nb_conv_per_level,
            conv_size=conv_size,
            unet_feat_count=unet_feat_count,
            feat_multiplier=feat_multiplier,
            activation=activation,
            gt_folder=gt_folder,
            compute_distances=compute_distances)

    # post-process
    print("Resampling segmentation back to original affine space (same as the input MRIs)......")
    resample_seg_according_to_base_vol(
        to_resample=raw_segs,
        to_resample_file_extension=raw_segs_file_ext,
        base=vols_with_correct_aff,
        base_file_extension=vols_with_correct_aff_file_extension,
        save_path=final_save_dir)

    # evaluate
    print("evaluating......")
    evaluate_own(gt_dir=gt_dir_for_eval,
                 seg_dir=final_save_dir)
    

def resample_eval(ind):
    print("Resampling segmentation back to original affine space (same as the input MRIs)......")

    resample_seg_according_to_base_vol(
        to_resample='/home/zshang/SP/data/grand_train_all/SP_exp/processed_test/sheep/raw_seg'+ind,
        to_resample_file_extension='*_synthseg.nii.gz',
        base="/home/zshang/SP/data/grand_train_all/SP_exp/processed_test/sheep/img",
        base_file_extension="*.nii.gz",
        save_path='/home/zshang/SP/data/grand_train_all/SP_exp/processed_test/sheep/fin_seg'+ind)

    # evaluate
    print("evaluating......")
    evaluate_own(
                gt_dir='/home/zshang/SP/data/grand_train_all/SP_exp/processed_test/DHCP_PRETERM/seg_T2',
                seg_dir='/home/zshang/SP/data/grand_train_all/SP_exp/processed_test/DHCP_PRETERM/res/fin_T2'+ind)


print("start")
# for i in ['03', '04', '05', '06', '07','08', '09', '10', '11','12','13','14','15','16', '17', '18', '19']:
#     print("epoch: "+ i)
#     resample_eval(ind=i)

resample_eval(ind='')

print("end")
