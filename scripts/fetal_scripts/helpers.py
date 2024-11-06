
import nibabel as nib
import glob
import os
import numpy as np
from ext.lab2im.edit_volumes import align_volume_to_ref, crop_volume_around_region, pad_volume, resample_volume, resample_volume_like, crop_volume, crop_volume_with_idx
from ext.lab2im.utils import get_volume_info, save_volume, get_list_labels, list_images_in_folder
from SynthSeg.evaluate import evaluation



def print_file_info():
    folder_path = '/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg'
    file_extension = '*_dseg.nii.gz'
    files = glob.glob(os.path.join(folder_path, '**', file_extension), recursive=True)
    sizes = np.zeros(3)
    i=0

    for file in files:
        # print(i)
        i+=1
        im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(file, return_volume=True, aff_ref=None,max_channels=10)
        print(aff[0,0], aff[1,1], aff[2,2])
        # assert [aff[0,0], aff[1,1], aff[2,2]] == [-1.125, -1.125, 1.125]

    # print(get_list_labels(label_list=None, labels_dir="/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/seg", save_label_list=None, FS_sort=False))


def center_labels():
    print("center_labels()")
    folder_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1_extracereb"
    file_extension = '*.nii.gz'
    save_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1_extracereb_centered"
    allow_huge_input_sizes = False
    files = glob.glob(os.path.join(folder_path, '**', file_extension), recursive=True)

    # return
    for file in files:
        # Load volumns
        im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(file, return_volume=True, aff_ref=None, max_channels=10)
        assert isinstance(im[0, 0, 0], np.float64)

        new_vol, cropping, new_aff = crop_volume_around_region(im,
                                                           mask=None,
                                                           masking_labels=None,
                                                           threshold=0.1,
                                                           margin=0,
                                                           cropping_shape=None,
                                                           cropping_shape_div_by=None,
                                                           aff=aff,
                                                           overflow='strict')
        # if any dimension of the original size of the labels is larger than 256, it would remain the original size, while the dimensions smaller than 256 would be padded to 256.
        final_vol, final_aff = pad_volume(new_vol, 256, padding_value=0, aff=new_aff, return_pad_idx=False)

        if not allow_huge_input_sizes:
            assert final_vol.shape == (256,256,256), final_vol.shape
        else: # Here, we center-crop the remaining large dimensions into 256
            if final_vol.shape != (256,256,256):
                print("large volumn: " + file)
                final_vol = crop_volume(final_vol, cropping_margin=None, cropping_shape=(256,256,256), aff=None, return_crop_idx=False, mode='center')
        
        assert final_vol.shape == (256,256,256), final_vol.shape
        save_file_name = os.path.join(save_path, file.split('/')[-1])

        # if (cropping[3] - cropping[0] > 192) | (cropping[4] - cropping[1] > 192) | (cropping[5] - cropping[2] > 192):
        #     print(np.array([cropping[3] - cropping[0], cropping[4] - cropping[1], cropping[4] - cropping[2]]))

        save_volume(final_vol, final_aff, h, save_file_name, dtype='float64')



def add_extra_cerebral_as_additional_label():
    print("add_extra_cerebral_as_additional_label()")
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/seg"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/img"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel"
    seg_list = sorted(glob.glob(seg_path + '/*'))
    print(len(seg_list))
    img_list = sorted(glob.glob(img_path + '/*'))
    print(len(img_list))
    assert len(img_list) == len(seg_list)
    for i in range(len(img_list)):
        # print(seg_list[i].split('/')[-1].split('_')[0].split('-')[1])

        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))
        # print(file_name)

        assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True,aff_ref=None, max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],return_volume=True,aff_ref=None, max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        # assert np.sum(img_im > 0) == np.sum(img_im != 0), np.min(img_im)
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("!!!reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.0001))
            img_im[img_im < 0] = 0.0
        assert np.all(seg_im >= 0)
        assert np.all(img_im >= 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        ext_ce = np.logical_and(seg_im == 0, img_im != 0)
        seg_im[ext_ce] = np.dtype(seg_im[0, 0, 0]).type(10)

        # print(np.sum(np.logical_and(seg_im != 0, img_im == 0)))
        # assert np.sum(img_im == 0) == np.sum(seg_im == 0), str(np.sum(img_im > 0)) + "  " + str(np.sum(seg_im > 0))
        # print(np.dtype(seg_im[0, 0, 0]).type(10))
        # print(np.average(img_im))
        file_name = os.path.join(save_path, seg_list[i].split('/')[-1])
        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)

def add_extra_cerebral_as_additional_label_synth():
    print("add_extra_cerebral_as_additional_label()")
    seg_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/ZURICH/original_data/mri_train"
    save_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1_extracereb"
    seg_list = sorted(glob.glob(seg_path + '/*'))
    # print(seg_list)
    img_list = sorted(glob.glob(img_path + '/*'))
    # print(img_list)
    assert len(seg_list) / len(img_list) == 2
    for i in range(len(img_list)):
        for j in range(2):
            # print(seg_list[i].split('/')[-1].split('_')[0].split('-')[1])

            print("processing: " + str(seg_list[i*2+j].split('/')[-1]))
            print(img_list[i].split('/')[-1].split('_')[0])

            assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i*2+j].split('/')[-1].split('_')[0]

            img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True,aff_ref=None, max_channels=10)
            seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i*2+j],return_volume=True,aff_ref=None, max_channels=10)
            assert (np.all(img_shp == seg_shp))
            assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
            assert (img_n_dims == img_n_dims)
            assert (img_n_channels == seg_n_channels)
            assert (np.all(np.abs(img_im_res-seg_im_res) < 0.001)), str(img_im_res)+str(seg_im_res)
            assert np.sum(img_im > 0) == np.sum(img_im != 0)
            assert np.all(seg_im >= 0)
            assert np.all(img_im >= 0)
            assert isinstance(seg_im[0, 0, 0], np.float64)
            assert isinstance(img_im[0, 0, 0], np.float64)
            assert isinstance(img_im_res[0], np.float32)
            assert isinstance(seg_im_res[0], np.float32)

            ext_ce = np.logical_and(seg_im == 0, img_im != 0)
            seg_im[ext_ce] = np.dtype(seg_im[0, 0, 0]).type(10)

            # print(np.sum(np.logical_and(seg_im != 0, img_im == 0)))
            # assert np.sum(img_im == 0) == np.sum(seg_im == 0), str(np.sum(img_im > 0)) + "  " + str(np.sum(seg_im > 0))
            # print(np.dtype(seg_im[0, 0, 0]).type(10))
            # print(np.average(img_im))
            file_name = os.path.join(save_path, seg_list[i*2+j].split('/')[-1])
            save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)

def remove_mri_ex_cereb_by_masking():
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/vienna/seg"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/vienna/img"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/vienna/no_extra_cereb"
    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))

        assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        assert np.sum(img_im > 0) == np.sum(img_im != 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)

        ext_ce = np.logical_and(seg_im == 0, img_im != 0)
        img_im[ext_ce] = np.dtype(img_im[0, 0, 0]).type(0)

        file_name = os.path.join(save_path, img_list[i].split('/')[-1])
        save_volume(volume=img_im, aff=img_aff, header=img_h, path=file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)


def sort_files():
    import shutil

    folder_path = '/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal'
    file_extension = '*_desc-restore_T2w.nii.gz'
    files = glob.glob(os.path.join(folder_path, '**', file_extension), recursive=True)
    for f in files:
        shutil.move(f, "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/img")

def load_results():
    val = True
    folder = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/temp_final_week_useful/arrs"
    if val:
        for i in ['003', '004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019']:
            # for i in ['008', '015', '018']:
            print(f"dice{i}.npy")
            dice = np.load(os.path.join(folder, f"dice_{i}.npy"))
            # mean_distance = np.load(os.path.join(folder, "mean_distance.npy"))
            # hausdorff = np.load(os.path.join(folder, "hausdorff.npy"))
            # print("hausdorff")
            # print(np.mean(hausdorff, axis=1))
            # print("mean_distance")
            # print(np.mean(mean_distance, axis=1))
            # print("dice")
            # print(np.mean(dice, axis=1))
            if np.mean(np.mean(dice, axis=1)[1:]) * 100 > 72:
                print(np.mean(np.mean(dice, axis=1)[1:]) * 100)

            print('\n')

def draw_boxplots_based_onresult_array():
    import matplotlib.pyplot as plt
    import numpy as np

    data_path = "/home/zshang/SP/data/ZURICH/experiments/results/prelim/dice.npy"
    save_path = "/home/zshang/SP/data/ZURICH/experiments/results/prelim/prelimdice_e8.png"
    table_name = "prelim: dice_e8_ZURICH"

    # data_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/seg/dice.npy"
    # save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/dice.png"
    # table_name = "unify_voxel_size_e12->masked cerebral: dice"
    x_label = "label"
    y_label = "dice"

    data = np.load(data_path)
    fig, ax = plt.subplots()
    print(np.mean(np.mean(data, axis=1)[1:]))
    for i in range(data.shape[0]):
        ax.boxplot(data[i, :], positions=[i], widths=0.6, showfliers=False)

    # Set the x-axis labels and title
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xticklabels([f'{i}' for i in range(data.shape[0])])
    ax.set_title(table_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.savefig(save_path)


def invert_mri_intensities():
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/orig/seg"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/synthseg_data/zurich/no_extra_cereb_not_centered_mri"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete"
    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0].split('-')[1]))

        assert img_list[i].split('/')[-1].split('_')[0].split('-')[1] == seg_list[i].split('/')[-1].split('_')[0].split('-')[1]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        assert np.sum(img_im > 0) == np.sum(img_im != 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)

        mean = np.mean(img_im[seg_im != 0])
        def normalize(x):
            return mean + mean - x

        img_im = normalize(img_im)
        img_im[seg_im == 0] = 0
        img_im[img_im < 0] = 0

        file_name = os.path.join(save_path, "inv_" + img_list[i].split('/')[-1])
        print(file_name)
        save_volume(volume=img_im, aff=img_aff, header=img_h, path=file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)


def resample_seg_according_to_base_vol(to_resample = '/home/zshang/SP/data/CHUV/experiments/results/prelim/raw_seg',
                                       to_resample_file_extension = '*_synthseg.nii.gz',
                                       base='/home/zshang/SP/data/CHUV/less_bg_with_extra_cereb/img',
                                       base_file_extension = '*_T2w.nii.gz',
                                       save_path = "/home/zshang/SP/data/CHUV/experiments/results/prelim/fin_seg"):

    to_resample_files = sorted(glob.glob(os.path.join(to_resample, '**', to_resample_file_extension), recursive=True))
    base_files = sorted(glob.glob(os.path.join(base, '**', base_file_extension), recursive=True))
    assert len(to_resample_files) == len(base_files)

    for i in range(len(to_resample_files)):
        im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(to_resample_files[i], return_volume=True, aff_ref=None, max_channels=10)
        b_im, b_shp, b_aff, b_n_dims, b_n_channels, b_h, b_im_res = get_volume_info(base_files[i], return_volume=True, aff_ref=None, max_channels=10)

        volume2 = resample_volume_like(vol_ref=b_im, aff_ref=b_aff, vol_flo=im, aff_flo=aff, interpolation='nearest')

        print(i)
        file_name = os.path.join(save_path, to_resample_files[i].split('/')[-1])
        save_volume(volume=volume2, aff=b_aff, header=b_h, path=file_name, res=b_im_res, dtype="float64", n_dims=b_n_dims)


def evaluate_own(gt_dir='/home/zshang/SP/data/CHUV/less_bg_with_extra_cereb/seg',
                 seg_dir='/home/zshang/SP/data/CHUV/experiments/results/prelim/fin_seg'):

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


def remove_most_backgrounds_and_center_vol(
                            seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1/orig/seg",
                            img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1/orig/img",
                            splitby = "_desc-",
                            save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1/test_lessbg/img",
                            save_path2 = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1/test_lessbg/seg"):

    pad = False
    margin = 5
    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split(splitby)[0]))

        assert img_list[i].split('/')[-1].split(splitby)[0] == seg_list[i].split('/')[-1].split(splitby)[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.001))
            img_im[img_im < 0] = 0.0

        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)

        img_im, cropping, img_aff = crop_volume_around_region(img_im,
                                                               mask=None,
                                                               masking_labels=None,
                                                               threshold=0.1,
                                                               margin=margin,
                                                               cropping_shape=None,
                                                               cropping_shape_div_by=None,
                                                               aff=img_aff,
                                                               overflow='strict')

        # crop seg the using the same margin as the mri
        seg_im, seg_aff = crop_volume_with_idx(seg_im, cropping, aff=seg_aff, n_dims=seg_n_dims, return_copy=True)
        assert np.all(seg_im.shape == img_im.shape)

        if pad:
            final_img_im, final_img_aff = pad_volume(img_im, 160, padding_value=0, aff=img_aff, return_pad_idx=False)
            final_seg_im, final_seg_aff = pad_volume(seg_im, 160, padding_value=0, aff=seg_aff, return_pad_idx=False)
            assert np.all(final_seg_im.shape == final_img_im.shape)
            assert final_seg_im.shape == (160, 160, 160), final_seg_im.shape
        else:
            final_img_im, final_img_aff = img_im, img_aff
            final_seg_im, final_seg_aff = seg_im, seg_aff


        img_file_name = os.path.join(save_path, "lessbg" + img_list[i].split('/')[-1])
        seg_file_name = os.path.join(save_path2, "lessbg" + seg_list[i].split('/')[-1])

        # final_img_im, final_img_aff = img_im, img_aff
        # final_seg_im, final_seg_aff = seg_im, seg_aff

        save_volume(volume=final_img_im, aff=final_img_aff, header=img_h, path=img_file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)
        save_volume(volume=final_seg_im, aff=final_seg_aff, header=seg_h, path=seg_file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)


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


def replace_gt_with_smoothed_label():
    from scipy.spatial.distance import cdist

    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/smoothed_7"
    label_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/selected_corrected_label_4"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/smoothed_74"
    label_to_smooth = 4
    temp_label = 20

    seg_list = sorted(glob.glob(seg_path + "/*"))
    label_list = sorted(glob.glob(label_path + "/*"))
    assert len(label_list) == len(seg_list)

    for i in range(len(label_list)):
        print("processing: " + str(label_list[i].split('/')[-1].split('_')[2]))

        assert label_list[i].split('/')[-1].split('_')[2] == seg_list[i].split('/')[-1].split('_')[2]

        label_im, label_shp, label_aff, label_n_dims, label_n_channels, label_h, label_im_res = get_volume_info(label_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        assert (np.all(label_shp == seg_shp))
        assert (np.all(np.abs(label_aff - seg_aff) < 0.0001)), print(str(label_aff) + str(seg_aff))
        assert (label_n_dims == label_n_dims)
        assert (label_n_channels == seg_n_channels)
        assert (np.all(label_im_res == seg_im_res))
        assert np.sum(label_im > 0) == np.sum(label_im != 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(label_im[0, 0, 0], np.float64)

        # overlay the smoothed label onto the original seg with a temporary label
        seg_im[label_im == 1] = temp_label

        # candidate locations to replace each missing label with (using the underlying label)
        replace_candidate_inds = np.where((seg_im != temp_label) & (seg_im != label_to_smooth))
        replace_candidate_inds = np.column_stack((replace_candidate_inds[0], replace_candidate_inds[1], replace_candidate_inds[2]))

        # all remaining holes to replace
        remaining_inds = np.where(seg_im == label_to_smooth)
        # print(np.sum(seg_im == label_to_smooth))
        for ind in zip(*remaining_inds):
            ind = np.array(ind)
            # print(ind)
            ind_for_replace = np.argmin(cdist([ind], replace_candidate_inds))

            label_to_replace = seg_im[replace_candidate_inds[ind_for_replace][0],
                                        replace_candidate_inds[ind_for_replace][1],
                                        replace_candidate_inds[ind_for_replace][2]]

            seg_im[ind[0], ind[1], ind[2]] = label_to_replace
            # print(label_to_replace)
        # return
        # convert temp label back to the label to smooth
        seg_im[seg_im == temp_label] = label_to_smooth

        save_name = os.path.join(save_path, "smoothed_" + seg_list[i].split('/')[-1])
        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=save_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)


def adjust_volumn_to_trainable_by_near():

    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/vienna/seg"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/vienna/no_extra_cereb_mri"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/vienna/near_img"
    save_path2 = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/vienna/near_seg"

    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))

        assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)

        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), str(img_aff) + str(seg_aff)
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        assert np.sum(img_im > 0) == np.sum(img_im != 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        img_im, img_aff = resample_volume(volume=img_im, aff=img_aff, new_vox_size=1.0, interpolation='linear', blur=True)
        seg_im, seg_aff = resample_volume(volume=seg_im, aff=seg_aff, new_vox_size=1.0, interpolation='nearest', blur=False)


        img_im, cropping, img_aff = crop_volume_around_region(img_im,
                                                       mask=None,
                                                       masking_labels=None,
                                                       threshold=0.00001,
                                                       margin=0,
                                                       cropping_shape=None,
                                                       cropping_shape_div_by=None,
                                                       aff=img_aff,
                                                       overflow='strict')

        seg_im, seg_aff = crop_volume_with_idx(volume=seg_im, crop_idx=cropping, aff=seg_aff, n_dims=None, return_copy=True)

        final_img_im, final_img_aff = pad_volume(img_im, 160, padding_value=0, aff=img_aff, return_pad_idx=False)
        final_seg_im, final_seg_aff = pad_volume(seg_im, 160, padding_value=0, aff=seg_aff, return_pad_idx=False)

        assert final_img_im.shape == final_seg_im.shape
        # assert img_im.shape == seg_im.shape, str(img_im.shape) + str(seg_im.shape)
        assert final_img_im.shape == (160, 160, 160)

        img_file_name = os.path.join(save_path, "near_" + img_list[i].split('/')[-1])
        seg_file_name = os.path.join(save_path2, "near_" + seg_list[i].split('/')[-1])

        save_volume(volume=final_img_im, aff=final_img_aff, header=img_h, path=img_file_name, res=np.array([1.0,1.0,1.0], dtype='float32'), dtype="float64", n_dims=img_n_dims)
        save_volume(volume=final_seg_im, aff=final_seg_aff, header=seg_h, path=seg_file_name, res=np.array([1.0,1.0,1.0], dtype='float32'), dtype="float64", n_dims=seg_n_dims)


def t_test_on_label():
    from scipy.stats import ttest_rel
    label_ind = 4
    folder1 = "/Users/ziyaoshang/Desktop/fa2023/SP/results/chuv/smooth_47"
    folder2 = "/Users/ziyaoshang/Desktop/fa2023/SP/results/chuv/unify_voxel_size_all_extra_label_less_bg_rigorous_e7"
    dice1 = np.load(os.path.join(folder1, f"dice.npy"))[label_ind]
    dice2 = np.load(os.path.join(folder2, f"dice.npy"))[label_ind]

    t_statistic, p_value = ttest_rel(dice1, dice2)
    print(p_value)


def divide_bg_using_kmeans():
    print("divide_bg_using_kmeans()")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import warnings

    # warnings.filterwarnings("ignore")

    all_bg_folder = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel"
    img_folder = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/img"
    save_path = '/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel_bgsubd'
    bg_labels = [10, 11, 12, 13]
    n_clusters = len(bg_labels)

    seg_list = sorted(glob.glob(all_bg_folder + '/*'))
    img_list = sorted(glob.glob(img_folder + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1]))

        assert img_list[i].split('/')[-1].split('_')[1].split('-')[1] == \
               seg_list[i].split('/')[-1].split('_')[1].split('-')[1]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True, aff_ref=None, max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],return_volume=True,aff_ref=None, max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.001)), img_aff - seg_aff
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        # assert np.sum(img_im > 0) == np.sum(img_im != 0)
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("!!!reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.0001))
            img_im[img_im < 0] = 0.0
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        # inds = np.where(np.logical_or(seg_im == 0, seg_im == 10))
        inds = np.where(seg_im == 10)
        vects = np.column_stack([inds[0], inds[1], inds[2], img_im[inds]])
        # print(inds[0].shape)
        normalized_data = MinMaxScaler().fit_transform(vects.copy()[:,-1].reshape(-1, 1))
        # normalized_data = MinMaxScaler().fit_transform(vects.copy())
        # normalized_data = vects.copy()
        # normalized_data[:, -1] *= 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
        kmeans.fit(normalized_data)
        labels = kmeans.labels_

        for l in range(labels.shape[0]):
            # print(seg_im[tuple(vects[l, :3].astype('int'))])
            # print(seg_im[(27, 44, 37)])
            seg_im[tuple(vects[l, :3].astype('int'))] = bg_labels[labels[l]]

        # centers = kmeans.cluster_centers_
        file_name = os.path.join(save_path, seg_list[i].split('/')[-1])

        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name,
                    res=img_im_res, dtype="float64", n_dims=seg_n_dims)


def align_vol_to_ras_coords():
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/gt"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/img"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/img_f"
    save_path2 = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/training_data/vienna/near_training_vienna/gt_f"

    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))

        assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)


        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.001)), str(img_aff) + str(seg_aff)
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        assert np.sum(img_im > 0) == np.sum(img_im != 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        img_im, img_aff = align_volume_to_ref(img_im, img_aff, aff_ref=np.eye(4), return_aff=True, n_dims=img_n_dims)
        seg_im, seg_aff = align_volume_to_ref(seg_im, seg_aff, aff_ref=np.eye(4), return_aff=True, n_dims=seg_n_dims)

        img_file_name = os.path.join(save_path, img_list[i].split('/')[-1])
        seg_file_name = os.path.join(save_path2, seg_list[i].split('/')[-1])

        save_volume(volume=img_im, aff=img_aff, header=img_h, path=img_file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)
        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=seg_file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)


def discrete_label_smoothing():
    from scipy.signal import convolve
    from scipy.ndimage import gaussian_filter

    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/ZURICH_test/res/final_seg"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/delete/ZURICH_test/res/smoothed_final_seg"
    n_labels = 8

    seg_list = sorted(glob.glob(seg_path + '/*'))

    for i in range(len(seg_list)):
        print("processing: " + str(seg_list[i].split('/')[-1].split('_')[0]))

        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)

        all_labels = []
        for lb in range(n_labels):
            prob_array = np.vectorize(lambda x: float(x == lb))(seg_im)
            prob_array = prob_array.reshape(*seg_im.shape)
            all_labels.append(prob_array)

        all_labels = np.stack(all_labels, axis=-1)

        for lb in range(n_labels):
            all_labels[..., lb] = gaussian_filter(all_labels[..., lb], sigma=1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=1)

        all_labels = np.argmax(all_labels, axis=-1)
        print(all_labels.shape)

        seg_file_name = os.path.join(save_path, "smothed_" + seg_list[i].split('/')[-1])
        save_volume(volume=all_labels, aff=seg_aff, header=seg_h, path=seg_file_name, res=seg_im_res,
                    dtype="float64", n_dims=seg_n_dims)

def force_50perc_with_noclst():
    seg_list = sorted(glob.glob('/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/synth1v1+dhcp/*'))
    synth = seg_list[:160]
    dhcp = seg_list[160:]

    assert np.all([(("_rec-" in f) or ("_json_" in f)) for f in synth])
    assert np.all(["_dseg_FeTA_labels.nii.gz" in f for f in dhcp])

    weights = np.array([0.5/len(synth)] * len(synth) + [0.5/len(dhcp)] * len(dhcp))
    print(weights)
    print(len(seg_list))
    assert len(weights) == len(seg_list)
    assert np.sum(weights) == 1.0

    np.save("/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/weights_features/synth1v1+dhcp_sep_noclst/weights_synth1v1+dhcp_sep_noclst.npy", weights)

def sheep_temp():
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/test_data/raw/sheep"
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/grand_experiment/test_data/processed/sheep"

    pad = False
    margin = 5
    img_list = sorted(glob.glob(img_path + '/*'))

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],
                                                                                                  return_volume=True,
                                                                                                  aff_ref=None,
                                                                                                  max_channels=10)
        assert (img_n_dims == img_n_dims)

        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.001))
            img_im[img_im < 0] = 0.0

        assert isinstance(img_im[0, 0, 0], np.float64)

        img_im, cropping, img_aff = crop_volume_around_region(img_im,
                                                               mask=None,
                                                               masking_labels=None,
                                                               threshold=0.1,
                                                               margin=margin,
                                                               cropping_shape=None,
                                                               cropping_shape_div_by=None,
                                                               aff=img_aff,
                                                               overflow='strict')

        if pad:
            final_img_im, final_img_aff = pad_volume(img_im, 160, padding_value=0, aff=img_aff, return_pad_idx=False)
        else:
            final_img_im, final_img_aff = img_im, img_aff

        img_file_name = os.path.join(save_path, "lessbg" + img_list[i].split('/')[-1])
        save_volume(volume=final_img_im, aff=final_img_aff, header=img_h, path=img_file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)


def temp():
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1_orig/seg"
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/transfer_T1_orig/img"

    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    # assert 
    print(len(img_list))
    print(len(seg_list))
    seg_sub = []
    img_sub = []
    for i in range(len(seg_list)):
        seg_sub.append(seg_list[i].split('/')[-1].split('_desc')[0])
    for j in range(len(img_list)):
        img_sub.append(img_list[j].split('/')[-1].split('_desc')[0])

    delsimg = [s for s in img_sub if s not in seg_sub]
    delsseg = [s for s in seg_sub if s not in img_sub]
    print(delsimg)
    print(len(delsimg))
    print(delsseg)
    print(len(delsseg))
    assert len(img_list) - len(delsimg) == len(seg_list) - len(delsseg)

    for i in seg_list:
        if np.any(np.array([(sub in i) for sub in delsseg])):
            print(i.split('/')[-1])
            os.remove(i)
    for j in img_list:
        if np.any(np.array([(sub in j) for sub in delsimg])):
            print(j.split('/')[-1])
            os.remove(j)


        # print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))

        # if not img_list[i].split('/')[-1].split('_')[0] == seg_list[i+diff].split('/')[-1].split('_')[0]:
        #     print(img_list[i].split('/')[-1].split('_')[0])
        #     diff+=1

        # img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True,aff_ref=None,max_channels=10)
        # seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i], return_volume=True, aff_ref=None,max_channels=10)
        # assert np.all(img_im == seg_im)
        # print("s")


# sort_files()
# print_file_info()
# add_extra_cerebral_as_additional_label()
# add_extra_cerebral_as_additional_label_synth()
# divide_bg_using_kmeans()
# center_labels()
# adjust_volumn_to_trainable_by_near()
# align_vol_to_ras_coords()
# remove_mri_ex_cereb_by_masking()
# sort_files()
# load_results()
# draw_boxplots_based_onresult_array()
# invert_mri_intensities()
# resample_seg_according_to_base_vol()
# evaluate_own()
# remove_most_backgrounds_and_center_vol()
# replace_gt_with_smoothed_label()
# t_test_on_label()
# discrete_label_smoothing()
# force_50perc_with_noclst()

# temp()

# seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/processed_others/zurich_test/img"
# seg_list = sorted(glob.glob(seg_path + '/*'))

# with open("/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/filenames.txt", 'a') as f:
#     for i in range(len(seg_list)):
#         f.write(seg_list[i].split("/")[-1] + '\n')

# input_file = '/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/filenames.txt'  
# output_file = '/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/testing.txt'  

# with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
#     for line in infile:
#         if line[:6] != "lessbg":
#             print(line)
#             modified_line = line
#         else:
#             modified_line = line[6:]
#         outfile.write(modified_line)


print("done")
