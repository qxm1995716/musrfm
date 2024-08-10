import numpy as np
from PIL import Image
import os


def patch_imgs_creator(patches, min_v=0.0, max_v=0.3, c=14):
    num_patches = len(patches)
    print('There are %d patches after MCHR processing.')
    patch_vis_list = []
    for idx in range(num_patches):
        element = patches[idx]
        height = element.shape[0] * (element.shape[-1] // c)
        width = element.shape[1]
        mrs_img = np.zeros([height, width, 3], dtype=np.uint8)

        for iidx in range(element.shape[-1] // 12):
            p = element[:, :, iidx * c + 1: iidx * c + 4]
            p[p < min_v] = min_v
            p[p > max_v] = max_v
            # re-scale the array
            p = (p - min_v) / (max_v - min_v) * 255
            p = p.astype(np.uint8)
            mrs_img[iidx * element.shape[0]: (iidx + 1) * element.shape[0], :, :] = p

        # get the pictures
        B = mrs_img[:, :, 0]
        G = mrs_img[:, :, 1]
        R = mrs_img[:, :, 2]
        mrs_img = np.concatenate([R[:, :, None], G[:, :, None], B[:, :, None]], axis=-1)
        patch_vis_list.append(mrs_img)

    return patch_vis_list


def imgs_write(mrs_patches_list, dict_):
    n = len(mrs_patches_list)
    if not os.path.exists(dict_):
        os.makedirs(dict_, exist_ok=True)
        
    for idx in range(n):
        fname = dict_ + '/Patch_' + str(idx) + '.jpg'
        p = mrs_patches_list[idx]
        img = Image.fromarray(p)
        img.save(fname, 'JPEG')

    return print('The data have output to the target dict.')

