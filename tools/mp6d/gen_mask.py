import scipy.io
import numpy as np
import glob
import cv2
import os

s = scipy.io.loadmat(
    "/media/sda1/r10922190/BOP_DATASETS/mp6d/data/0000/000000-meta.mat")

image = cv2.imread(
    '/media/sda1/r10922190/BOP_DATASETS/mp6d/data/0000/000000-label.png')

paths = glob.glob("/media/sda1/r10922190/BOP_DATASETS/mp6d/data_syn*/*-label.png")

# # * for data
# for (i, path) in enumerate(paths):
#     scene_id = path.split('/')[-2]
#     img_id = path.split('/')[-1][:6]
#     s = scipy.io.loadmat(
#         f"/media/sda1/r10922190/BOP_DATASETS/mp6d/data/{scene_id}/{img_id}-meta.mat")
#     if not os.path.exists(f"/media/sda1/r10922190/BOP_DATASETS/mp6d/data/{scene_id}/mask_visib"):
#         os.makedirs(
#             f"/media/sda1/r10922190/BOP_DATASETS/mp6d/data/{scene_id}/mask_visib")
#     image = cv2.imread(path)
#     for inst_id in s['cls_indexes']:
#         inst_img = np.zeros_like(image)
#         inst_img[np.where(image == inst_id)] = 255
#         cv2.imwrite(
#             f"/media/sda1/r10922190/BOP_DATASETS/mp6d/data/{scene_id}/mask_visib/{img_id}_{inst_id[0]:06d}_mask.png", inst_img)

#     print(f'Current: {i} Scene: {scene_id}  Total: {len(paths)}')


# * for data_syn
for (i, path) in enumerate(paths):
    scene_id = path.split('/')[-2]
    img_id = path.split('/')[-1][:6]
    s = scipy.io.loadmat(
        f"/media/sda1/r10922190/BOP_DATASETS/mp6d/{scene_id}/{img_id}-meta.mat")
    if not os.path.exists(f"//media/sda1/r10922190/BOP_DATASETS/mp6d/{scene_id}/mask_visib"):
        os.makedirs(
            f"//media/sda1/r10922190/BOP_DATASETS/mp6d/{scene_id}/mask_visib")
    image = cv2.imread(path)
    for inst_id in s['cls_indexes']:
        inst_img = np.zeros_like(image)
        inst_img[np.where(image == inst_id)] = 255
        cv2.imwrite(
            f"/media/sda1/r10922190/BOP_DATASETS/mp6d/{scene_id}/mask_visib/{img_id}_{inst_id[0]:06d}_mask.png", inst_img)

    print(f'Current: {i}  Total: {len(paths)}')

print(s['intrinsic_matrix'])
