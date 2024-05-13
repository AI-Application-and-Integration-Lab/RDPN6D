# -*- coding: utf-8 -*-
import copy
import hashlib
import logging
import os
import os.path as osp
import time
import open3d as o3d
import pickle
import random
from PIL import Image
import cv2
import mmcv
import numpy as np
import ref
import torch
import torch.multiprocessing as mp
from core.base_data_loader import Base_DatasetFromList
from core.utils.augment import AugmentRGB
from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, my_warp_affine, read_image_cv2, xyz_to_region
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    load_detections_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.utils.rot_reps import mat_to_ortho6d_np
from core.utils.ssd_color_transform import ColorAugSSDTransform
from core.utils.utils import egocentric_to_allocentric
from core.utils import quaternion_lf, lie_algebra
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.common.file_io import PathManager
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge
from lib.utils.utils import dprint, lazy_property
from transforms3d.quaternions import mat2quat

from .dataset_factory import register_datasets

logger = logging.getLogger(__name__)


global camera_xyz_dis
camera_xyz_dis = []


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        mask = transforms.apply_segmentation(cocosegm2mask(
            annotation["segmentation"], h=im_H, w=im_W))
        annotation["segmentation"] = mask

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(
            np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def build_gdrn_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size))

    augmentation = []
    if 'tless' not in cfg.DATASETS.TRAIN[0]:
        augmentation.append(T.ResizeShortestEdge(
            min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


class GDRN_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_gdrn_augmentation(
            cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(
                ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info(
                "Color augmnetation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # fmt: on
        self.cfg = cfg
        self.cov = []
        self.mean = []
        self.min = []
        self.max = []
        self.ins_count = 0
        for i in range(22):
            self.cov.append([])
            self.mean.append([])
            self.min.append([])
            self.max.append([])
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(
                aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x)
                                    for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(
                len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(
                    obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(
                    obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(
                model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.MODEL.CDPN.PNP_NET.NUM_PM_POINTS)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(
                model_path, vertex_scale=data_ref.vertex_scale)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array(
                [size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(
                    model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"]
                                    for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)

        if dataset_dict["dataset_name"] == 'delta':
            dataset_name = 'delta_train'
        else:
            dataset_name = dataset_dict["dataset_name"]

        image = read_image_cv2(
            dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]
        #img_type = dataset_dict.get("img_type", "real")
        # currently only replace bg for train ###############################
        # if self.split == "train":
        #     # some synthetic data already has bg, img_type should be real or something else but not syn
        #     img_type = dataset_dict.get("img_type", "real")
        #     if img_type == "syn":
        #         log_first_n(logging.WARNING, "replace bg", n=10)
        #         assert "segmentation" in dataset_dict["inst_infos"]
        #         mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
        #         image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)
        #     else:  # real image
        #         if np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
        #             log_first_n(logging.WARNING, "replace bg for real", n=10)
        #             assert "segmentation" in dataset_dict["inst_infos"]
        #             mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
        #             image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)
        #         else:
        #             mask_trunc = None

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:

            image = self._color_aug(image, 'code')

        mask_trunc = None
        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is now allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # CHW -> HWC
        coord_2d = get_2d_coord_np(
            im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        #################################################################################
        if self.split != "train":

            # don't load annotations at test time
            test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
            if test_bbox_type == "gt":
                bbox_key = "bbox"
            else:
                bbox_key = f"bbox_{test_bbox_type}"
            assert not self.flatten, "Do not use flattened dicts for test!"
            # here get batched rois
            roi_infos = {}
            # yapf: disable
            roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                        "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                        bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                        "scale", "resize_ratio", "model_info", "roi_xyz", "fps"
                        ]
            for _key in roi_keys:
                roi_infos[_key] = []

            # yapf: enable
            # TODO: how to handle image without detections
            #   filter those when load annotations or detections, impslement a function for this
            # "annotations" means detections

             # add mask ----------------------------------------
            # mask_path = dataset_dict['depth_file'].split('/')
            # mask_path[-2] = 'mask_visib'
            # mask_path[-1] = mask_path[-1][:6] + '_000000.png'
            # mask_path = '/'.join(mask_path)
            # mask_img = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
            # mask_img = mask_img.reshape(mask_img.shape[0], mask_img.shape[1], 1)

            # load depth image
            depth_img = np.array(Image.open(
                dataset_dict['depth_file'])).astype(np.float32)
            if 'depth_factor' in dataset_dict.keys():
                depth_img = depth_img / dataset_dict['depth_factor']
            else:
                depth_img = depth_img / 1000.0
            depth_img = depth_img[:, :, np.newaxis]

            #depth_img = depth_img * mask_img
            for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):

                # inherent image-level infos
                roi_infos["scene_im_id"].append(dataset_dict["scene_im_id"])
                roi_infos["file_name"].append(dataset_dict["file_name"])
                roi_infos["im_H"].append(im_H)
                roi_infos["im_W"].append(im_W)
                roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

                # roi-level infos
                roi_infos["inst_id"].append(inst_i)
                roi_infos["model_info"].append(inst_infos["model_info"])

                roi_cls = inst_infos["category_id"]
                fps_points = self._get_fps_points(
                    dataset_name)[roi_cls]  # * new
                roi_infos["fps"].append(fps_points)
                roi_infos["roi_cls"].append(roi_cls)
                roi_infos["score"].append(1.0)

                # extent
                roi_extent = self._get_extents(dataset_name)[roi_cls]
                roi_infos["roi_extent"].append(roi_extent)

                bbox = BoxMode.convert(
                    inst_infos[bbox_key], inst_infos["bbox_mode"], BoxMode.XYXY_ABS)
                bbox = np.array(transforms.apply_box([bbox])[0])
                roi_infos[bbox_key].append(bbox)
                roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
                x1, y1, x2, y2 = bbox
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
                scale = min(scale, max(im_H, im_W)) * 1.0

                roi_infos["bbox_center"].append(bbox_center.astype("float32"))
                roi_infos["scale"].append(scale)
                roi_infos["roi_wh"].append(
                    np.array([bw, bh], dtype=np.float32))
                roi_infos["resize_ratio"].append(out_res / scale)

                # load xyz =======================================================
                if cfg.TRAIN.VIS:
                    mask_xyz_interp = cv2.INTER_LINEAR
                else:
                    mask_xyz_interp = cv2.INTER_NEAREST
                # try:
                #     xyz_info = mmcv.load(inst_infos["xyz_path"])
                #     #print(inst_infos["xyz_path"])
                #     x1, y1, x2, y2 = xyz_info["xyxy"]
                #     # float16 does not affect performance (classification/regresion)
                #     xyz_crop = xyz_info["xyz_crop"]
                #     xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
                #     xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
                #     roi_xyz = crop_resize_by_warp_affine(xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)
                #     roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
                #     # normalize xyz to [0, 1] using extent
                #     roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
                #     roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
                #     roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5
                #     roi_infos["roi_xyz"].append(roi_xyz)
                # except:

                #     xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
                #     roi_xyz = crop_resize_by_warp_affine(xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)
                #     roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
                #     # normalize xyz to [0, 1] using extent
                #     roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
                #     roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
                #     roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5
                #     roi_infos["roi_xyz"].append(roi_xyz)

                # CHW, float32 tensor
                # roi_image
                roi_img = crop_resize_by_warp_affine(
                    image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)

                roi_img = self.normalize_image(cfg, roi_img)
                # depth xyz-----------------------------------------

                resize_ratio = out_res / scale

                depth_img2 = crop_resize_by_warp_affine(
                    depth_img, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                )
                depth_img2 = depth_img2[:, :, np.newaxis]

                rows, cols = 256, 256
                ymap = np.array([[j for i in range(cols)]
                                for j in range(rows)]).astype(np.float32)
                xmap = np.array([[i for i in range(cols)]
                                for j in range(rows)]).astype(np.float32)
                compute_cov = False
                # if compute_cov:
                #     rows, cols = 480,640
                #     ymap = np.array([[j for i in range(cols)] for j in range(rows)]).astype(np.float32)
                #     xmap = np.array([[i for i in range(cols)] for j in range(rows)]).astype(np.float32)
                #     zmap = np.array([xmap, ymap]).transpose(1,2,0)
                #     outputmap = crop_resize_by_warp_affine(
                #     zmap, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                #     ).transpose(2, 0, 1)
                #     xmap = outputmap[0]
                #     ymap = outputmap[1]
                H = my_warp_affine(coord_2d, bbox_center, scale,
                                   input_res, interpolation=cv2.INTER_LINEAR)
                offset_matrix = np.zeros((3, 3))
                offset_matrix[:2, :] = H
                offset_matrix[2][2] = 1

                if compute_cov:
                    depth_img2 = depth_img2
                    newCameraK = K  # np.matmul(offset_matrix,K)
                else:
                    depth_img2 = depth_img2 / resize_ratio
                    newCameraK = np.matmul(offset_matrix, K)
                # 20240131
                org_cam_cx = K[0][2]
                org_cam_cy = K[1][2]
                org_cam_fx = K[0][0]
                org_cam_fy = K[1][1]
                cam_cx = newCameraK[0][2]
                cam_cy = newCameraK[1][2]
                cam_fx = newCameraK[0][0]
                cam_fy = newCameraK[1][1]
                xmap_masked = xmap[:, :, np.newaxis]
                ymap_masked = ymap[:, :, np.newaxis]
                pt2 = depth_img2.astype(np.float32)
                org_pt2 = (depth_img2 * resize_ratio).astype(np.float32)
                org_pt0 = (xmap_masked - org_cam_cx) * org_pt2 / org_cam_fx
                org_pt1 = (ymap_masked - org_cam_cy) * org_pt2 / org_cam_fy

                pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
                org_depth_xyz = np.concatenate(
                    (org_pt0, org_pt1, org_pt2), axis=2).transpose(2, 0, 1)
                depth_xyz = np.concatenate(
                    (pt0, pt1, pt2), axis=2).transpose(2, 0, 1)

                if compute_cov:
                    tt = np.concatenate((pt0, pt1, pt2), axis=2).transpose(
                        2, 0, 1).reshape(3, -1).T

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(tt[tt[:, 2] > 2])
                    # o3d.visualization.draw_geometries([pcd],
                    #               zoom=0.3412,
                    #               front=[0.4257, -0.2125, -0.8795],
                    #               lookat=[2.6172, 2.0475, 1.532],
                    #               up=[-0.0694, -0.9768, 0.2024])
                    # time.sleep(3000)
                    # break
                    mean_and_cov = pcd.compute_mean_and_covariance()
                    self.ins_count += 1
                    self.max[roi_cls].append(np.array([tt[tt[:, 2] > 0].max(
                        0)[0], tt[tt[:, 2] > 0].max(0)[1], tt[tt[:, 2] > 0].max(0)[2]]))
                    self.min[roi_cls].append(np.array([tt[tt[:, 2] > 0].min(
                        0)[0], tt[tt[:, 2] > 0].min(0)[1], tt[tt[:, 2] > 0].min(0)[2]]))
                    self.cov[roi_cls].append(np.array(
                        [mean_and_cov[1][0][0], mean_and_cov[1][1][1], mean_and_cov[1][2][2]]))
                    self.mean[roi_cls].append(
                        np.array([mean_and_cov[0][0], mean_and_cov[0][1], mean_and_cov[0][2]]))

                # * Regnet
                regnet = False
                if regnet == False:
                    roi_img = np.concatenate((roi_img, depth_xyz), axis=0)
                if regnet:
                    roi_coord_2d_256 = crop_resize_by_warp_affine(
                        coord_2d, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                    ).transpose(2, 0, 1)

                    roi_img = np.concatenate(
                        (roi_img, roi_coord_2d_256), axis=0)
                    roi_img = np.concatenate((roi_img, depth_xyz), axis=0)

                roi_infos["roi_img"].append(roi_img.astype("float32"))

                # roi_coord_2d
                roi_coord_2d = crop_resize_by_warp_affine(
                    coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                ).transpose(
                    2, 0, 1
                )  # HWC -> CHW

                roi_coord_2d = np.concatenate(
                    (depth_xyz[:, ::4, ::4], roi_coord_2d))
                roi_coord_2d = np.concatenate(
                    (org_depth_xyz[:, ::4, ::4], roi_coord_2d))
                roi_infos["roi_coord_2d"].append(
                    roi_coord_2d.astype("float32"))

            for _key in roi_keys:
                if _key in ["roi_img", "roi_coord_2d", "roi_xyz", "fps"]:
                    dataset_dict[_key] = torch.as_tensor(
                        np.array(roi_infos[_key])).contiguous()
                elif _key in ["model_info", "scene_im_id", "file_name"]:
                    # can not convert to tensor
                    dataset_dict[_key] = roi_infos[_key]
                else:
                    dataset_dict[_key] = torch.as_tensor(
                        np.array(roi_infos[_key]))

            return dataset_dict
        #######################################################################################
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]

        # extent
        roi_extent = self._get_extents(dataset_name)[roi_cls]
        dataset_dict["roi_extent"] = torch.as_tensor(
            np.array(roi_extent), dtype=torch.float32)

        # load xyz =======================================================
        xyz_info = mmcv.load(inst_infos["xyz_path"])
        # print(inst_infos["xyz_path"])
        x1, y1, x2, y2 = xyz_info["xyxy"]
        # float16 does not affect performance (classification/regresion)
        xyz_crop = xyz_info["xyz_crop"]
        xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
        xyz[y1: y2 + 1, x1: x2 + 1, :] = xyz_crop

        # NOTE: full mask
        mask_obj = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (
            xyz[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
        if cfg.INPUT.SMOOTH_XYZ:
            xyz = self.smooth_xyz(xyz)

        if cfg.TRAIN.VIS:
            xyz = self.smooth_xyz(xyz)

        # override bbox info using xyz_infos
        inst_infos["bbox"] = [x1, y1, x2, y2]
        inst_infos["bbox_mode"] = BoxMode.XYXY_ABS

        # USER: Implement additional transformations if you have other types of data
        # inst_infos.pop("segmentation")  # NOTE: use mask from xyz
        anno = transform_instance_annotations(
            inst_infos, transforms, image_shape, keypoint_hflip_indices=None)

        # augment bbox ===================================================
        bbox_xyxy = anno["bbox"]
        bbox_center, scale = self.aug_bbox(cfg, bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        # add mask ----------------------------------------
        # mask_path = dataset_dict['depth_file'].split('/')
        # mask_path[-2] = 'mask_visib'
        # mask_path[-1] = mask_path[-1][:6] + '_000000.png'
        # mask_path = '/'.join(mask_path)
        # mask_img = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
        # mask_img = mask_img.reshape(mask_img.shape[0], mask_img.shape[1], 1)

        # depth xyz-----------------------------------------
        depth_img = np.array(Image.open(
            dataset_dict['depth_file'])).astype(np.float32)
        if 'depth_factor' in dataset_dict.keys():
            depth_img = depth_img / dataset_dict['depth_factor']
        else:
            depth_img = depth_img / 1000.0

        depth_img = depth_img[:, :, np.newaxis]
        depth_output = depth_img.copy()
        #depth_img = depth_img * mask_img
        # mask_depth
        # depth_mask_img = crop_resize_by_warp_affine(
        #     depth_mask_img, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        # )
        # depth_mask_img = depth_mask_img[:, :, np.newaxis]
        # rows, cols = 256,256
        # ymap = np.array([[j for i in range(cols)] for j in range(rows)]).astype(np.float32)
        # xmap = np.array([[i for i in range(cols)] for j in range(rows)]).astype(np.float32)

        # H = my_warp_affine(coord_2d, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR)
        # offset_matrix  = np.zeros((3,3))
        # offset_matrix[:2, :] = H
        # offset_matrix[2][2] = 1
        # newCameraK = np.matmul(offset_matrix,K)
        # cam_cx = newCameraK[0][2]
        # cam_cy = newCameraK[1][2]
        # cam_fx = newCameraK[0][0]
        # cam_fy = newCameraK[1][1]
        # xmap_masked = xmap[:, :, np.newaxis]
        # ymap_masked = ymap[:, :, np.newaxis]
        # pt2 = depth_mask_img.astype(np.float32)
        # pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        # pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        # mask_depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2).transpose(2,0,1)
        #
        # Org_depth_paper
        #
        # normal depth
        depth_img = crop_resize_by_warp_affine(
            depth_img, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        )
        depth_img = depth_img[:, :, np.newaxis]
        rows, cols = 256, 256
        ymap = np.array([[j for i in range(cols)]
                        for j in range(rows)]).astype(np.float32)
        xmap = np.array([[i for i in range(cols)]
                        for j in range(rows)]).astype(np.float32)

        H = my_warp_affine(coord_2d, bbox_center, scale,
                           input_res, interpolation=cv2.INTER_LINEAR)
        offset_matrix = np.zeros((3, 3))
        offset_matrix[:2, :] = H
        offset_matrix[2][2] = 1
        newCameraK = np.matmul(offset_matrix, K)

        cam_cx = newCameraK[0][2]
        cam_cy = newCameraK[1][2]
        cam_fx = newCameraK[0][0]
        cam_fy = newCameraK[1][1]
        xmap_masked = xmap[:, :, np.newaxis]
        ymap_masked = ymap[:, :, np.newaxis]

        if scale == 0:
            print('name:', dataset_dict['depth_file'])
        resize_ratio = out_res / scale
        depth_img = depth_img / resize_ratio
        pt2 = depth_img.astype(np.float32)
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy

        depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2).transpose(2, 0, 1)

        # ----------- depth xyz end-----------------------------------------
        # cv2.imwrite('oriimg.png', crop_resize_by_warp_affine(
        #    image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        # ))
        # np.save('test.npy', depth_xyz)

        # CHW, float32 tensor
        # roi_image ------------------------------------
        # roi_img = crop_resize_by_warp_affine(
        #     image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        # ).transpose(2, 0, 1)

        # import open3d as o3d
        # pcd = o3d.io.read_point_cloud('/disk2/RGBD-6dpose/GDR-Net/datasets/BOP_DATASETS/lmo/models_eval/obj_0000{:02d}.ply'.format(2))
        # pcd = np.asarray(pcd.points)
        # roi_img = crop_resize_by_warp_affine(
        #     image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        # )
        # R, t =inst_infos["pose"][:,:3], inst_infos["pose"][:, 3]
        # dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
        # R = R.astype('float64')
        # t = t.astype('float64') * 1000
        # t[2] = t[2] / resize_ratio
        # K = K.astype('float64')
        # points2d, _ = cv2.projectPoints(pcd, R, t, K,distCoeffs=dist_coeffs)
        # for point in points2d:
        #     if point[0][0] >= 0 and point[0][0] < 640 and point[0][1] >=0 and point[0][1] < 480:
        #         image[int(point[0][1])][int(point[0][0])][0] = 255
        #         image[int(point[0][1])][int(point[0][0])][1] = 255
        #         image[int(point[0][1])][int(point[0][0])][2] = 255
        # cv2.imwrite('oriimg.png', image)
        # H = my_warp_affine(coord_2d, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR)
        # TT = np.zeros((3,3))
        # TT[:2, :] = H
        # TT[2][2] = 1
        # points2d, _ = cv2.projectPoints(pcd, R, t, np.matmul(TT,K),distCoeffs=dist_coeffs)
        # for point in points2d:
        #     if point[0][0] >= 0 and point[0][0] < 256 and point[0][1] >=0 and point[0][1] < 256:
        #         roi_img[int(point[0][1])][int(point[0][0])][0] = 255
        #         roi_img[int(point[0][1])][int(point[0][0])][1] = 255
        #         roi_img[int(point[0][1])][int(point[0][0])][2] = 255
        # cv2.imwrite('roi_img.png', roi_img)
        # visualize end -------------------------------------------------------

        # calculate new camera K----------------------------------------------

        # calculate new camera K end ------------------------------------------

        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        roi_img = self.normalize_image(cfg, roi_img)

        # * Regnet
        regnet = False
        if regnet == False:
            roi_img = np.concatenate((roi_img, depth_xyz), axis=0)
        if regnet:
            roi_coord_2d_256 = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

            roi_img = np.concatenate((roi_img, roi_coord_2d_256), axis=0)
            roi_img = np.concatenate((roi_img, depth_xyz), axis=0)

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        # ablation study :
        roi_coord_2d = np.concatenate((depth_xyz[:, ::4, ::4], roi_coord_2d))
        # roi_mask ---------------------------------------
        # (mask_trunc < mask_visib < mask_obj)
        mask_visib = anno["segmentation"].astype("float32") * mask_obj
        if mask_trunc is None:
            mask_trunc = mask_visib
        else:
            mask_trunc = mask_visib * mask_trunc.astype("float32")

        if cfg.TRAIN.VIS:
            mask_xyz_interp = cv2.INTER_LINEAR
        else:
            mask_xyz_interp = cv2.INTER_NEAREST

        # maybe truncated mask (true mask for rgb)
        roi_mask_trunc = crop_resize_by_warp_affine(
            mask_trunc[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # use original visible mask to calculate xyz loss (try full obj mask?)
        roi_mask_visib = crop_resize_by_warp_affine(
            mask_visib[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        roi_mask_obj = crop_resize_by_warp_affine(
            mask_obj[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # roi_xyz ----------------------------------------------------
        roi_xyz = crop_resize_by_warp_affine(
            xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

        # region label
        # org_img cv2.imwrite('Org_rgb_paper.png',  image)
        # roi_img cv2.imwrite('roi_rgb_paper.png',  roi_img.transpose(1,2,0)[:, :, :3] * 255)
        # roi_xyz cv2.imwrite('roi_xyz_paper.png',  roi_xyz *255 / (roi_xyz.max() - roi_xyz.min()))
        # org_xyz cv2.imwrite('org_xyz_paper.png', xyz *255 / (xyz.max() - xyz.min()))
        # roi_depth_img cv2.imwrite('roi_depth_paper.png', depth_img * 255 / (depth_img.max() - depth_img.min()) )
        # roi_depth_xyz cv2.imwrite('roi_depth_xyz_paper.png', depth_xyz.transpose(1,2,0) *255 / (depth_xyz.max() - depth_xyz.min()))
        # roi_coord_2d_x cv2.imwrite('roi_coord_2d_x.png', roi_coord_2d.transpose(1,2,0)[:, :, 3:4] * 128)
        # roi_coord_2d_y cv2.imwrite('roi_coord_2d_y.png', roi_coord_2d.transpose(1,2,0)[:, :, 4:5] * 128)
        # roi_mask cv2.imwrite('roi_mask.png', roi_mask_obj * 255 )
        # roi_region  cv2.imwrite('roi_region.png', region.astype(np.int) *255 / (region.astype(np.int).max() - region.astype(np.int).min()))
        ##
        # roi_delta cv2.imwrite('roi_delta.png', delta *255 / (delta.max() - delta.min()))
        if r_head_cfg.NUM_REGIONS > 1:
            fps_points = self._get_fps_points(dataset_name)[roi_cls]
            roi_region, delta = xyz_to_region(roi_xyz, fps_points)  # HW
            R, t = inst_infos["pose"][:, :3], inst_infos["pose"][:, 3]
            # ablation study
            delta = R.dot(delta.reshape(-1, 3).T).T.reshape((64, 64, 3))
            roi_xyz = delta  # * new
            dataset_dict["roi_region"] = torch.as_tensor(
                roi_region.astype(np.int32)).contiguous()

        # roi_region:
        ##   region = np.repeat(roi_region[:,: , np.newaxis].astype(np.float16), 3 , axis=2)
        # for i in range(64):
        # for k in range(64):
        ##         region[i, k] = self._get_fps_points(dataset_name)[roi_cls][roi_region[i, k] - 1]
        ##
        ##   cv2.imwrite('roi_region.png', (region * 255 / (region.max() - region.min())).astype(np.int))
        ##  cv2.imwrite('roi_delta.png', (delta * 255 / (delta.max() - delta.min())).astype(np.int))
        roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
        # normalize xyz to [0, 1] using extent
        roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
        roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
        roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5

        # convert target to int for cls
        if ("CE" in r_head_cfg.XYZ_LOSS_TYPE) or ("cls" in cfg.MODEL.CDPN.NAME):
            # assume roi_xyz has been normalized in [0, 1]
            roi_xyz_bin = np.zeros_like(roi_xyz)
            roi_x_norm = roi_xyz[0]
            roi_x_norm[roi_x_norm < 0] = 0  # clip
            roi_x_norm[roi_x_norm > 0.999999] = 0.999999
            # [0, BIN-1]
            roi_xyz_bin[0] = np.asarray(
                roi_x_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            roi_y_norm = roi_xyz[1]
            roi_y_norm[roi_y_norm < 0] = 0
            roi_y_norm[roi_y_norm > 0.999999] = 0.999999
            roi_xyz_bin[1] = np.asarray(
                roi_y_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            roi_z_norm = roi_xyz[2]
            roi_z_norm[roi_z_norm < 0] = 0
            roi_z_norm[roi_z_norm > 0.999999] = 0.999999
            roi_xyz_bin[2] = np.asarray(
                roi_z_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            # the last bin is for bg
            roi_masks = {"trunc": roi_mask_trunc,
                         "visib": roi_mask_visib, "obj": roi_mask_obj}
            roi_mask_xyz = roi_masks[r_head_cfg.XYZ_LOSS_MASK_GT]
            roi_xyz_bin[0][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
            roi_xyz_bin[1][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
            roi_xyz_bin[2][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN

            if "CE" in r_head_cfg.XYZ_LOSS_TYPE:
                dataset_dict["roi_xyz_bin"] = torch.as_tensor(
                    roi_xyz_bin.astype("uint8")).contiguous()
            if "/" in r_head_cfg.XYZ_LOSS_TYPE and len(r_head_cfg.XYZ_LOSS_TYPE.split("/")[1]) > 0:
                dataset_dict["roi_xyz"] = torch.as_tensor(
                    roi_xyz.astype("float32")).contiguous()
        else:
            dataset_dict["roi_xyz"] = torch.as_tensor(
                roi_xyz.astype("float32")).contiguous()

        # pose targets ----------------------------------------------------------------------
        pose = inst_infos["pose"]
        allo_pose = egocentric_to_allocentric(pose)
        quat = inst_infos["quat"]
        allo_quat = mat2quat(allo_pose[:3, :3])

        # ====== actually not needed ==========
        if pnp_net_cfg.ROT_TYPE == "allo_quat":
            dataset_dict["allo_quat"] = torch.as_tensor(
                allo_quat.astype("float32"))
        elif pnp_net_cfg.ROT_TYPE == "ego_quat":
            dataset_dict["ego_quat"] = torch.as_tensor(quat.astype("float32"))
        # rot6d
        elif pnp_net_cfg.ROT_TYPE == "ego_rot6d":
            dataset_dict["ego_rot6d"] = torch.as_tensor(
                mat_to_ortho6d_np(pose[:3, :3].astype("float32")))
        elif pnp_net_cfg.ROT_TYPE == "allo_rot6d":
            dataset_dict["allo_rot6d"] = torch.as_tensor(
                mat_to_ortho6d_np(allo_pose[:3, :3].astype("float32")))
        # log quat
        elif pnp_net_cfg.ROT_TYPE == "ego_log_quat":
            dataset_dict["ego_log_quat"] = quaternion_lf.qlog(
                torch.as_tensor(quat.astype("float32"))[None])[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_log_quat":
            dataset_dict["allo_log_quat"] = quaternion_lf.qlog(
                torch.as_tensor(allo_quat.astype("float32"))[None])[0]
        # lie vec
        elif pnp_net_cfg.ROT_TYPE == "ego_lie_vec":
            dataset_dict["ego_lie_vec"] = lie_algebra.rot_to_lie_vec(
                torch.as_tensor(pose[:3, :3].astype("float32")[None])
            )[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_lie_vec":
            dataset_dict["allo_lie_vec"] = lie_algebra.rot_to_lie_vec(
                torch.as_tensor(allo_pose[:3, :3].astype("float32"))[None]
            )[0]
        else:
            raise ValueError(f"Unknown rot type: {pnp_net_cfg.ROT_TYPE}")
        dataset_dict["ego_rot"] = torch.as_tensor(
            pose[:3, :3].astype("float32"))
        dataset_dict["trans"] = torch.as_tensor(
            inst_infos["trans"].astype("float32"))

        dataset_dict["roi_points"] = torch.as_tensor(
            self._get_model_points(dataset_name)[roi_cls].astype("float32"))
        dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

        dataset_dict["roi_img"] = torch.as_tensor(
            roi_img.astype("float32")).contiguous()
        dataset_dict["roi_coord_2d"] = torch.as_tensor(
            roi_coord_2d.astype("float32")).contiguous()

        dataset_dict["roi_mask_trunc"] = torch.as_tensor(
            roi_mask_trunc.astype("float32")).contiguous()
        dataset_dict["roi_mask_visib"] = torch.as_tensor(
            roi_mask_visib.astype("float32")).contiguous()
        dataset_dict["roi_mask_obj"] = torch.as_tensor(
            roi_mask_obj.astype("float32")).contiguous()

        dataset_dict["bbox_center"] = torch.as_tensor(
            bbox_center, dtype=torch.float32)
        dataset_dict["scale"] = scale
        dataset_dict["bbox"] = anno["bbox"]  # NOTE: original bbox
        dataset_dict["roi_wh"] = torch.as_tensor(
            np.array([bw, bh], dtype=np.float32))
        dataset_dict["resize_ratio"] = resize_ratio = out_res / scale
        z_ratio = inst_infos["trans"][2] / resize_ratio
        obj_center = anno["centroid_2d"]
        delta_c = obj_center - bbox_center
        dataset_dict["trans_ratio"] = torch.as_tensor(
            [delta_c[0] / bw, delta_c[1] / bh, z_ratio]).to(torch.float32)

        # * new
        dataset_dict["depth_xyz"] = torch.as_tensor(
            depth_xyz.astype("float32")).contiguous()
        dataset_dict["fps"] = torch.as_tensor(
            fps_points.astype("float32")).contiguous()
        return dataset_dict

    def smooth_xyz(self, xyz):
        """smooth the edge areas to reduce noise."""
        xyz = np.asarray(xyz, np.float32)
        xyz_blur = cv2.medianBlur(xyz, 3)
        edges = get_edge(xyz)
        xyz[edges != 0] = xyz_blur[edges != 0]
        return xyz

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_gdrn_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    dataset_dicts = filter_invalid_in_dataset_dicts(
        dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    dataset = GDRN_DatasetFromList(
        cfg, split="train", lst=dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_gdrn_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(
            cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # load test detection results
    if cfg.MODEL.LOAD_DETS_TEST:
        det_files = cfg.DATASETS.DET_FILES_TEST
        assert len(cfg.DATASETS.TEST) == len(det_files)
        load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.DET_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.DET_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = GDRN_DatasetFromList(
        cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=trivial_batch_collator, **kwargs
    )
    return data_loader
