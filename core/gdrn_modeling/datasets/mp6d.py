
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from transforms3d.quaternions import mat2quat, quat2mat
from tqdm import tqdm
import numpy as np
import mmcv
from collections import OrderedDict
import time
import hashlib
import copy
import logging
import os
import os.path as osp
import sys
import scipy.io
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import ref
from lib.utils.utils import dprint, iprint, lazy_property
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.pysixd import inout, misc

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class MP6D_Dataset:
    """use image_sets(scene/image_id) and image root to get data; Here we use
    bop models, which are center aligned and have some offsets compared to
    original models."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.ann_files = data_cfg["ann_files"]  # provide scene/im_id list
        self.image_prefixes = data_cfg["image_prefixes"]  # image root

        self.dataset_root = data_cfg["dataset_root"]  # BOP_DATASETS/mp6d/
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/mp6d/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        # True (load masks but may not use it)
        self.with_masks = data_cfg["with_masks"]
        # True (load depth path here, but may not use it)
        self.with_depth = data_cfg["with_depth"]
        self.with_xyz = data_cfg["with_xyz"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get(
            "cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg["filter_invalid"]

        self.align_K_by_change_pose = data_cfg.get(
            "align_K_by_change_pose", False)
        # default: 0000~0059 and synt
        self.cam = np.array(
            [[567.53720406,   0.,         312.66570357],
             [0.,         569.36175922, 257.1729701],
                [0.,           0.,           1.]],
            dtype="float32",
        )
        # 0060~0091
        # cmu_cam = np.array([[1077.836, 0.0, 323.7872], [0.0, 1078.189, 279.6921], [0.0, 0.0, 1.0]], dtype='float32')
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [
            cat_id for cat_id, obj_name in ref.mp6d.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id)
                                     for obj_id, obj in enumerate(self.objs))
        ##########################################################
    def _load_bbox_file(self, bbox_path):
        bbox = []
        with open(bbox_path, 'r') as f:
            for line in f.readlines():
                ins_name, x1, y1, x2, y2 = line.split(' ')
                assert float(x2) > float(x1) and float(y2) > float(y1), bbox_path
                bbox.append((float(x1), float(y1), float(x2) - float(x1), float(y2) -float(y1)))
        return bbox
                
        
    def _load_from_idx_file(self, idx_file, image_root):
        """
        ====ycbv====
        idx_file: the scene/image ids
        image_root/scene contains:
            scene_gt.json
            scene_gt_info.json
            scene_camera.json
        ===MP6D====
        idx_file: 
        """
        xyz_root = osp.join(image_root, "xyz_crop")
        scene_gt_dicts = {}
        scene_gt_info_dicts = {}
        scene_cam_dicts = {}
        scene_im_ids = []  # store tuples of (scene_id, im_id, data_src)
        with open(idx_file, "r") as f:
            for line in f:
                line_split = line.strip("\r\n").split("/")  # data/0000/000000
                if line_split[0] == 'data':
                    scene_id = int(line_split[1])  # 0000
                    im_id = int(line_split[2])  # 000000
                    scene_im_ids.append((scene_id, im_id, line_split[0]))
                elif line_split[0] == 'data_syn_1':
                    scene_id = int(78)  # 0078
                    im_id = int(line_split[1])  # 000000
                    scene_im_ids.append((scene_id, im_id, line_split[0]))
                elif line_split[0] == 'data_syn_2':
                    scene_id = int(79)  # 0079
                    im_id = int(line_split[1])  # 000000
                    scene_im_ids.append((scene_id, im_id, line_split[0]))
        ######################################################
        scene_im_ids = sorted(scene_im_ids)  # sort to make it reproducible
        dataset_dicts = []

        num_instances_without_valid_segmentation = 0
        num_instances_without_valid_box = 0

        for (scene_id, im_id, data_src) in tqdm(scene_im_ids):
            record = {}
            
            if data_src == "data":
                rgb_path = osp.join(
                    image_root, f"{data_src}/{scene_id:04d}/{im_id:06d}-color.png")
                metadata = scipy.io.loadmat(
                    f"{image_root}/{data_src}/{scene_id:04d}/{im_id:06d}-meta.mat")
                bbox_path = osp.join(
                    image_root, f"{data_src}/{scene_id:04d}/{im_id:06d}-box.txt")
                if self.with_depth:
                    depth_file = osp.join(
                        image_root, f"{data_src}/{scene_id:04d}/{im_id:06d}-depth.png")

                
            else:
                rgb_path = osp.join(
                    image_root, f"{data_src}/{im_id:06d}-color.png")
                metadata = scipy.io.loadmat(
                    f"{image_root}/{data_src}/{im_id:06d}-meta.mat")
                bbox_path = osp.join(
                    image_root, f"{data_src}/{im_id:06d}-box.txt")
                if self.with_depth:
                    depth_file = osp.join(
                        image_root, f"{data_src}/{im_id:06d}-depth.png")

                record["depth_file"] = osp.relpath(depth_file, PROJ_ROOT)
            assert osp.exists(rgb_path), rgb_path
           # assert osp.exists(f"{image_root}/{data_src}/{im_id:06d}-meta.mat"), f"{image_root}{data_src}/{im_id:06d}-meta.mat"
            assert osp.exists(depth_file), depth_file
            assert osp.exists(bbox_path), bbox_path
            assert metadata is not None
            scene_bboxes = self._load_bbox_file(bbox_path)
            metadata_K = metadata['intrinsic_matrix']
            metadata_factor_depth = metadata['factor_depth']
            cam_anno = np.array(metadata_K, dtype=np.float32).reshape(3, 3)
            adapth_this_K = False

            depth_factor = 1000.0 / metadata_factor_depth
            # dprint(record['cam'])
            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "image_id": self._unique_im_id,
                "scene_im_id": f"{scene_id}/{im_id}",  # for evaluation
                "cam": cam_anno,  # self.cam,
                "depth_factor": depth_factor,
                "img_type": data_src,
            }
            record["depth_file"] = osp.relpath(depth_file, PROJ_ROOT)
            insts = []
            metadata['poses'] = metadata['poses'].transpose((2,0,1))
            
            # metadata dict_keys(['cls_indexes', 'factor_depth', 'intrinsic_matrix', 'poses', 'center'])
            for anno_i, anno in enumerate(metadata['poses']):
                obj_id = metadata["cls_indexes"][anno_i][0]
                if obj_id not in self.cat_ids:
                    continue
                # 0-based label now
                cur_label = self.cat2label[obj_id]
        
                ################ pose ###########################
                R = np.array(anno[:, :3], dtype="float32")
                trans =np.array(anno[:, 3], dtype="float32") / 1000.0  # mm->m
                pose = np.hstack([R, trans.reshape(3, 1)])
               
                if adapth_this_K:
                    # pose_uw = inv(K_uw) @ K_cmu @ pose_cmu
                    pose = np.linalg.inv(cam_anno) @ metadata_K @ pose
                    # R = pose[:3, :3]
                    trans = pose[:3, 3]

                quat = mat2quat(pose[:3, :3])

                ############# bbox ############################
                bbox = scene_bboxes[anno_i]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(min(x1, self.width), 0)
                y1 = max(min(y1, self.height), 0)
                x2 = max(min(x2, self.width), 0)
                y2 = max(min(y2, self.height), 0)
                bbox = [x1, y1, x2, y2]
                if self.filter_invalid:
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if bh <= 1 or bw <= 1:
                        num_instances_without_valid_box += 1
                        continue

                ############## mask #######################
                if self.with_masks:  # either list[list[float]] or dict(RLE)
                    if data_src == 'data':
                        
                        mask_visib_file = osp.join(
                            image_root,
                            f"{data_src}/{scene_id:04d}/mask_visib/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}_mask.png",
                        )
                        mask_full_file = osp.join(
                            image_root,
                            f"{data_src}/{scene_id:04d}/mask_visib/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}_mask.png",
                        )
                    else:
                        mask_visib_file = osp.join(
                            image_root,
                            f"{data_src}/mask_visib/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}_mask.png",
                        )
                        mask_full_file = osp.join(
                            image_root,
                            f"{data_src}/mask_visib/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}_mask.png",
                        ) # same as mask visib
                    assert osp.exists(mask_visib_file), mask_visib_file
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask[:, : , 0]
                    area = mask.sum()
                    if area < 30 and self.filter_invalid:
                        num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask)

                   
                    assert osp.exists(mask_full_file), mask_full_file

                    # load mask full
                    mask_full = mmcv.imread(mask_full_file, "unchanged")
                    mask_full = mask_full[:, :, 0]
                    mask_full = mask_full.astype("bool")
                    mask_full_rle = binary_mask_to_rle(
                        mask_full, compressed=True)

                proj = (self.cam @ trans.T).T  # NOTE: use self.cam here
                proj = proj[:2] / proj[2]

                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox,  # TODO: load both bbox_obj and bbox_visib
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": trans,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                    "mask_full": mask_full_rle,
                }

                if self.with_xyz:
                    if data_src == "data":
                        xyz_path = osp.join(
                            xyz_root,
                            f"data/{scene_id:04d}/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}-xyz.pkl",
                        )
                        inst["xyz_path"] = xyz_path
                    else:
                        xyz_path = osp.join(
                            xyz_root,
                            f"{data_src}/{im_id:06d}_{(int(metadata['cls_indexes'][anno_i])):06d}-xyz.pkl",
                        )
                        inst["xyz_path"] = xyz_path
                    assert osp.exists(xyz_path), xyz_path
                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                # TODO: using full mask and full xyz
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]
                insts.append(inst)
            if len(insts) == 0:  # and self.filter_invalid:
                continue
            record["annotations"] = insts
            dataset_dicts.append(record)
            self._unique_im_id += 1

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    num_instances_without_valid_segmentation
                )
            )
        if num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(
                    num_instances_without_valid_box)
            )
        return dataset_dicts

    def __call__(self):  # YCBV_Dataset
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    self.with_xyz,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(
            self.cache_dir,
            "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name),
        )

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        logger.info("loading dataset dicts: {}".format(self.name))
        t_start = time.perf_counter()
        dataset_dicts = []
        self._unique_im_id = 0
        for ann_file, image_root in zip(self.ann_files, self.image_prefixes):
            # logger.info("loading coco json: {}".format(ann_file))
            dataset_dicts.extend(
                self._load_from_idx_file(ann_file, image_root))

        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(
            len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(
            self.models_root, "models_{}.pkl".format(self.name))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.mp6d.obj2id[obj_name]:02d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(
                model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_mp6d_metadata(obj_names, ref_key):
    """task specific metadata."""
    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
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

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


mp6d_model_root = "BOP_DATASETS/mp6d/models_cad/"
################################################################################
default_cfg = dict(
    # name="ycbv_train_real",
    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/"),
    models_root=osp.join(
        DATASETS_ROOT, "BOP_DATASETS/mp6d/models_cad"),  # models_simple
    objs=ref.mp6d.objects,  # all objects
    # NOTE: this contains all classes
    # ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/image_sets/train.txt")],
    # image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/train_real")],
    scale_to_meter=0.001,
    with_masks=True,  # (load masks but may not use it)
    with_depth=True,  # (load depth path here, but may not use it)
    with_xyz=True,
    height=480,
    width=640,
    align_K_by_change_pose=False,
    cache_dir=osp.join(PROJ_ROOT, ".cache"),
    use_cache=True,
    num_to_load=-1,
    filter_invalid=True,
    ref_key="mp6d",
)
SPLITS_YCBV = {}
update_cfgs = {
    "mp6d_train": {
        "ann_files": [osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/image_set/train_data_list.txt")],
        "image_prefixes": [osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/")],
        
    },
    "mp6d_test": {
        "ann_files": [osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/image_set/test_data_list.txt")],
        "image_prefixes": [osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/")],
        "with_xyz": False
    },
}
for name, update_cfg in update_cfgs.items():
    used_cfg = copy.deepcopy(default_cfg)
    used_cfg["name"] = name
    used_cfg.update(update_cfg)
    num_to_load = -1
    if "_100" in name:
        num_to_load = 100
    used_cfg["num_to_load"] = num_to_load
    SPLITS_YCBV[name] = used_cfg
    
    
# single object splits ######################################################
for obj in ref.mp6d.objects:
    name = "mp6d_{}_{}".format(obj, "train")
    if name not in SPLITS_YCBV:
        SPLITS_YCBV[name] = dict(
            name=name,
            dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/"),
            models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/models_cad"),  # models_simple
            objs=[obj],
            ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/image_set/train_data_list.txt")],
            image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/")],
            scale_to_meter=0.001,
            with_masks=True,  # (load masks but may not use it)
            with_depth=True,  # (load depth path here, but may not use it)
            with_xyz=True,
            height=480,
            width=640,
            cache_dir=osp.join(PROJ_ROOT, ".cache"),
            use_cache=True,
            num_to_load=-1,
            filter_invalid=True,
            ref_key="mp6d",
        )
    
def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_YCBV:
        used_cfg = SPLITS_YCBV[name]
    else:
        assert (
            data_cfg is not None
        ), f"dataset name {name} is not registered. available datasets: {list(SPLITS_YCBV.keys())}"
        used_cfg = data_cfg
    DatasetCatalog.register(name, MP6D_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="mp6d",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_mp6d_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_YCBV.keys())


#### tests ###############################################
def test_vis():
    # python -m core.datasets.ycbv_d2 ycbv_test
    dataset_name = sys.argv[1]
    with_xyz = True if 'train' in dataset_name else False
    meta = MetadataCatalog.get(dataset_name)
    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dataset_name)

    logger.info("Done loading {} samples with {:.3f}s.".format(
        len(dicts), time.perf_counter() - t_start))

    dirname = "output/mp6d_test-data-vis"
    os.makedirs(dirname, exist_ok=True)
    objs = meta.objs
    for d in dicts:
        print(d.keys())
        img = read_image_cv2(d["file_name"], format="BGR")
        depth = mmcv.imread(d["depth_file"], "unchanged") / 1000.0

        imH, imW = img.shape[:2]
        annos = d["annotations"]
        masks = [cocosegm2mask(anno["segmentation"], imH, imW)
                 for anno in annos]
        bboxes = [anno["bbox"] for anno in annos]
        bbox_modes = [anno["bbox_mode"] for anno in annos]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS)
             for box, box_mode in zip(bboxes, bbox_modes)]
        )
        kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        # 0-based label
        cat_ids = [anno["category_id"] for anno in annos]
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t)
                   for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
        # # TODO: visualize pose and keypoints
        labels = [objs[cat_id] for cat_id in cat_ids]
        for _i in range(len(annos)):
            img_vis = vis_image_mask_bbox_cv2(
                img,
                masks[_i: _i + 1],
                bboxes=bboxes_xyxy[_i: _i + 1],
                labels=labels[_i: _i + 1],
            )
            img_vis_kpts2d = misc.draw_projected_box3d(
                img_vis.copy(), kpts_2d[_i])
            if with_xyz:
                xyz_path = annos[_i]["xyz_path"]
                xyz_info = mmcv.load(xyz_path)
                x1, y1, x2, y2 = xyz_info["xyxy"]
                xyz_crop = xyz_info["xyz_crop"].astype(np.float32)
                xyz = np.zeros((imH, imW, 3), dtype=np.float32)
                xyz[y1: y2 + 1, x1: x2 + 1, :] = xyz_crop
                xyz_show = get_emb_show(xyz)
                xyz_crop_show = get_emb_show(xyz_crop)
                img_xyz = img.copy() / 255.0
                mask_xyz = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (
                    xyz[:, :, 2] != 0)).astype("uint8")
                fg_idx = np.where(mask_xyz != 0)
                img_xyz[fg_idx[0], fg_idx[1], :] = (
                    0.5 * xyz_show[fg_idx[0], fg_idx[1], :3] +
                    0.5 * img_xyz[fg_idx[0], fg_idx[1], :]
                )
                img_xyz_crop = img_xyz[y1: y2 + 1, x1: x2 + 1, :]
                img_vis_crop = img_vis[y1: y2 + 1, x1: x2 + 1, :]
                # diff mask
                diff_mask_xyz = np.abs(
                    masks[_i] - mask_xyz)[y1: y2 + 1, x1: x2 + 1]

                grid_show(
                    [
                        img[:, :, [2, 1, 0]],
                        img_vis[:, :, [2, 1, 0]],
                        img_vis_kpts2d[:, :, [2, 1, 0]],
                        depth,
                        # xyz_show,
                        diff_mask_xyz,
                        xyz_crop_show,
                        img_xyz[:, :, [2, 1, 0]],
                        img_xyz_crop[:, :, [2, 1, 0]],
                        img_vis_crop[:, :, ::-1],
                    ],
                    [
                        "img",
                        "vis_img",
                        "img_vis_kpts2d",
                        "depth",
                        "diff_mask_xyz",
                        "xyz_crop_show",
                        "img_xyz",
                        "img_xyz_crop",
                        "img_vis_crop",
                    ],
                    row=3,
                    col=3,
                )
            else:
                grid_show(
                    [
                        img[:, :, [2, 1, 0]],
                        img_vis[:, :, [2, 1, 0]],
                        img_vis_kpts2d[:, :, [2, 1, 0]],
                        depth,
                    ],
                    ["img", "vis_img", "img_vis_kpts2d", "depth"],
                    row=2,
                    col=2,
                )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m this_module dataset_name
        "dataset_name" can be any pre-registered ones
    """
    from lib.vis_utils.image import grid_show

    from lib.utils.setup_logger import setup_logger
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_cv2
    print("used_cfg[objs]", used_cfg["objs"])
    obj_names = used_cfg["objs"]
    # models_root = osp.join(DATASETS_ROOT, "BOP_DATASETS/mp6d/models_cad")
    # for i, obj_name in enumerate(obj_names):
    #     model = inout.load_ply(
    #             osp.join(
    #                 models_root,
    #                 f"obj_{ref.mp6d.obj2id[obj_name]:02d}.ply",
    #             ),
    #             vertex_scale=0.001,
    #         )
    #         # NOTE: the bbox3d_and_center is not obtained from centered vertices
    #         # for BOP models, not a big problem since they had been centered
    #     print('model name:', obj_name)
    #     print('models_info', misc.get_bop_models_info(model["pts"]))

    print("sys.argv:", sys.argv)
    setup_logger()
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
