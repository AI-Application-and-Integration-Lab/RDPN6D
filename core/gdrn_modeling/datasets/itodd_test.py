import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref

from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class ITODD_BOP_TEST_Dataset(object):
    """itodd bop test splits."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root = data_cfg.get("dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/test"))
        assert osp.exists(self.dataset_root), self.dataset_root

        self.ann_file = data_cfg["ann_file"]  # json file with scene_id and im_id items

        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/itodd/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.itodd.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj, obj_id in enumerate(self.objs))
        ##########################################################

    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        # it is slow because of loading and converting masks to rle
        targets = mmcv.load(self.ann_file)

        scene_im_ids = [(item["scene_id"], item["im_id"]) for item in targets]
        scene_im_ids = sorted(list(set(scene_im_ids)))

        # load infos for each scene
        # NOTE: currently no gt info available
        # gt_dicts = {}
        # gt_info_dicts = {}
        cam_dicts = {}
        for scene_id, im_id in scene_im_ids:
            scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
            # if scene_id not in gt_dicts:
            #     gt_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            # if scene_id not in gt_info_dicts:
            #     gt_info_dicts[scene_id] = mmcv.load(
            #         osp.join(scene_root, "scene_gt_info.json")
            #     )  # bbox_obj, bbox_visib
            if scene_id not in cam_dicts:
                cam_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_camera.json"))

        for scene_id, im_id in tqdm(scene_im_ids):
            str_im_id = str(im_id)
            scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")

            # gt_dict = gt_dicts[scene_id]
            # gt_info_dict = gt_info_dicts[scene_id]
            cam_dict = cam_dicts[scene_id]

            rgb_path = osp.join(scene_root, "gray/{:06d}.tif").format(im_id)
            assert osp.exists(rgb_path), rgb_path

            depth_path = osp.join(scene_root, "depth/{:06d}.tif".format(im_id))

            scene_im_id = f"{scene_id}/{im_id}"

            K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000

            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "image_id": im_id,
                "scene_im_id": scene_im_id,  # for evaluation
                "cam": K,
                "depth_factor": depth_factor,
                "img_type": "real",  # NOTE: has background
            }
            dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "There are {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "There are {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

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
        cache_path = osp.join(self.cache_dir, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.itodd.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_itodd_metadata(obj_names, ref_key):
    """task specific metadata."""

    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


##########################################################################


SPLITS_ITODD = dict(
    itodd_bop_test=dict(
        name="itodd_bop_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/test"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/models"),
        objs=ref.itodd.objects,
        ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/test_targets_bop19.json"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=960,
        width=1280,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="itodd",
    ),
)

# single obj splits for itodd bop test
for obj in ref.itodd.objects:
    for split in ["test"]:
        name = "itodd_pbr_{}_{}".format(obj, split)
        ann_files = [
            osp.join(
                DATASETS_ROOT,
                "BOP_DATASETS/itodd/image_set/{}_{}.txt".format(obj, split),
            )
        ]
        if name not in SPLITS_ITODD:
            SPLITS_ITODD[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/test"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/models"),
                objs=[obj],  # only this obj
                scale_to_meter=0.001,
                ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/itodd/test_targets_bop19.json"),
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=960,
                width=1280,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=False,
                ref_key="itodd",
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
    if name in SPLITS_ITODD:
        used_cfg = SPLITS_ITODD[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, ITODD_BOP_TEST_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="itodd",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_itodd_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_ITODD.keys())


#### tests ###############################################
def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_mmcv(d["file_name"], format="BGR")
        detection_utils.check_image_size(d, img)
        depth = mmcv.imread(d["depth_file"], "unchanged") / d["depth_factor"]

        imH, imW = img.shape[:2]
        # annos = d["annotations"]
        # masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
        # bboxes = [anno["bbox"] for anno in annos]
        # bbox_modes = [anno["bbox_mode"] for anno in annos]
        # bboxes_xyxy = np.array(
        #     [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)])
        # kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        # quats = [anno["quat"] for anno in annos]
        # transes = [anno["trans"] for anno in annos]
        # Rs = [quat2mat(quat) for quat in quats]
        # # 0-based label
        # cat_ids = [anno["category_id"] for anno in annos]
        # K = d["cam"]
        # kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]

        # # TODO: visualize pose and keypoints
        # labels = [objs[cat_id] for cat_id in cat_ids]
        # img_vis = vis_image_bboxes_cv2(img, bboxes=bboxes_xyxy, labels=labels)
        # img_vis = vis_image_mask_bbox_cv2(img, masks, bboxes=bboxes_xyxy, labels=labels)
        # img_vis_kpts2d = img.copy()
        # for anno_i in range(len(annos)):
        #     img_vis_kpts2d = misc.draw_projected_box3d(img_vis_kpts2d, kpts_2d[anno_i])
        # grid_show([img[:, :, [2, 1, 0]], img_vis[:, :, [2, 1, 0]], img_vis_kpts2d[:, :, [2, 1, 0]], depth],
        #           [f"img:{d['file_name']}", "vis_img", "img_vis_kpts2d", 'depth'],
        #           row=2,
        #           col=2)
        grid_show(
            [img[:, :, [2, 1, 0]], depth],
            [f"img:{d['file_name']}", "depth"],
            row=1,
            col=2,
        )


if __name__ == "__main__":
    """Test the  dataset loader.

    python this_file.py dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_mmcv
    from detectron2.data import detection_utils

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()