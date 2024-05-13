import hashlib
import logging
import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
import random
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask, mask2bbox_xywh
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class LM_BLENDER_Dataset(object):
    """lm blender data, from pvnet-rendering."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.ann_files = data_cfg["ann_files"]  # json files with image ids and pose/bbox
        self.image_prefixes = data_cfg["image_prefixes"]

        self.dataset_root = data_cfg["dataset_root"]  # lm_renders_blender/
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/lm/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]  # True (load masks but may not use it)
        self.with_depth = data_cfg["with_depth"]  # True (load depth path here, but may not use it)
        self.with_xyz = data_cfg["with_xyz"]
        self.depth_factor = data_cfg["depth_factor"]  # 1000.0

        self.cam = data_cfg["cam"]  #
        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg["cache_dir"]  # .cache
        self.use_cache = data_cfg["use_cache"]  # True
        # sample uniformly to get n items
        self.n_per_obj = data_cfg.get("n_per_obj", 10000)
        self.filter_invalid = data_cfg["filter_invalid"]
        ##################################################
        if self.cam is None:
            self.cam = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):  # LM_BLENDER
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    self.with_xyz,
                    self.n_per_obj,
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
        dataset_dicts = []  #######################################################
        assert len(self.ann_files) == len(self.image_prefixes), f"{len(self.ann_files)} != {len(self.image_prefixes)}"

        for ann_file, scene_root in zip(tqdm(self.ann_files), self.image_prefixes):
            # each scene is an object
            assert osp.exists(ann_file), ann_file
            scene_gt_dict = mmcv.load(ann_file)
            # sample uniformly (equal space)
            indices = list(scene_gt_dict.keys())
            if self.n_per_obj > 0:
                sample_num = min(self.n_per_obj, len(scene_gt_dict))
                sel_indices_idx = np.linspace(0, len(scene_gt_dict) - 1, sample_num, dtype=np.int32)
                sel_indices = [indices[int(_i)] for _i in sel_indices_idx]
            else:
                sel_indices = indices

            for str_im_id in tqdm(sel_indices):
                int_im_id = int(str_im_id)
                rgb_path = osp.join(scene_root, "{}.jpg").format(str_im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "{}_depth_opengl.png".format(str_im_id))

                obj_name = osp.basename(ann_file).split("_")[0]  # obj_gt.json
                obj_id = ref.lm_full.obj2id[obj_name]
                if obj_name not in self.objs:
                    continue

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": f"{obj_id}/{int_im_id}",
                    "cam": self.cam,
                    "img_type": "syn_blender",  # has bg
                }

                cur_label = self.obj2label[obj_name]  # 0-based label
                anno = scene_gt_dict[str_im_id][0]  # only one object
                R = np.array(anno["cam_R_m2c"]).reshape(3, 3)
                t = np.array(anno["cam_t_m2c"]).reshape(-1) / 1000
                pose = np.hstack([R, t.reshape(3, 1)])
                quat = mat2quat(R).astype("float32")
                proj = (record["cam"] @ t.T).T
                proj = proj[:2] / proj[2]

                bbox_visib = anno["bbox_visib"]
                x1, y1, w, h = bbox_visib
                if self.filter_invalid:
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue

                mask_path = osp.join(scene_root, "{}_mask_opengl.png".format(str_im_id))
                mask = mmcv.imread(mask_path, "unchanged")
                mask = (mask > 0).astype(np.uint8)

                area = mask.sum()
                if area < 3:  # filter out too small or nearly invisible instances
                    self.num_instances_without_valid_segmentation += 1
                    continue
                mask_rle = binary_mask_to_rle(mask, compressed=True)

                xyz_path = osp.join(scene_root, "{}_xyz_bop.pkl".format(str_im_id))
                assert osp.exists(xyz_path), xyz_path

                visib_fract = anno.get("visib_fract", 1.0)
                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox_visib,  # TODO: load both bbox_obj and bbox_visib
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": t,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                    "xyz_path": xyz_path,
                    "visib_fract": visib_fract,
                }

                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]
                record["annotations"] = [inst]
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        # if self.num_to_load > 0:
        #     self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
        #     random.shuffle(dataset_dicts)
        #     dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info(
            "loaded dataset dicts, num_images: {}, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start)
        )

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
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{ref.lm_full.obj2id[obj_name]:06d}.ply"),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_lm_metadata(obj_names, ref_key):
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


LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup
LM_OCC_OBJECTS = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
################################################################################

SPLITS_LM_BLENDER = dict(
    lm_blender_13_train=dict(
        name="lm_blender_13_train",  # BB8 training set
        dataset_root=osp.join(DATASETS_ROOT, "lm_renders_blender/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        objs=LM_13_OBJECTS,  # selected objects
        ann_files=[
            osp.join(DATASETS_ROOT, "lm_renders_blender/renders/{}_gt.json".format(_obj)) for _obj in LM_13_OBJECTS
        ],
        image_prefixes=[
            osp.join(DATASETS_ROOT, "lm_renders_blender/renders/{}".format(_obj)) for _obj in LM_13_OBJECTS
        ],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        with_xyz=True,
        depth_factor=1000.0,
        cam=ref.lm_full.camera_matrix,
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        n_per_obj=-1,  # num per class, -1 for all 10k
        filter_invalid=False,
        ref_key="lm_full",
    ),
    lmo_blender_train=dict(
        name="lmo_blender_train",
        dataset_root=osp.join(DATASETS_ROOT, "lm_renders_blender/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        objs=LM_OCC_OBJECTS,  # selected objects
        ann_files=[
            osp.join(DATASETS_ROOT, "lm_renders_blender/renders/{}_gt.json".format(_obj)) for _obj in LM_OCC_OBJECTS
        ],
        image_prefixes=[
            osp.join(DATASETS_ROOT, "lm_renders_blender/renders/{}".format(_obj)) for _obj in LM_OCC_OBJECTS
        ],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        with_xyz=True,
        depth_factor=1000.0,
        cam=ref.lmo_full.camera_matrix,
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        n_per_obj=-1,  # n per class, -1 for all 10k
        filter_invalid=False,
        ref_key="lmo_full",
    ),
)

# single obj splits
for obj in ref.lm_full.objects:
    for split in ["train"]:
        for name_prefix in ["lm", "lmo"]:
            name = "{}_blender_{}_{}".format(name_prefix, obj, split)
            ref_key = f"{name_prefix}_full"
            ann_files = [osp.join(DATASETS_ROOT, "lm_renders_blender/renders/{}_gt.json".format(obj))]
            if split in ["train"]:
                filter_invalid = True
            else:
                raise ValueError("{}".format(split))
            if name not in SPLITS_LM_BLENDER:
                SPLITS_LM_BLENDER[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "lm_renders_blender/"),
                    models_root=osp.join(DATASETS_ROOT, f"BOP_DATASETS/{name_prefix}/models"),
                    objs=[obj],  # only this obj
                    ann_files=ann_files,
                    image_prefixes=[osp.join(DATASETS_ROOT, f"lm_renders_blender/renders/{obj}")],
                    scale_to_meter=0.001,
                    with_masks=True,  # (load masks but may not use it)
                    with_depth=True,  # (load depth path here, but may not use it)
                    with_xyz=True,
                    depth_factor=1000.0,
                    cam=ref.__dict__[ref_key].camera_matrix,
                    height=480,
                    width=640,
                    cache_dir=osp.join(PROJ_ROOT, ".cache"),
                    use_cache=True,
                    n_per_obj=-1,
                    filter_invalid=False,
                    ref_key=ref_key,
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
    if name in SPLITS_LM_BLENDER:
        used_cfg = SPLITS_LM_BLENDER[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, LM_BLENDER_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lm_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_LM_BLENDER.keys())


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
        img = read_image_cv2(d["file_name"], format="BGR")
        depth = mmcv.imread(d["depth_file"], "unchanged") / 1000.0

        anno = d["annotations"][0]  # only one instance per image
        imH, imW = img.shape[:2]
        mask = cocosegm2mask(anno["segmentation"], imH, imW)
        bbox = anno["bbox"]
        bbox_mode = anno["bbox_mode"]
        bbox_xyxy = np.array(BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS))
        kpt3d = anno["bbox3d_and_center"]
        quat = anno["quat"]
        trans = anno["trans"]
        R = quat2mat(quat)
        # 0-based label
        cat_id = anno["category_id"]
        K = d["cam"]
        kpt_2d = misc.project_pts(kpt3d, K, R, trans)
        # # TODO: visualize pose and keypoints
        label = objs[cat_id]
        # img_vis = vis_image_bboxes_cv2(img, bboxes=bboxes_xyxy, labels=labels)
        img_vis = vis_image_mask_bbox_cv2(img, [mask], bboxes=[bbox_xyxy], labels=[label])
        img_vis_kpt2d = img.copy()
        img_vis_kpt2d = misc.draw_projected_box3d(
            img_vis_kpt2d, kpt_2d, middle_color=None, bottom_color=(128, 128, 128)
        )

        xyz_info = mmcv.load(anno["xyz_path"])
        xyz = np.zeros((imH, imW, 3), dtype=np.float32)
        xyz_crop = xyz_info["xyz_crop"].astype(np.float32)
        x1, y1, x2, y2 = xyz_info["xyxy"]
        xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
        xyz_show = get_emb_show(xyz)

        grid_show(
            [img[:, :, [2, 1, 0]], img_vis[:, :, [2, 1, 0]], img_vis_kpt2d[:, :, [2, 1, 0]], depth, xyz_show],
            ["img", "vis_img", "img_vis_kpts2d", "depth", "emb_show"],
            row=2,
            col=3,
        )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m core.datasets.lm_blender dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_cv2

    print("sys.argv:", sys.argv)
    setup_logger()
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())
    test_vis()
