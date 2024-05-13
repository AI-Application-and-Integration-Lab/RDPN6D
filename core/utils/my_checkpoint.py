import pickle, os
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from mmcv.runner import _load_checkpoint
from torch.nn.parallel import DataParallel, DistributedDataParallel
from pytorch_lightning.lite.wrappers import _LiteModule


class MyCheckpointer(DetectionCheckpointer):
    """https://github.com/aim-
    uofa/AdelaiDet/blob/master/adet/checkpoint/adet_checkpoint.py Same as
    :class:`DetectronCheckpointer`, but is able to convert models in AdelaiDet,
    such as LPF backbone."""

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        # HACK: deal with lite model
        if isinstance(model, (DistributedDataParallel, DataParallel, _LiteModule)):
            model = model.module
        super().__init__(
            model,
            save_dir,
            save_to_disk=save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        if filename.startswith("torchvision://") or filename.startswith(("http://", "https://")):
            loaded = _load_checkpoint(filename)  # load torchvision pretrained model using mmcv
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded
