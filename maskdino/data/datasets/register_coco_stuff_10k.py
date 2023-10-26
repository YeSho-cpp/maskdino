# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
COCO_CATEGORIES = [
 
    {"id": 1, "name": "cell", "supercategory": "cell"},
 
]


def _get_coco_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_coco_stuff_10k(root):
    root = os.path.join(root, "coco", "coco_stuff_10k")
    meta = _get_coco_stuff_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train", "annotations_detectron2/train"),
        ("test", "images_detectron2/test", "annotations_detectron2/test"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_10k_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

def register_custom_coco_dataset(dataset_path: str) -> None:
   dataset_path=os.path.join(dataset_path,"coco")
   annotations_path = dataset_path + "/annotations/"
   register_coco_instances(
       "coco_train",
       {},
       annotations_path + "instances_train2017.json",
       dataset_path + "/train2017",
   )
   register_coco_instances(
       "coco_valid",
       {},
       annotations_path + "instances_val2017.json",
       dataset_path + "/val2017",
   )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_coco_stuff_10k(_root)
register_custom_coco_dataset(_root)