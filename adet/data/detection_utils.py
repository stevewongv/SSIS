import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

from detectron2.structures import (
    BitMasks
)

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_beziers_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers
    if "relation" in annotation:
        relation = transform_relation_annotations(annotation["relation"],transforms)
        annotation["relation"] = relation
        
    return annotation

def transform_relation_annotations(relation, transforms):
    #1,2
    relation = np.asfarray(relation,dtype="float32")[None]
    relation = transforms.apply_coords(relation)[0]
    return relation

def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)
    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)
    if "association" in annos[0]:
        relation = [obj.get("association",[]) for obj in annos]
        instance.gt_relation = torch.as_tensor(relation,dtype=torch.int64)
    if "light" in annos[0]:
        light =[obj.get("light",[])[:2] for obj in annos]
        instance.gt_light = torch.as_tensor(light,dtype=torch.float32)
    if "relation" in annos[0]:
        relations  = [obj.get("relation",[]) for obj in annos]
        instance.gt_relations = torch.as_tensor(relations, dtype=torch.float32)
    if "dismap" in annos[0]:
        dismaps =  [obj.get("dismap",[]) for obj in annos]
        instance.gt_dismaps = torch.stack([torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32) for x in dismaps])
    return instance


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.
    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty() & (instances.gt_masks.tensor.sum(1).sum(1) > 20.0))

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    half_len = int(len(m)/2)
    m[:half_len] &= m[half_len:]
    m[half_len:] &= m[:half_len]

    return instances[m]

def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

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
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
