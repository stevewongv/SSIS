import copy
import logging
import os.path as osp

import numpy as np
import cv2
import torch
from fvcore.common.file_io import PathManager
from PIL import Image, ImageEnhance
from pycocotools import mask as maskUtils
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations, filter_empty_instances)

from scipy.ndimage import distance_transform_edt

import cv2
from skimage import measure

from detectron2.data import MetadataCatalog,DatasetCatalog



"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        # fmt: off
        self.basis_loss_on       = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set             = cfg.MODEL.BASIS_MODULE.ANN_SET
        # fmt: on

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if self.is_train:
                if np.random.rand() > 0.5 and len(annos)<10:
                    annos, image = self.double_object(annos,image)
                    dataset_dict["image"] = torch.as_tensor(
                        np.ascontiguousarray(image.transpose(2, 0, 1))
                    )
                annos = self.add_distance_map(annos)
                
            instances = annotations_to_instances(
                annos, image_shape, mask_format='bitmask'
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
    
    def add_distance_map(self, annos):
        masks = [anno['segmentation'] for anno in annos]
        distance_maps = []
        for mask in masks:
            distance_map = distance_transform_edt(mask)
            distance_map = mask - distance_map / (np.max(distance_map)+1e-7)
            distance_maps.append(distance_map)
        for i, distance_map in enumerate(distance_maps):
            annos[i]['dismap'] = distance_map
        
        return annos

    def double_object(self, annos, image):
        
        def object_shift(mask,w,h,shadow,img,fullmask):
            length = h if h < w else w
            shift_x = int((np.random.rand() - 0.5) * 2 * length /1.5)
            shift_x = shift_x if shift_x != 0 else 10
            shift_y = int((np.random.rand()+0.1) * length /1.5 )
            shift_y = shift_y if shift_y != 0 else 10
            obj = img * mask[:,:,None].repeat(3,axis=2)
            
            obj = mask_shift(obj, -shift_x, -shift_y)
            obj_mask = mask_shift(mask,-shift_x, -shift_y)
            
            obj *= 1 - (mask)[:,:,None].repeat(3,axis=2)
            obj_mask *= 1- (mask)
            
            shadow_region = shadow[:,:,None].repeat(3,axis=2)*img
            shadow_average = shadow_region.sum(axis=(0,1)) / shadow.sum()
            shadow_remap = shadow
            if shadow_average.min() < 80:
                shadow_remap = (shadow_region < shadow_average+30) * shadow[:,:,None].repeat(3,axis=2)
                shadow_remap = np.clip(shadow_remap.sum(2),0,1)
                shadow_region = shadow_remap[:,:,None].repeat(3,axis=2)*img
                shadow_average =  shadow_region.sum(axis=(0,1)) / shadow_remap.sum()
            shadow_mask = mask_shift(shadow_remap,-shift_x, -shift_y)
            shadow_mask *= 1- (mask+shadow)
            shadow_mask = (shadow_mask*(1-fullmask))[:,:,None].repeat(3,axis=2)
            shadow_shift = shadow_mask * img
            shift_average = shadow_shift.sum(axis=(0,1)) / (shadow_mask.sum() /3 + 0.000001)
            shadow_shift =  shadow_shift /  (shift_average+ 0.0000001) * (  shadow_average )
            res = img * (1-obj_mask[:,:,None]) + obj
            res = shadow_shift.astype('uint8') * shadow_mask + (1-shadow_mask) * res
            return res.astype('uint8') , obj_mask>0, shadow_mask[:,:,0]>0
                    
        def mask_shift(mask,y,x):
            padding = abs(x) if abs(x) > abs(y) else abs(y)
            h,w = mask.shape[:2]
            if len(mask.shape) == 2:
                res = np.pad(mask, ((padding,padding),(padding,padding)))
            else:
                res = np.pad(mask, ((padding,padding),(padding,padding),(0,0)))
            res = res[padding-x:padding-x+h,padding-y:padding-y+w]
            return  res

        masks = [anno['segmentation'] for anno in annos]
        bboxs = [anno['bbox'] for anno in annos]
        obj_num = int(len(masks)/2)
        random_num = 1
        nums = np.arange(obj_num)
        np.random.shuffle(nums)
        random_nums = nums[:random_num]
        count = 0
        for random_obj1 in random_nums:
            
            shadow = random_obj1 + obj_num 
            h1,w1 = bboxs[random_obj1][3] - bboxs[random_obj1][1], bboxs[random_obj1][2] - bboxs[random_obj1][0]
            full_mask = np.clip(np.array(masks).sum(0),0,1)
            
            image, obj_mask,sha_mask = object_shift(masks[random_obj1],w1,h1, masks[shadow], image, full_mask)
            sha_mask_small = sha_mask[1::2,1::2]
            obj_mask_small = obj_mask[1::2,1::2]
            if sha_mask_small.sum() > 300 and obj_mask_small.sum() > 300:
                new_obj = {
                    'iscrowd': 0,
                    'category_id': 0,
                    'association': obj_num+1,
                    'segmentation': obj_mask,
                    'bbox': BoxMode.convert(maskUtils.toBbox(maskUtils.encode(np.array(obj_mask[:,:,None], order='F',dtype='uint8'))[0])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0],
                    'bbox_mode': annos[0]['bbox_mode'],
                    'light': maskUtils.toBbox(maskUtils.encode(np.array(sha_mask[:,:,None], order='F',dtype='uint8'))[0])
                }
                new_obj_sha = {
                    'iscrowd': 0,
                    'category_id': 1,
                    'association': obj_num+1,
                    'segmentation': sha_mask,
                    'bbox': BoxMode.convert(maskUtils.toBbox(maskUtils.encode(np.array(sha_mask[:,:,None], order='F',dtype='uint8'))[0])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0],
                    'bbox_mode': annos[0]['bbox_mode'], 
                    'light': maskUtils.toBbox(maskUtils.encode(np.array(obj_mask[:,:,None], order='F',dtype='uint8'))[0])
                }
                for i, mask in enumerate(masks):
                    if i != random_obj1:
                        mask = mask * (1 - obj_mask)
                        annos[i]['segmentation'] = mask
                        mask = maskUtils.encode(np.array(mask[:,:,None], order='F',dtype='uint8'))
                        annos[i]['bbox'] = BoxMode.convert(maskUtils.toBbox(mask[0])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0]
                annos.insert(obj_num,new_obj)
                annos.append(new_obj_sha)
                obj_num+=1
                for i in range(obj_num*2):
                    if i < obj_num:
                        box = annos[i+obj_num]['bbox']
                        annos[i]['relation'] = np.array([box[2]/2+box[0]/2,box[3]/2+box[1]/2])
                    else:
                        box = annos[i-obj_num]['bbox']
                        annos[i]['relation'] = np.array([box[2]/2+box[0]/2,box[3]/2+box[1]/2])
                masks = [anno['segmentation'] for anno in annos]
                
                count += 1
            else:
                for i, mask in enumerate(masks):
                    if i != random_obj1:
                        mask = mask * (1 - obj_mask)
                        annos[i]['segmentation'] = mask
                        mask = maskUtils.encode(np.array(mask[:,:,None], order='F',dtype='uint8'))
                        annos[i]['bbox'] = BoxMode.convert(maskUtils.toBbox(mask[0])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0]
                for i in range(obj_num*2):
                    if i < obj_num:
                        box = annos[i+obj_num]['bbox']
                        annos[i]['relation'] = np.array([box[2]/2+box[0]/2,box[3]/2+box[1]/2])
                    else:
                        box = annos[i-obj_num]['bbox']
                        annos[i]['relation'] = np.array([box[2]/2+box[0]/2,box[3]/2+box[1]/2])
            
            
        return annos, image
        

