import os
import cv2
from torchvision.datasets import CocoDetection
from .copy_paste import copy_paste_class
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        
        return self.transforms(**output)


class SobaDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(SobaDetection, self).__init__(root, transforms, transform, target_transform)
        # from pycocotools.coco import COCO
        from pysobatools.soba import SOBA
        self.soba = SOBA(annFile)
        self.ids = list(sorted(self.soba.imgs.keys()))
        # self.coco = COCO(annFile)
        # self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        soba = self.soba
        img_id = self.ids[index]
        ann_ids = soba.getAnnIds(imgIds=img_id)
        target = soba.loadAnns(ann_ids)
        # target_asso = soba.loadAssoAnns(ann_ids)

        path = soba.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target#, target_asso

    def __len__(self):
        return len(self.ids)


@copy_paste_class
class SobaDetectionCP(SobaDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(SobaDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # # filter images without detection annotations
        # ids = []
        # for img_id in self.ids:
        #     ann_ids = self.soba.getAnnIds(imgIds=img_id, iscrowd=None)
        #     anno = self.soba.loadAnns(ann_ids)
        #     if has_valid_annotation(anno):
        #         ids.append(img_id)
        # self.ids = ids

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.soba.getAnnIds(imgIds=img_id)
        target = self.soba.loadAnns(ann_ids)
        # target = soba.loadAnns(ann_ids)
        # target_asso = self.soba.loadAssoAnns(ann_ids)

        path = self.soba.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        
        for ix, obj in enumerate(target):
            masks.append(self.soba.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])
        # asso_masks = []
        # asso_bboxes = []
        # for ix, obj in enumerate(target_asso):
        #     asso_masks.append(self.soba.annToMask(obj))
            # asso_bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            # 'asso_masks': asso_masks,
            # 'asso_bboxes': asso_bboxes
        }
        
        return self.transforms(**output)