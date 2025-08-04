import os, torch
import torchvision
from pycocotools import mask as coco_mask
from torchvision.transforms import functional as F

class CocoWithMasks(torchvision.datasets.CocoDetection):
    def __getitem__(self, idx):
        # 1) Load image and its raw annotations
        img, anns = super().__getitem__(idx)
        img = img.convert("RGB")  # ensure 3 channels

        # prepare lists to collect boxes, labels, masks
        boxes, labels, masks = [], [], []

        for obj in anns:
            # 2) Convert COCO bbox [x, y, w, h] → [x1, y1, x2, y2]
            xmin, ymin, w, h = obj['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])

            # 3) Grab the class label
            labels.append(obj['category_id'])

            # 4) Decode segmentation (RLE or polygon) into a H×W mask
            # get image dimensions
            width, height = img.size

            for obj in anns:
                # … box & label code …
                segm = obj['segmentation']

                # 1) Polygon → RLE
                if isinstance(segm, list):
                    # segm is a list of polygons (could be multiple disconnected parts)
                    rles = coco_mask.frPyObjects(segm, height, width)
                    # merge them into a single RLE
                    rle = coco_mask.merge(rles)
                # 2) Already RLE dict (has "counts" & "size")
                elif isinstance(segm, dict) and 'counts' in segm:
                    rle = segm
                else:
                    raise ValueError(f"Unsupported segmentation type: {type(segm)}")

                # decode the RLE into a H×W numpy mask
                m = coco_mask.decode(rle)  # yields ndarray of shape [H, W]
                masks.append(torch.as_tensor(m, dtype=torch.uint8))

        # 5) Stack lists into tensors:
        boxes  = torch.as_tensor(boxes, dtype=torch.float32)   # [N,4]
        labels = torch.as_tensor(labels, dtype=torch.int64)    # [N]
        masks  = torch.stack(masks)                            # [N,H,W]

        # 6) Build the target dict
        coco_image_id = self.ids[idx]
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([coco_image_id], dtype=torch.int64)
        }

        # 7) Finally, convert image to a Tensor
        img = F.to_tensor(img)  # [3,H,W], floats in [0,1]

        return img, target


# Usage:
def train_ds():
    train_ds = CocoWithMasks(root="dataset/images/train",
                         annFile="dataset/annotations/instances_train.json")

    return train_ds

def get_val_ds():
    val_ds = CocoWithMasks(root="dataset/images/val",
                           annFile="dataset/annotations/instances_val.json")
    return val_ds