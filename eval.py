import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader

# 1) import your val‐dataset factory and model constructor+loader
from mrcnn.Prepare_Train_Dataset import get_val_ds
from mrcnn.Model import get_model

def mask_to_rle(mask_numpy):
    """
    Convert a H×W binary mask (0/1 numpy array) to COCO RLE dict.
    """
    rle = coco_mask.encode(np.asfortranarray(mask_numpy.astype(np.uint8)))
    # pycocotools returns bytes for 'counts' in Python3 => decode for JSON serializability
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def evaluate_coco(model, ann_file, device, batch_size=1, score_thresh=0.05):
    # — load ground truth COCO object
    coco_gt = COCO(ann_file)

    # — build a DataLoader over your validation set
    val_ds     = get_val_ds()
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    results = []
    model.to(device).eval()

    with torch.no_grad():
        for images, targets in val_loader:
            # images: tuple of [3,H,W] tensors; targets: tuple of dicts
            images = [img.to(device) for img in images]
            outputs = model(images)

            # one output dict per image in batch
            for output, target in zip(outputs, targets):
                image_id = int(target['image_id'].item())

                boxes  = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks  = output['masks'].cpu().numpy()  # shape [N,1,H,W]

                for i in range(boxes.shape[0]):
                    if scores[i] < score_thresh:
                        continue

                    x1, y1, x2, y2 = boxes[i].tolist()
                    w = x2 - x1
                    h = y2 - y1

                    result = {
                        "image_id":    image_id,
                        "category_id": int(labels[i]),
                        "bbox":        [x1, y1, w, h],
                        "score":       float(scores[i]),
                        "segmentation": mask_to_rle(masks[i, 0])
                    }
                    results.append(result)

    # — load results into COCOeval
    coco_dt = coco_gt.loadRes(results)

    # 5 separate runs to get both bbox & segm metrics
    print("\n>>> BBOX AP metrics:")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n>>> SEGMENTATION AP metrics:")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():

    annotation_json = "dataset/annotations/instances_val.json"
    weights_path    = "best_maskrcnn.pth"
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size      = 1        # how many images per forward‐pass
    score_thresh    = 0.05     # min detection score to keep


    # build model & load trained weights
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # run the COCO‐style evaluation
    evaluate_coco(model, annotation_json, device, batch_size, score_thresh)


if __name__ == "__main__":
    main()
