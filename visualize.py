import torch
import matplotlib.pyplot as plt
import numpy as np

from mrcnn.Prepare_Train_Dataset import get_val_ds
from mrcnn.Model                 import get_model

def visualize_predictions(model, dataset, device,
                          n_images=5, score_thresh=0.05):
    """
    Run the model on the first n_images of the dataset
    and plot each with class-colored boxes & masks.
    """
    model.to(device).eval()

    # Build a color map: one distinct color per category id
    # Assumes dataset has .coco (pycocotools COCO) with getCatIds()
    cat_ids = dataset.coco.getCatIds()
    cmap    = plt.get_cmap("tab20", len(cat_ids))
    color_map = {cid: cmap(i) for i, cid in enumerate(cat_ids)}

    # Determine which indices to show
    total = len(dataset)
    n_show = min(n_images, total)

    for idx in range(n_show):
        img, _ = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        # Convert to H×W×3 uint8
        img_np = img.mul(255).permute(1,2,0).byte().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_np)

        # Count kept detections
        keep = pred['scores'] >= score_thresh
        dets = int(keep.sum().item())
        ax.set_title(f"Image {idx} — {dets} detections (thr={score_thresh})")

        # Overlay each detection
        for box, mask, score, label in zip(
                pred['boxes'],
                pred['masks'],
                pred['scores'],
                pred['labels']
            ):

            if score < score_thresh:
                continue

            cid = int(label.cpu().item())
            color = color_map.get(cid, (0,1,0,1))  # fallback green

            # 1) draw box
            x1, y1, x2, y2 = box.cpu().numpy().tolist()
            ax.add_patch(
                plt.Rectangle((x1, y1),
                              x2-x1, y2-y1,
                              edgecolor=color,
                              fill=False, linewidth=2)
            )

            # 2) overlay mask
            m = mask[0].cpu().numpy()  # H×W binary
            # build colored mask image
            colored_mask = np.zeros_like(img_np, dtype=np.uint8)
            for ch in range(3):
                colored_mask[..., ch] = (m * int(color[ch]*255))
            # stack alpha channel as float array
            alpha = 0.5 * m
            overlay = np.dstack((colored_mask, alpha))

            ax.imshow(overlay)

        ax.axis('off')
        plt.show()


def main():
    # ← EDIT these before running ↓↓↓
    WEIGHTS    = "best_maskrcnn.pth"
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCORE_THR  = 0.5    # only show detections with score ≥ this
    N_IMAGES   = 5       # how many images from the val set to visualize
    # ↑↑↑ EDIT these ↑↑↑

    # rebuild model & load checkpoint
    model = get_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))

    # get validation dataset
    val_ds = get_val_ds()

    visualize_predictions(model, val_ds, DEVICE,
                          n_images=N_IMAGES,
                          score_thresh=SCORE_THR)


if __name__ == "__main__":
    main()
