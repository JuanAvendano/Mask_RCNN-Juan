# inference.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from mrcnn.Prepare_Train_Dataset import train_ds
from mrcnn.Model import model

model=model()
def load_model_weights(path, device):
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
train_ds = train_ds()
def run_inference(idx: int = 2, score_threshold: float = 0.5):
    # 1) set up device and model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device).eval()

    # 2) grab one sample from your training dataset
    img, _ = train_ds[idx]  # img is a [3,H,W] tensor in [0,1]

    # 3) forward-pass
    with torch.no_grad():
        outputs = model([img.to(device)])
    pred = outputs[0]

    # 4) convert image back to numpy for plotting
    img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

    # 5) plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    ax = plt.gca()

    for box, mask, score in zip(pred['boxes'], pred['masks'], pred['scores']):
        if score < score_threshold:
            continue

        # draw box
        x1, y1, x2, y2 = box.cpu().numpy()
        ax.add_patch(plt.Rectangle((x1, y1),
                                   x2 - x1, y2 - y1,
                                   fill=False, linewidth=2))

        # overlay mask in green
        m = mask[0].cpu().numpy()
        colored_mask = np.zeros_like(img_np)
        colored_mask[..., 1] = (m * 255).astype(np.uint8)
        ax.imshow(np.dstack((colored_mask, m * 0.5)))

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mask R-CNN inference & viz")
    p.add_argument("--idx", type=int, default=0,
                   help="Which sample index from train_ds to run on")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Score threshold for displaying predictions")
    args = p.parse_args()

    run_inference(idx=args.idx, score_threshold=args.threshold)
