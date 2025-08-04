import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

def visualize_masks(images_dir, annotation_file, output_dir=None, num_samples=5):
    """
    Load images and their COCO-format masks, overlay masks on images, and display/save the results.

    Args:
        images_dir (str): Path to the directory containing training images.
        annotation_file (str): Path to the COCO-style JSON annotation file.
        output_dir (str, optional): Directory to save overlay images (if None, images are not saved). Defaults to None.
        num_samples (int): Number of random samples to display. Defaults to 5.
    """
    # Load COCO annotations
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    sampled_ids = random.sample(img_ids, min(num_samples, len(img_ids)))

    for img_id in sampled_ids:
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Plot image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Overlay each mask with random semi-transparent color
        for ann in anns:
            mask = coco.annToMask(ann)
            color = np.random.rand(3)
            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[..., :3] = color
            colored_mask[..., 3] = mask * 0.5  # set alpha channel
            plt.imshow(colored_mask)

        plt.axis('off')
        plt.title(f"Image ID: {img_id} - {img_info['file_name']}")

        # Save overlay if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{img_id}_overlay.png")
            plt.savefig(out_path, bbox_inches='tight')

        plt.show()

if __name__ == "__main__":
    # Example usage
    images_dir = "../dataset/images/val"
    annotation_file = "../dataset/annotations/instances_val.json"
    # Optional: specify an output directory to save overlays
    # output_dir = "../overlays"

    # Visualize 10 random training images with masks
    visualize_masks(images_dir, annotation_file, output_dir=output_dir, num_samples=5)
