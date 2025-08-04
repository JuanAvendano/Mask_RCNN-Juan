import random
import torchvision.transforms.functional as F
import torch

class RandomHorizontalFlip:
    """Flip image + boxes + masks horizontally with given probability."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 1) flip the PIL image
            image = F.hflip(image)

            # 2) flip boxes (x1,x2 swap around width)
            w, h = image.size
            boxes = target["boxes"]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

            # 3) flip masks (Tensor[N,H,W])
            masks = target["masks"]
            target["masks"] = masks.flip(-1)
        return image, target

class ToTensor:
    """Convert PIL image to Tensor, leave target unchanged."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(augment: bool = False):
    """
    Returns a function that applies:
      - RandomHorizontalFlip(0.5)   if augment=True
      - ToTensor()                  always last
    """
    transforms = []
    if augment:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())

    def apply_all(image, target):
        for t in transforms:
            image, target = t(image, target)
        return image, target

    return apply_all