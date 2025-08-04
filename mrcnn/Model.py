import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_model():
    # Load a pre-trained Mask R-CNN
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # If dataset has a different number of classes (including background):
    num_classes = 1 + 14  # N =  object classes

    # Replace the box‐classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor too
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer    = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model
