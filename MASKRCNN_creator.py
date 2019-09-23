'''
This file describes FPN backbone based on mobileNetV3 using torchvision utils.
Moreover, torchvision version of MASK RCNN using this backbone is also described.
'''
from mobilenetv3 import MobileNetV3_forFPN, MobileNetV3 #vanilla version for weights copying

def load_pretrained_fpn(model, path): 
    import torch
    
    '''
    This function copies weights to a given model from a given checkpoint through vanilla model
    '''
    mobNetv3 = MobileNetV3(mode='small')
    state_dict = torch.load(path, map_location='cpu')
    mobNetv3.load_state_dict(state_dict)
    for param, base_param in zip(model.parameters(), mobNetv3.parameters()):
        if param.size() == base_param.size():
            param.data = base_param.data
        else:
            print('wrong size')
    return model

def mobilenetV3_fpn_backbone(pretrained_path=False):
    '''
    This function builds FPN on mobileNetV3  using torchvision utils
    '''
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    backbone = MobileNetV3_forFPN()
    
    if pretrained_path:
        load_pretrained_fpn(backbone, pretrained_path)
        
    return_layers = {'layers_os4': 0, 'layers_os8': 1, 'layers_os16': 2, 'layers_os32': 3}
    # TO-DO: it should be possible to choose which layers to use 
    in_channels_list = [16, 24, 48, 96]
    out_channels = 100
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

def maskrcnn_mobileNetV3_fpn(num_classes=3, backbone_chkp=False, **kwargs):
    '''
    This function builds torchvision version of MASK RCNN using FPN-mobileNetV3 backbone.
    '''

    from torchvision.models.detection.mask_rcnn import MaskRCNN
    backbone = mobilenetV3_fpn_backbone(backbone_chkp)
    return MaskRCNN(backbone, num_classes, **kwargs)

def keypointmaskrcnn_mobileNetV3_fpn(num_classes=3, num_keypoints=4, backbone_chkp=False, **kwargs):
    '''
    This function builds torchvision version of MASK RCNN using FPN-mobileNetV3 backbone.
    '''
    print(num_keypoints)
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
    backbone = mobilenetV3_fpn_backbone(backbone_chkp)
    return KeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)

def kp_mask_rcnn_mobileNetV3_fpn(num_classes=3, num_keypoints=4, backbone_chkp=False, **kwargs):
    '''
    This function builds torchvision version of MASK RCNN using FPN-mobileNetV3 backbone.
    '''
    print(num_keypoints)
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
    backbone = mobilenetV3_fpn_backbone(backbone_chkp)
    model = KeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)

    out_channels = model.backbone.out_channels

    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=[0, 1, 2, 3],
        output_size=14,
        sampling_ratio=2)

    mask_layers = (256, 256, 256, 256)
    mask_dilation = 1
    mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

    mask_predictor_in_channels = 256  # == mask_layers[-1]
    mask_dim_reduced = 256
    mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                        mask_dim_reduced, num_classes)

    model.roi_heads.mask_roi_pool = mask_roi_pool
    model.roi_heads.mask_head = mask_head
    model.roi_heads.mask_predictor = mask_predictor

    return model


