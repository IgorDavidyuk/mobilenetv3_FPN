import torch
from mobilenetv3 import MobileNetV3_forFPN, MobileNetV3, load_pretrained_fpn

def test_loaded_weights():
    torch.backends.cudnn.deterministic = True
    path = '/home/davidyuk/Projects/backbones/pytorch-mobilenet-v3/mobilenetv3_small_67.4.pth.tar'
    mn3_fpn = MobileNetV3_forFPN()
    mn3_fpn = load_pretrained_fpn(mn3_fpn, path)

    mobNetv3 = MobileNetV3(mode='small')
    state_dict = torch.load(path, map_location='cpu')
    mobNetv3.load_state_dict(state_dict)
    mobNetv3 = mobNetv3.features[:12]
    for param, base_param in zip(mn3_fpn.parameters(), mobNetv3.parameters()):
        assert ((param == base_param).all()), 'params differ'
    #print(len(tuple(mn3_fpn.parameters())),len(tuple(mobNetv3.parameters())))
    # mobNetv3.eval()
    # mn3_fpn.eval()

    image = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        output = mn3_fpn.forward(image)
        output1 = mobNetv3.forward(image)
        
    if (output == output1).all():
        print('test passed')
    else:
        print('test failed')
    torch.backends.cudnn.deterministic = False

def compare_output(model1, model2, tensor=torch.rand(1, 3, 64, 64), mode_train=True):
    if mode_train:
        model1.train(); model2.train()
    else:
        model1.eval(); model2.eval()
    with torch.no_grad():
        output1 = model1.forward(tensor)
        output2 = model2.forward(tensor)
    return output1, output2