from torchvision.models import resnet50, resnet34

from utils import local

if local:
    model = resnet50(pretrained=True)
else:
    model = resnet34(pretrained=True)
