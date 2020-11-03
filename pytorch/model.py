from torchvision.models import resnet50, resnet34

from utils import local, Config


if local:
    model = resnet50(pretrained=Config['finetune'])
else:
    model = resnet34(pretrained=Config['finetune'])
