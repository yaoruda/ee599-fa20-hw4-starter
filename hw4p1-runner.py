import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import json
from sklearn import preprocessing

data_dir = './data/test/images'
checkpoint_path = './hw4p1.pth'

batch_size = 8

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_set = datasets.ImageFolder(data_dir, transform=transform)

dataloader = DataLoader(
    test_set,
    shuffle=False,
    batch_size=batch_size)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = checkpoint['model']

encoder = preprocessing.LabelEncoder()
encoder.fit(checkpoint['label'])

model.eval()
running_corrects = 0
k = 0
for inputs, labels in dataloader:
    k += 1
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        # convert dataset idx to categoryId
        ids = [test_set.classes[idx] for idx in labels]
        # convert categoryId to model one-hot output
        one_hot_labels = torch.Tensor(encoder.transform(ids))

        running_corrects += torch.sum(pred == one_hot_labels)

        if not (k % 10):
            print(f'batch: {k:4} / {len(dataloader)}')

scores = {
  'Success': '1',
  'Accuracy': 100.0 * running_corrects.item() / len(dataloader.dataset)
}

out = {'scores': scores}
print(json.dumps(out))
