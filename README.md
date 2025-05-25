# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS

### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name: Sri Sai Priya S

### Register Number: 212222240103

```python
from google.colab import drive
drive.mount('/content/drive')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
root = '/content/drive/MyDrive/train_test_exp4'

train_data = datasets.ImageFolder(os.path.join(root, 'Train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'Test'), transform=test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')
vggmodel=models.vgg11(pretrained=True)
for param in vggmodel.parameters():
    param.requires_grad = False
torch.manual_seed(42)
vggmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(1024, 2),
                                 nn.LogSoftmax(dim=1))
vggmodel
for param in vggmodel.parameters():
    print(param.numel())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vggmodel.classifier.parameters(), lr=0.001)
dummy_input = torch.randn(1, 3, 224, 224)
output_features = vggmodel.features(dummy_input)
output_size = output_features.view(output_features.size(0), -1).shape[1]

# Modify the first linear layer in the classifier to match the correct input size
vggmodel.classifier = nn.Sequential(
    nn.Linear(output_size, 1024),  # Update input size here
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2),
    nn.LogSoftmax(dim=1)
)
import time
start_time = time.time()

epochs = 1

max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        b+=1

        # Apply the model
        y_pred = vggmodel(X_train)
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Print interim results
        if b%200 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/8000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = vggmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/3000:.3f}%')

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

image_index = 10
# Check if the index is within the range of the dataset
if image_index < len(test_data):
    im = inv_normalize(test_data[image_index][0])
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
    plt.show()
else:
    print(f"Index {image_index} is out of range. The dataset has {len(test_data)} images.")

vggmodel.eval()
with torch.no_grad():
    new_pred = vggmodel(test_data[image_index][0].view(1,3,224,224)).argmax()
class_names[new_pred.item()]
torch.save(vggmodel.state_dict(),'Sri Sai Priya S.pt')

```

### OUTPUT

![image](https://github.com/user-attachments/assets/948d04f5-dead-4efc-a046-fca4a3e63e9e)

![image](https://github.com/user-attachments/assets/88fae377-5dcf-4e91-ab30-f5a01456e1c4)

![image](https://github.com/user-attachments/assets/f7387c7a-3f09-460d-af24-32acf75ca0c8)

![image](https://github.com/user-attachments/assets/7621d4ea-e4d5-4311-8247-42ca6118312c)

## RESULT

VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.

