from unet_pytorch import UNet
import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical

import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

device = 'cuda' if torch.cuda.is_available else 'cpu' 
print("device : "+str(device))

n_classes = 4  # Number of classes for segmentation

def collect_images(path):
    images_set = []
    for directory_path in glob.glob(path):
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.png")), key=natural_keys):
            img = cv2.imread(img_path, 1)
            # img = cv2.resize(img, (SIZE_Y, SIZE_X))
            images_set.append(img)
    return images_set

train_images = collect_images("train_image/")
train_images = np.array(train_images)

train_masks = collect_images("train_semantic/")
train_masks = np.array(train_masks)
train_masks = train_masks[:, :, :, 0]

#########data normalization
train_images = np.expand_dims(train_images, axis=4)
train_images = train_images / 255

train_masks_input = np.expand_dims(train_masks, axis=3)

y_train = train_masks_input
X_train = train_images

############make train mask data which have 4 categories to display data in categorical way

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
###############################################################


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

X_train = torch.from_numpy(X_train.astype(np.float32)).clone().squeeze().permute(0,3,1,2)
y_train_cat = torch.from_numpy(y_train_cat.astype(np.float32)).clone().permute(0,3,1,2)


model = UNet(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4)
criterion = nn.CrossEntropyLoss()

to_pil_image = T.ToPILImage()
print("=========start training========")
for i in range(X_train.shape[0]):
    print("step:"+str(i))
    model.train()
    x = X_train[i,:,:,:].to(device)
    y = y_train_cat[i,:,:,:].to(device)
    predict = model(x)

    loss = criterion(predict, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item()}")

torch.save(model.to('cpu').state_dict(), 'pytorch_weight.pth')



############## predict

class Predict:

    def __init__(self, model_parameter_file_name, test_image_path):
        self.model_parameter_file_name = model_parameter_file_name
        self.test_images = collect_images("%s" % test_image_path)

    def write_predicted_images(self):
        test_images = self.test_images
        test_images = np.expand_dims(test_images, axis=4)
        test_images = test_images / 255
        test_images = torch.from_numpy(test_images.astype(np.float32)).clone().squeeze().permute(0,3,1,2).to('cpu')
        model = UNet(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        model.load_state_dict(torch.load(self.model_parameter_file_name, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            for i in range(test_images.shape[0]):
                y_pred = model(test_images[i, :, :, :])

                y_pred = y_pred.detach().numpy().copy().transpose((1, 2, 0))
                max = np.argmax(y_pred, axis=2)
                max = max * 40
                gray_img = np.zeros((368, 1232, 3))
                gray_img = gray_img.astype('int8')
                gray_img[:, :, 0] = max
                gray_img[:, :, 1] = max * 2
                gray_img[:, :, 2] = max
                cv2.imwrite('predict/%d_pytorch.png' % i, gray_img)


Predict('pytorch_weight.pth', "./testing_images/").write_predicted_images()
