from bcdunet import BCDU_net_D3
import glob
import tensorflow as tf
import PIL
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from warnings import filterwarnings

filterwarnings('ignore')
plt.rcParams["axes.grid"] = False
np.random.seed(101)

import re
numbers = re.compile(r'(\d+)')

# data processing
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
filelist_trainx_ph2 = sorted(glob.glob(r'train_image\*.png'), key=numericalSort)
X_train_ph2 = np.array([cv2.cvtColor(np.array(Image.open(fname)),cv2.COLOR_BGR2RGB) for fname in filelist_trainx_ph2])

filelist_trainy_ph2 = sorted(glob.glob(r'train_semantic\*.png'), key=numericalSort)
Y_train_ph21 = np.array([cv2.cvtColor(np.array(Image.open(fname)),cv2.COLOR_BGR2RGB) for fname in filelist_trainy_ph2])

def resize(filename, size = (256,256)):
    im = Image.open(filename)
    im_resized = im.resize(size, Image.ANTIALIAS)
    return (im_resized)
#   im_resized.save('/resized_ph2/X_train/X_img_'+str(i)+'.bmp', dpi = (192,256))

def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

X_train_ph2_resized = []
Y_train_ph2_resized = []

for i in range(len(filelist_trainx_ph2)):
    X_train_ph2_resized.append(resize(filelist_trainx_ph2[i]))
    Y_train_ph2_resized.append(resize(filelist_trainy_ph2[i]))

X_train_ph2 = np.array([cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB) for img in X_train_ph2_resized], dtype=np.float32)
Y_train_ph21 = np.array([cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB) for img in Y_train_ph2_resized], dtype=np.float32)
Y_train_ph2 = np.zeros([133, 256, 256])
for i in range(len(filelist_trainx_ph2)-1):
  img= cv2.cvtColor(Y_train_ph21[i],cv2.COLOR_RGB2GRAY)
  Y_train_ph2[i, :, :] = img

print(len(Y_train_ph2))
print(len(X_train_ph2))

#Split train, test and validation data
x_train1, x_test, y_train1, y_test = train_test_split(X_train_ph2, Y_train_ph2, test_size = 0.2, random_state = 101)
x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size = 0.1, random_state = 101)

print("Length of the Training Set   : {}".format(len(x_train)))
print("Length of the Test Set       : {}".format(len(x_test)))
print("Length of the Validation Set : {}".format(len(x_val)))

#normalization of data
tr_mask    = np.expand_dims(y_train, axis=3)
te_mask    = np.expand_dims(y_test, axis=3)
val_mask   = np.expand_dims(y_val, axis=3)

tr_data   = dataset_normalized(x_train)
te_data   = dataset_normalized(x_test)
val_data  = dataset_normalized(x_val)
print(tr_data.shape)

tr_mask   = tr_mask /255.
te_mask   = te_mask /255.
val_mask  = val_mask /255.
print(tr_mask.shape)
print('dataset Normalized')

#training section
#weights will be saved here
checkpoint_filepath = 'checkpoint/weight_bcdunet.ckpt'
model = BCDU_net_D3(input_size=(256, 256, 3))
model.summary()

print('Training')
batch_size = 8
nb_epoch = 10

mcp_save = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
history = model.fit(tr_data, tr_mask,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    shuffle=True,
                    verbose=1,
                    validation_data=(val_data, val_mask), callbacks=[mcp_save, reduce_lr_loss])

#model saved here
print('Trained model saved')
model_json = history.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Test section

def enhance(model,img):

    sub1 = model.predict(img.reshape(1, 256, 256, 3)).flatten()
    sub = sub1[:]
    for i in range(len(sub)):


        if sub[i] > 0.8:
            sub[i] = 1
        else:
            sub[i] = 0

    return sub

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('weight_bcdunet')
filelist_trainy = sorted(glob.glob(r'train_image\*.png'), key=numericalSort)

i=-1
for fname in filelist_trainy:
    i+=1
    image = Image.open(fname)
    original = image
    print(image)
    list1 = np.zeros([1, 256, 256, 3])
    image = np.array(image.resize((256, 256), Image.BILINEAR))
    list1[0, :, :, :] = image
    mask = enhance(loaded_model, list1[0]).reshape(256, 256)
    result = original.copy()
    result = np.array(result.resize((256, 256), Image.BILINEAR))
    result[mask != 0] = image[mask != 0]
    result[mask == 0] = (255, 255, 255)
    segimg = Image.fromarray(result, 'RGB')
    #predictions will be saved here 
    segimg.save('output/'+str(i)+'.jpg')
