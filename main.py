from unet import multi_unet_model
import os
import glob
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import load_model

n_classes = 4  # Number of classes for segmentation


def collect_images(path):
    images_set = []
    for directory_path in glob.glob(path):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
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

# uncomment this section if you would like to train the network
#
# class Model:
#
#     def __init__(self, img_height, img_width, img_channels, n_class):
#         self.img_height = img_height
#         self.img_width = img_width
#         self.img_channels = img_channels
#         self.n_class = n_class
#         self.model = multi_unet_model(n_classes=self.n_class, IMG_HEIGHT=self.img_height, IMG_WIDTH=self.img_width,
#                                       IMG_CHANNELS=self.img_channels)
#
#     def get_model(self):
#         return self.model
#
#     def compile_model(self):
#         return model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = Model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_classes).get_model()
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# history = model.fit(X_train, y_train_cat,
#                     batch_size=4,
#                     verbose=1,
#                     epochs=5,
#                     # class_weight=class_weights,
#                     shuffle=False)
#
# model.save('test8.hdf5')


############## predict

class Predict:

    def __init__(self, model_parameter_file_name, test_image_path):
        self.model_parameter_file_name = model_parameter_file_name
        self.test_images = collect_images("%s" % test_image_path)

    def write_predicted_images(self):
        model_testing = load_model('%s' % self.model_parameter_file_name, compile=False)
        test_images = self.test_images
        test_images = np.expand_dims(test_images, axis=4)
        test_images = test_images / 255
        y_pred = model_testing.predict(test_images)
        max = np.argmax(y_pred, axis=3)
        max = max * 40

        for a in range(0, max.shape[0]):
            gray_img = np.zeros((368, 1232, 3))
            gray_img = gray_img.astype('int8')
            gray_img[:, :, 0] = max[a]
            gray_img[:, :, 1] = max[a] * 2
            gray_img[:, :, 2] = max[a]

            cv2.imwrite('predict/%d.png' % a, gray_img)


Predict('test7.hdf5', "testing_images/").write_predicted_images()

########### load and continue to run the model

# uncomment if you would like to continue to run

# model_cont_run = load_model('test6.hdf5', compile=False)
#
# model_cont_run.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = model_cont_run.fit(X_train, y_train_cat,
#                              batch_size=4,
#                              verbose=1,
#                              epochs=30,
#                              shuffle=False)
#
# model_cont_run.save('test8.hdf5')
