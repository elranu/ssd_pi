from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

class SSDInference():
    def __init__(self, confidence_threshold = 0.5, 
        weights_path = "trained_weights/VGG_VOC0712Plus_SSD_300x300_iter_240000.h5"):

        self.img_height = 300
        self.img_width = 300
        self.confidence_threshold = confidence_threshold
        self.weights_path = weights_path
        self.__compile_model()
        self.classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    
    def __compile_model(self):
        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=20,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        self.model.load_weights(self.weights_path, by_name=True)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def __apply_confidence(self, y_pred):
        return [y_pred[k][y_pred[k,:,1] > self.confidence_threshold] for k in range(y_pred.shape[0])]

    def predict(self, img_path= "predictPics/", input_images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg"]): 

        self.orig_images = [] #stores loaded original images
        resized_images = [] # Store resized versions of the images here.

        for current_img in input_images:
            self.orig_images.append(imread(img_path + current_img))
            img = image.load_img(img_path + current_img, target_size=(self.img_height, self.img_width))
            resized_images.append(image.img_to_array(img))

        y_pred = self.model.predict(np.array(resized_images))
        self.y_pred_thresh = self.__apply_confidence(y_pred)
        return self.y_pred_thresh
        
    def print_last_prediction(self):
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        for idx, pred in enumerate(self.y_pred_thresh):
            print("Predicted boxes on image number("+ str(idx) +" ):\n")
            print("   class   conf xmin   ymin   xmax   ymax")
            for detected in pred:
                print("{0} {1} {2} {3} {4} {5}".format(self.classes[int(detected[0])], 
                        detected[1], detected[2], detected[3], detected[4], detected[5]))
        
            plt.imshow(self.orig_images[idx])
            current_axis = plt.gca()
            for box in pred:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[2] * self.orig_images[idx].shape[1] / self.img_width
                ymin = box[3] * self.orig_images[idx].shape[0] / self.img_height
                xmax = box[4] * self.orig_images[idx].shape[1] / self.img_width
                ymax = box[5] * self.orig_images[idx].shape[0] / self.img_height
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
                current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
            plt.show()

