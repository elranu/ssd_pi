from skimage import transform as tf, io as skio
from ssd300Inference import SSDPredictor
import cv2
from imageio import imread

image = skio.imread("predictPics/2.jpg", )
#image = imread("predictPics/1.jpg")

print(image.shape)
predictor = SSDPredictor()

label, predicted_image = predictor.predict(image)
print(label)

winname = "Image viwer"
cv2.namedWindow(winname)        # Create a named window
cv2.imshow(winname, predicted_image)
cv2.moveWindow(winname, 40,90)  # Move it to (40,30)
cv2.waitKey(0)

# predictor.demo_predict_images()
# predictor.print_last_prediction()

