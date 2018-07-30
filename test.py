from ssd_predictor import SSDPredictor
import cv2

image = cv2.imread("predictPics/2.jpg")

print(image.shape)
predictor = SSDPredictor()

label, predicted_image = predictor.predict(image, True)
print(label)

winname = "Image viwer"
cv2.namedWindow(winname)        # Create a named window
cv2.imshow(winname, predicted_image)
cv2.moveWindow(winname, 40,90)  # Move it to (40,30)
cv2.waitKey(0)

# predictor.demo_predict_images()
# predictor.print_last_prediction()

