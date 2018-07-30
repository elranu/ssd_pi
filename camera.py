import numpy as np
import os
import sys
import cv2

from ssd_predictor import SSDPredictor

class Camera:
    def __create_video_input(self, input_video_name):
        """Define VideoCapture object"""
        if(input_video_name == '0'):
            # 0: Built in webcam
            self.input_video = cv2.VideoCapture(0)       
        elif(input_video_name == '1'):
            # 1: External webcam
            self.input_video = cv2.VideoCapture(1)       
        else:
             # If not webcam, the open video
            self.input_video = cv2.VideoCapture(input_video_name)
    
    def __create_video_writer(self, output_video_name):
        """Define VideoWriter object"""
        
        # Save output video name
        self.output_video_name = output_video_name.split('.')[0]
        self.extension = output_video_name.split('.')[-1]
        
        self.output_video_name_temp = self.output_video_name + "." + self.extension
        
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")

        # Set FPS from video file
        fps = self.input_video.get(cv2.CAP_PROP_FPS)
        
        # Get videocapture's shape
        out_shape = (int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                     int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
        
        self.output_video = cv2.VideoWriter(self.output_video_name_temp, 
            fourcc, fps, out_shape)
    
    def get_frame(self):
        """Takes a frame from the video and returns it in the proper format with the original one"""
        
        # Capture frame-by-frame
        ret, original_image = self.input_video.read()
        
        # Check if frame was captured
        if(ret == False):
            raise TypeError("Image failed to capture")

        return original_image
    
    def classify(self, input_video_name, output_video_name, predictor):
        """Classify all the frames of the video and save the labeled video"""
        winname = "Image viewer"
        cv2.namedWindow(winname)
        
        self.__create_video_input(input_video_name)
        self.__create_video_writer(output_video_name)

        while(self.input_video.isOpened()):
            
            try:
                image = self.get_frame()
                pass
            except Exception as e:
                print(e)
                break
            
            label, predicted_image = predictor.predict(image, True)
            
            # Write image to video
            self.output_video.write(predicted_image)
                        
            #Show the image if flag is set
            # if(self.SHOW_IMAGE):
            cv2.imshow('image',predicted_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
        self.release()
        
        return self.output_video_name

    def release(self):
            """Release and destory everything"""
            
            self.input_video.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            print("Finished processing video, saving file to {}".format(
                self.output_video_name))