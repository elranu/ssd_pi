import numpy as np
import os
import sys
import cv2

class Camera:
    def __int__(self, image_height=300, image_width=300):
        self.image_height = image_height
        self.image_width = image_width

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
        
        # If mp4 file, save as avi then convert
        if(self.extension == 'mp4'):
            self.output_video_name_temp = self.output_video_name + '.avi'
            self.CONVERT_TO_MP4 = True
        else:
            self.output_video_name_temp = ".".join(self.output_video_name, self.extension)
        
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*"X264")

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
        
        self.__create_video_input(input_video_name)
        self.__create_video_writer(output_video_name)

        while(self.input_video.isOpened()):
            
            try:
                images, original_image = self.get_frame()
                pass
            except Exception as e:
                print(e)
                break
            
            label = predictor.predict(images)

            # Print the text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_image,label,(50,50), font, 
                        1,(255,255,255),2,cv2.LINE_AA)
            
            # Write image to video
            self.output_video.write(original_image)
                        
            # Show the image if flag is set
            if(self.SHOW_IMAGE):
                cv2.imshow('image',original_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        self.release()
        
        if(self.CONVERT_TO_MP4):
            self.convert_avi_to_mp4(self.output_video_name_temp,
                self.output_video_name)
        
        return self.output_video_name

