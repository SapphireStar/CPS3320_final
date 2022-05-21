from tkinter.tix import Tree
import torch
import numpy as np
import cv2
from time import time
import os
from PIL import Image as im


class MugDetection:

    def __init__(self, capture_index, model_name):
        # Start of init mediawriter
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.media_Writer = cv2.VideoWriter("./test.avi",fourcc,30,(400,300),True)
        # end of init mediawriter
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=False)
        else:
            model = torch.hub.load('./', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        
        return self.classes[int(x)]

    def get_confidence(self,str):
        str = str.split(',')[0]
        str = str.replace('tensor(','')
        str = str.replace(')','')
        return str

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + self.get_confidence(str(row[4])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                

        return frame

    def __call__(self,cap,showbox,showfps):
          
        ret, frame = cap.read()
        assert ret
            
        frame = cv2.resize(frame, (400,300))
            
        start_time = time()
        results = self.score_frame(frame)
        labels,cord = results
        try:
            if int(labels[0])==0:
                label = "Wearing Mask"
            else:
                label = "Not Wearing Mask"
        except:
            label = "not detecting anything"

        if showbox:
            frame = self.plot_boxes(results, frame)
            
        end_time = time()
        fps = 1/(end_time - start_time)
        
        if showfps:
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        self.mp4Writer(frame)
        
        return frame,label
      
        
    def mp4Writer(self,image):
        self.media_Writer.write(image)


