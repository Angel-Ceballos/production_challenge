from threading import Thread
import sys
import cv2
import numpy as np
from utils.utils import preprocess_video
if sys.version_info>=(3,0):
    from queue import Queue

else:
    from Queue import Queue

class FileVideoStream:
    def __init__(self,path,queueSize=15):
        self.stream=cv2.VideoCapture(path)
        self.stopped=False
        self.ended=False
        self.canvas = [np.zeros((480,640,3),dtype="uint8")]
        self.empty_frame = [np.zeros((1080,1920,3),dtype="uint8")]
       
        self.Q=Queue(maxsize=queueSize)
        self.P=Queue(maxsize=queueSize)
    def start(self):
        t=Thread(target=self.update,args=())
        t.daemon=True
        t.start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            if  self.Q.qsize()<80:
                grabbed,frame=self.stream.read()
                
                if grabbed:
                    #frame=cv2.resize(frame,(300,300),interpolation=cv2.INTER_LINEAR)
                    ori_imgs, framed_imgs = preprocess_video(frame)
                    self.Q.put(framed_imgs)
                    self.P.put(ori_imgs)
                else:
                    self.Q.put(self.canvas)
                    self.P.put(self.empty_frame)
                    # self.R.put([(512, 288, 1920, 1080, 0, 224)])
                    self.ended = True
    
    def read(self):
        return self.Q.get(),self.P.get(), self.ended
    def more(self):
        return self.Q.qsize()>0
    def stop(self):
        self.stopped=True
    
