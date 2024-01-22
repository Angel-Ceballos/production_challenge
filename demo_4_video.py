import time
import cv2
from math import ceil
import os
from utils.ReadSource import FileVideoStream
from queue import Queue
from threading import Thread
import numpy as np

from utils.YoloTrt import YoloTRT
from utils.labels import obj_dic
from utils.utils import postprocess, draw_results
from  natsort import natsorted


BATCH = 4
threshold = 0.25
iou_threshold = 0.25
SHOW = True

canvas = np.zeros((540,960,3), dtype="uint8")

#LOAD MODEL
model = YoloTRT('yolov8n-batch.trt', classes=['car', 'motorcycle'])
b,c,h,w = model.model.input_spec()[0]

def post(postprocess, draw, pred_boxes, input_hw, orig_img, label_map, subset, res):
    result = np.expand_dims(pred_boxes, axis=0)
    boxes = postprocess(result, input_hw=input_hw, orig_img=orig_img)
    # if SHOW:
    img_show = draw(boxes[0], orig_img, label_map, subset)
    img_show=cv2.resize(img_show,(960,540),interpolation=cv2.INTER_LINEAR)
    res.put(img_show)


# Video's path
video_path = "/home/angel/computer_vision/production_challenge/videos/challenge_videos"
video_list = natsorted(os.listdir(video_path))
video_list = [os.path.join(video_path, v) for v in video_list]

for i in range(ceil(len(video_list)/BATCH)):
    idx = i*BATCH
    v_srcs = [video_list[idx], video_list[idx+1], video_list[idx+2], video_list[idx+3]]
    print(v_srcs)
    stime=time.time()
    num_frames = 0
    fvs = []
    res_l = []

    for v_src in v_srcs:
    #START VIDEO READ THREADS
        fvs.append(FileVideoStream(v_src).start())
        time.sleep(1.0)
        res_l.append(Queue())

    while True:
        #GET FRAME
        framed_imgs_list = []
        ori_imgs_list = []
        f_end_l = []
        for fv in fvs:
            framed_imgs,ori_imgs, f_end = fv.read()
            framed_imgs_list.append(framed_imgs[0])
            ori_imgs_list.append(ori_imgs[0])
            f_end_l.append(f_end)
        if all(f_end_l):
            print("End")
            break
        
        #BATCHING
        try:
            batch = np.array([np.transpose(norm_img, (2, 0, 1)).astype("float32")/255 for norm_img in framed_imgs_list])
            num_frames += len(framed_imgs_list)    
        except ValueError:
            print("Error")
        #PREDICTING     
        results = model.predict(batch)
        thrds = []
        for i, ori_img in enumerate(ori_imgs_list):
            thrds.append(Thread(target = post, args =(postprocess, draw_results, results[i], (h,w), ori_img, obj_dic, model.classes, res_l[i])))
            thrds[i].start()
        src_imgs = [] 
        for i, t in enumerate(thrds):
            t.join()
            src_imgs.append(res_l[i].get())
        
        if SHOW:
            l_srcs = len(src_imgs)
            if l_srcs < 4:
                for i in range(BATCH-l_srcs):
                    src_imgs.append(canvas)
                    
            numpy_horizontal1 = np.hstack((src_imgs[0],src_imgs[1]))            
            numpy_horizontal2 = np.hstack((src_imgs[2],src_imgs[3]))
            numpy_verical=np.vstack((numpy_horizontal1,numpy_horizontal2))
            cv2.imshow("RESULT",numpy_verical)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break


    cv2.destroyAllWindows()
    print('Time Elapsed {:.1f}'.format(time.time() - stime))
    print('Frames Processed: ', num_frames)
    #CLEAN UP
    for f in fvs:
        f.stop()        





