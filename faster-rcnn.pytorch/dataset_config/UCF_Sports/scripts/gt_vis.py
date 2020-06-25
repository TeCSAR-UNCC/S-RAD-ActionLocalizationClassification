import os
import cv2

def main():
    """Visual debugging of ground truth."""
   
    FRAME_DIR = '/mnt/AI_RAID/UCF-Sports-Actions/Frames/'
    ANN_DIR ='/mnt/AI_RAID/UCF-Sports-Actions/ucfsports-anno/'
    video_id =[d for d in os.listdir(FRAME_DIR)]
    for video in video_id:
        frameid = os.listdir(os.path.join(FRAME_DIR + video))
        for frame in frameid:
            frame_no = int(frame.split('.jpg')[0])
            frame_path = os.path.join((os.path.join(FRAME_DIR,video)),frame)
            im = cv2.imread(frame_path)
            ann_path = ANN_DIR + video + '.txt'
            Lines = open(ann_path, 'r').readlines()
            x,y,w,h = [line.split()[1:] for line in Lines if int(line.split()[0]) == frame_no][0]
            x2_m = int(x)+ int(w)
            y2_m = int(y)+ int(h)
            x1,y1,x2,y2 = x,y,x2_m,y2_m
            im = cv2.rectangle(im, (int(x1),int(y1)), \
             (int(x2),int(y2)), (0, 0, 204), 2)
            #cv2.putText(im,"'{0}:{1:03f}'".format((int(x1), int(y1) + 15*ind), cv2.FONT_HERSHEY_PLAIN,\
            #            1.0, (0, 0, 255), thickness=1)
            filename = video + frame 
            cv2.imwrite(filename,im)
 




if __name__ == '__main__':
    
    main()