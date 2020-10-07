import os

'''code to create the dataset list file 
 Format: 
       /path_to_frames/ num_of_frames class_labels
'''

def main():
    FRAME_DIR = '/mnt/AI_RAID/UCF-Sports-Actions/Frames'
    ANN_DIR ='/mnt/AI_RAID/UCF-Sports-Actions/ucfsports-anno/videos.txt'
    train_videonames = [(x.strip().split())[0] for x in open(ANN_DIR) if  (x.strip().split())[2] == 'train']
    train_labels = ([(x.strip().split())[1] for x in open(ANN_DIR) if  (x.strip().split())[2] == 'train'])
    test_videonames =[(x.strip().split())[0] for x in open(ANN_DIR) if  (x.strip().split())[2] == 'test']
    test_labels = [(x.strip().split())[1] for x in open(ANN_DIR) if  (x.strip().split())[2] == 'test']
    train = 0
    test = 1 
    if train:
        with open('train_list.txt', 'w') as f:
            for x in range(len(train_videonames)):
                train_video_path=(os.path.join(FRAME_DIR,train_videonames[x]))
                frame_num=sorted([os.path.join(train_video_path,d) for d in os.listdir(train_video_path)] )
                #train_dirs=[(os.path.join(train_video_path,d)) for d in os.listdir(train_video_path)] 
                for i in frame_num:
                  f.write("%s %s\n" % (i,train_labels[x]))

    if test:
        with open('test_list.txt', 'w') as f:
            for x in range(len(test_videonames)):
                test_video_path=(os.path.join(FRAME_DIR,test_videonames[x]))
                frame_num=sorted([os.path.join(test_video_path,d) for d in os.listdir(test_video_path)] )
                #train_dirs=[(os.path.join(train_video_path,d)) for d in os.listdir(train_video_path)] 
                for i in frame_num:
                    f.write("%s %s\n" % (i,test_labels[x]))
                



if __name__ == '__main__':
    
    main()