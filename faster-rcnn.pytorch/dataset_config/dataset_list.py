import os

def main():

    #/mnt/AI_RAID/VIRAT/actev-data-repo/dataset/train/VIRAT_S_000001
    train =1
    val=1
    test =1
    HOME_DIR = '/mnt/AI_RAID/VIRAT/actev-data-repo/dataset'
    train_list_file='/mnt/AI_RAID/VIRAT/actev-data-repo/data_splits/train.lst'
    val_list_file='/mnt/AI_RAID/VIRAT/actev-data-repo/data_splits/val.lst'
    test_list_file='/mnt/AI_RAID/VIRAT/actev-data-repo/data_splits/test.lst'
    train_videonames = [x.strip() for x in open(train_list_file)]
    val_videonames = [x.strip() for x in open(val_list_file)]
    test_videonames = [x.strip() for x in open(test_list_file)]

    sequence_path = []
    count = 0
    count1 = 0
    count2 = 0
    
    if train:
        TRAIN_DIR = os.path.join(HOME_DIR,'train')
        with open('train_list.txt', 'w') as f:
            #for x in range(1):
            for x in range(len(train_videonames)):
                count =0          
                train_video_path=(os.path.join(TRAIN_DIR,train_videonames[x]))
                frame_num=[d for d in os.listdir(train_video_path)] 
                num_frames = list()
                for s in frame_num:
                    array = s.split("_")
                    start = int(array[0])
                    end = int(array[1])
                    num_frames.append(end -start)

                train_dirs=[(os.path.join(train_video_path,d)) for d in os.listdir(train_video_path)] 
                
                train_dirs= [s + "/frames/" for s in train_dirs]
                for item in train_dirs:
                    f.write("%s %s\n" % (item,num_frames[count]))
                    count = count+1
    if val:        
        VAL_DIR = os.path.join(HOME_DIR,'val')
        with open('val_list.txt', 'w') as f:
            for x in range(len(val_videonames)):
                count1=0         
                val_video_path=(os.path.join(VAL_DIR,val_videonames[x]))
                frame_num=[d for d in os.listdir(val_video_path)] 
                num_frames = list()
                for s in frame_num:
                    array = s.split("_")
                    start = int(array[0])
                    end = int(array[1])
                    num_frames.append(end -start)

                val_dirs=[(os.path.join(val_video_path,d)) for d in os.listdir(val_video_path)] 
                
                val_dirs= [s + "/frames/" for s in val_dirs]
                for item in val_dirs:
                    f.write("%s %s\n" % (item,num_frames[count1]))
                    count1 = count1+1
    if test:
        TEST_DIR = os.path.join(HOME_DIR,'test')
        with open('test_list.txt', 'w') as f:
            for x in range(len(test_videonames)):   
                count2=0       
                test_video_path=(os.path.join(TEST_DIR,test_videonames[x]))
                frame_num=[d for d in os.listdir(test_video_path)] 
                num_frames = list()
                for s in frame_num:
                    array = s.split("_")
                    start = int(array[0])
                    end = int(array[1])
                    num_frames.append(end -start)

                test_dirs=[(os.path.join(test_video_path,d)) for d in os.listdir(test_video_path)] 
                
                test_dirs= [s + "/frames/" for s in test_dirs]
                for item in test_dirs:
                    f.write("%s %s\n" % (item,num_frames[count2]))
                    count2 = count2+1
            
                
        
    print("x")
if __name__ == '__main__':
    
    main()
    