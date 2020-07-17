import os
import numpy as np

'''code to create the dataset list file 
 Format: 
       /path_to_frames/ num_of_frames class_labels
'''

def main():
    #FRAME_DIR = '/mnt/AI_RAID/UCF-Sports-Actions/Frames'
    ANN_DIR ='/mnt/AI_RAID/jhmdb/splits/JHMDB_RGB_1_split_2.testlist'
    a =[(x.strip().split('/')) for x in open(ANN_DIR)]
    labels =np.unique([a[i][0] for i in range(len(a))])
    framelist = [a[i][0] + '/' +a[i][1] for i in range(len(a))if a[i][2] == '00001.png']
    framepath=[os.path.join('/mnt/AI_RAID/jhmdb/frames/',fr) for fr in framelist]
    num_frame = []
    for frame in framepath :
        num_frame.append(len([name for name in os.listdir(frame) if os.path.isfile(os.path.join(frame, name))]))
    with open('JHMDB_RGB_1_split_2.testlist', 'w') as f:
            for x in range(len(framepath)):
                label = ((framepath[x].split('/mnt/AI_RAID/jhmdb/frames/'))[1]).split('/')[0]
                label_idx = [idx+1 for idx in range(len(labels)) if labels[idx] == label]
                f.write("%s %s %s\n" % (framepath[x],num_frame[x],label_idx[0]))
    print("Done")

if __name__ == '__main__':
    
    main()