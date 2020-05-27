import argparse
import os
import yaml
import numpy
import cv2

if __name__ == "__main__":
    yml_file = '/mnt/AI_RAID/VIRAT/actev-data-repo/dataset/train/VIRAT_S_000001/011829_011885/ground_truth.yaml'
    with open(yml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    with open('gt1_list.txt', 'w') as f:
        label = list()
        bbox = list()
        labels = []
        bboxes = []
        for x in range (len(data['annotations'])): #get the frame numbers from the sequence
            frame=data["annotations"][x]['frame']
            for y in range(len(data["annotations"][x]['actions'])):#get the bbox in that frame
                key,value = data["annotations"][x]['actions'][y].items()
                for i in range(len(key[1])):
                   label.append(value[1])
                for j in key[1]:
                      j = str(j)
                      split_results = j.split('[', 1)[1].split(']')[0]
                      bbox.append(split_results)
            #f.write("%s %s %s\n" % (int(frame),label,bbox))
        labels+=[label] 
        bboxes+=[bbox]
        f.write("%s %s %s\n" % ((frame),labels,bboxes))
    train_file = '/mnt/AI_RAID/VIRAT/actev-data-repo/dataset/train/train_list.txt'
    sequence_path = [x.strip().split('/frames/ ')[0] for x in open(train_file)]
    