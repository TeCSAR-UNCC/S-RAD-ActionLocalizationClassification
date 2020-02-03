#!/usr/bin/python3

import argparse
import os
import yaml
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_name", help="Name of the source video and annotations")
parser.add_argument("-a", "--anno_path", help="Path to dataset annotations")
parser.add_argument("-v", "--video_path", help="Path to dataset videos")
parser.add_argument("-o", "--output_path", help="Path for output sequences files")
parser.add_argument("--min_length", default=16, help="Minimum length of a valid test sequence")
parser.add_argument("--max_length", default=200, help="Maximum length of a valid test sequence")
parser.add_argument("--fetch_gt", help="Fetch sequence annotations and dump to YAML")
parser.add_argument("--fetch_frames", help="Fetch sequence frames")
parser.add_argument("--scale_frames", type=eval, help="Resize sequence frames to dimension (h,w)")

label_dict = {"BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
    "vehicle_turning_right": 30,
    "vehicle_moving": 31,
    "vehicle_stopping" : 32,
    "vehicle_starting" :33,
    "vehicle_turning_left": 34,
    "vehicle_u_turn": 35,
    "specialized_miscellaneous": 36,
    "DropOff_Person_Vehicle" : 37,
    "Misc" : 38,
    "Drop" : 39}

def load_yml_file_without_meta(yml_file):
    """Load the ActEV YAML annotation files."""
    with open(yml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        # get the meta index first
        mi = -1
        for i in range(len(data)):
            if "meta" not in data[i]:
                mi = i
                break
        assert mi >= 0

        return data[mi:]

def select_sequences(args):
    gt = load_yml_file_without_meta(args.anno_path+'/' + args.source_name + '.activities.yml')

    action_list = []
    max_frame = -1

    for action in gt:
        actlabels = action["act"]["act2"]
        for label in actlabels.keys():
            actID = label
            break # Just grab the first one
        actors = action["act"]["actors"]
        timespan = action["act"]["timespan"][0]["tsr0"]
        actor_list = []
        for actor in actors:
            actor_list.append(actor["id1"])
            actor_timespan_struct = actor["timespan"]
            for actor_ts in actor_timespan_struct:
                actor_timespan = actor_ts["tsr0"]
                if actor_timespan[0] > timespan[0]:
                    timespan[0] = actor_timespan[0]
                if actor_timespan[1] < timespan[1]:
                    timespan[1] = actor_timespan[1]
        action_list.append([actID, actor_list, timespan])
        if (timespan[1] > max_frame):
            max_frame = timespan[1]

    timeline = [None] * (max_frame)

    for action in action_list:
        act = action[0]
        actors = action[1]
        start_frame = action[2][0]
        end_frame = action[2][1]
        for i in range(start_frame, end_frame):
            if (timeline[i] == None):
                timeline[i] = [[act, actors]]
            else:
                timeline[i].append([act, actors])

    sequences = []
    action_list = []
    time = 0
    begin = 0
    end = -1
    last_step = []
    for timestep in timeline:
        end = time - 1
        seq_length = end-begin
        if (time != 0) and ((timestep != last_step) or (seq_length >= args.max_length)):
            if ((seq_length >= args.min_length) and (last_step != None)):
                sequences.append([begin,end])
                action_list.append(last_step)
            begin = time
        last_step = timestep
        time += 1

    return sequences, action_list

def fetch_id_bboxes(args):
    gt = load_yml_file_without_meta(args.anno_path+'/' + args.source_name + '.geom.yml')

    currID = -1
    id_list = []
    bbox_list = []
    for geom in gt:
        geomid = geom["geom"]["id1"]
        geomts = geom["geom"]["ts0"]
        geombbox = geom["geom"]["g0"]
        bbox = [int(x) for x in geombbox.split(' ')]
        if (geomid != currID):
            if (currID != -1):
                id_list.append([currID, bbox_list])
                bbox_list = []
            bbox_list.append([geomts, bbox])
            currID = geomid
        elif (geom == gt[-1]):
            bbox_list.append([geomts, bbox])
            id_list.append([currID, bbox_list])
            bbox_list = []
        else:
            bbox_list.append([geomts, bbox])

    return id_list

def construct_numpy_gt(gt_list, actor_list):
    
    numpy_gt = np.zeros((1,46), dtype=int)

    for gt in gt_list:
        frame = gt["frame"]
        action_list = gt["actions"]
        frame_annotations = np.zeros((len(actor_list), 46), dtype=int) # [frame, actor_id, x1, y1, x2, y2, label0,...,label39]
        for i in range(len(actor_list)):
            frame_annotations[i,0] = frame
            frame_annotations[i,1] = actor_list[i]
        for action in action_list:
            label = action["label"]
            label_idx = label_dict[label] + 6
            actors = action["actors"]
            for actor in actors:
                actor_id = actor["actor_id"]
                bbox = actor["bbox"]
                anno_row = np.where(frame_annotations[:,1] == actor_id)
                frame_annotations[anno_row, label_idx] = 1
                frame_annotations[anno_row, 2] = bbox[0]
                frame_annotations[anno_row, 3] = bbox[1]
                frame_annotations[anno_row, 4] = bbox[2]
                frame_annotations[anno_row, 5] = bbox[3]
                # print(frame_annotations[anno_row,:])
        numpy_gt = np.append(numpy_gt, frame_annotations, axis=0)
    return numpy_gt[1:,:]


def construct_gt(args, sequences, action_list):
    id_list = fetch_id_bboxes(args)
    # id_list = [[7, [[0, [0, 1, 2, 3]], [1,[0,2,4,6]]]], [6, [[0, [0, 1, 2, 3]], [1,[0,2,4,6]]]]]

    for idx in range(len(sequences)):
        sequence = sequences[idx]
        start = sequence[0]
        end = sequence[1]
        action_set = action_list[idx]
        gt_directory = args.output_path + '/' + args.source_name + '/' + "{:06d}_".format(int(start)) + "{:06d}".format(int(end))
        if not os.path.exists(gt_directory):
            os.makedirs(gt_directory)
        gt_list = []

        actor_list = []

        for frame in range(start,end):
            frame_actions = []
            for action in action_set:
                actors = action[1]
                frame_actors = []
                for actor in actors:
                    if not actor in actor_list:
                        actor_list.append(actor)
                    actorbboxes = []
                    for id in id_list:
                        if id[0] == actor:
                            actorbboxes = id[1]
                            break
                    bbox = []
                    for framebbox in actorbboxes:
                        if framebbox[0] == frame:
                            bbox = framebbox[1]
                            break
                    frame_actors.append({'actor_id':actor, 'bbox':bbox})
                frame_actions.append({'label':action[0], 'actors':frame_actors})
            gt_list.append({'frame':frame, 'actions':frame_actions})
        
        # ground_truth = {'annotations':gt_list}
        # ground_truth_file = gt_directory + '/ground_truth.yaml'
        # with open(ground_truth_file, 'w') as file:
        #     dump = yaml.dump(ground_truth, file)

        numpy_gt = construct_numpy_gt(gt_list, actor_list)
        ground_truth_npy_file = gt_directory + '/ground_truth.npy'
        np.save(ground_truth_npy_file, numpy_gt)

def generate_frames(args, sequences):
    video_file = args.video_path + '/' + args.source_name + '.mp4'
    frameNum = 0

    print(video_file)
    print(os.path.exists(video_file))
    video_cap = cv2.VideoCapture(video_file)

    ret, frame = video_cap.read()
    while(ret):
        for sequence in sequences:
            start = sequence[0]
            end = sequence[1]
            if (frameNum in range(start,end)):
                frame_directory = args.output_path + '/' + args.source_name + '/' + "{:06d}_".format(int(start)) + "{:06d}".format(int(end)) + '/frames'
                if not os.path.exists(frame_directory):
                    os.makedirs(frame_directory)
                image_file = frame_directory + "/{:06d}.jpg".format(int(frameNum))
                cv2.imwrite(image_file, frame)
                if (args.scale_frames):
                    scaled_frame = cv2.resize(frame, args.scale_frames[::-1], interpolation=cv2.INTER_LINEAR)
                    scaled_frame_directory = args.output_path + '/' + args.source_name + '/' + "{:06d}_".format(int(start)) + "{:06d}".format(int(end)) + '/scaled_frames'
                    if not os.path.exists(scaled_frame_directory):
                        os.makedirs(scaled_frame_directory)
                    scaled_image_file = scaled_frame_directory + "/{:06d}.jpg".format(int(frameNum))
                    cv2.imwrite(scaled_image_file, scaled_frame)
        frameNum += 1
        ret, frame = video_cap.read()

    video_cap.release()

if __name__ == "__main__":
    args = parser.parse_args()

    sequences, action_list = select_sequences(args)
    if (args.fetch_gt):
        construct_gt(args, sequences, action_list)

    if (args.fetch_frames):
        generate_frames(args, sequences)
    
    seq_array = np.array(sequences)
    seq_lengths = seq_array[:,1]-seq_array[:,0]
    min_length = np.amin(seq_lengths)
    max_length = np.amax(seq_lengths)
    mean_length = np.mean(seq_lengths)

    print(min_length, max_length, mean_length)