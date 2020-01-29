#!/usr/bin/python3

import argparse
import os
import yaml
import numpy
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
            currID = geomid
        elif (geom == gt[-1]):
            bbox_list.append([geomts, bbox])
            id_list.append([currID, bbox_list])
        else:
            bbox_list.append([geomts, bbox])

    return id_list

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

        for frame in range(start,end):
            frame_actions = []
            for action in action_set:
                actors = action[1]
                frame_actors = []
                for actor in actors:
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

        ground_truth = {'annotations':gt_list}
        ground_truth_file = gt_directory + '/ground_truth.yaml'
        with open(ground_truth_file, 'w') as file:
            dump = yaml.dump(ground_truth, file)

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
    
    seq_array = numpy.array(sequences)
    seq_lengths = seq_array[:,1]-seq_array[:,0]
    min_length = numpy.amin(seq_lengths)
    max_length = numpy.amax(seq_lengths)
    mean_length = numpy.mean(seq_lengths)

    print(min_length, max_length, mean_length)