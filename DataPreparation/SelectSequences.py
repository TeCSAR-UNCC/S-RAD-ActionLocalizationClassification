#!/usr/bin/python3

import argparse
import os
import yaml
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--anno_path", help="Path to dataset annotations")
parser.add_argument("-m", "--min_length", default=30, help="Minimum length of a valid test sequence")

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

if __name__ == "__main__":
    args = parser.parse_args()

    gt = load_yml_file_without_meta(args.anno_path+'/VIRAT_S_000000.activities.yml')

    action_list = []
    max_frame = -1

    for action in gt:
        # print(action)
        actID = action["act"]["id2"]
        timespan = action["act"]["timespan"][0]["tsr0"]
        # print(actID, timespan)
        action_list.append([actID, timespan])
        if (timespan[1] > max_frame):
            max_frame = timespan[1]
    
    # print(action_list)
    # print(max_frame)

    timeline = [None] * (max_frame)

    for action in action_list:
        act = action[0]
        start_frame = action[1][0]
        end_frame = action[1][1]
        # print(start_frame,end_frame)
        for i in range(start_frame, end_frame):
            if (timeline[i] == None):
                timeline[i] = [act]
            else:
                timeline[i].append(act)

    # print(timeline)
    sequences = []
    time = 0
    begin = 0
    end = -1
    last_step = []
    for timestep in timeline:
        if (time != 0) and (timestep != last_step):
            end = time - 1
            seq_length = end-begin
            if (seq_length > args.min_length):
                sequences.append([begin,end])
            begin = time
        last_step = timestep
        time += 1
    # print(sequences)
    
    seq_array = numpy.array(sequences)
    seq_lengths = seq_array[:,1]-seq_array[:,0]
    min_length = numpy.amin(seq_lengths)
    max_length = numpy.amax(seq_lengths)
    mean_length = numpy.mean(seq_lengths)

    print(min_length, max_length, mean_length)