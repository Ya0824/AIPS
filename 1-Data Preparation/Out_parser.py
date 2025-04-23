#!/usr/bin/env python3
"""
Standalone script to process power reports for security analysis.
"""
import re
import numpy as np
import argparse
import sys

def procPowRepo(stimunum, timelist, filepath, output_path):
    """ process power reports obtained from ptpx simulation """
    # open power reports
    try:
        with open(filepath, 'r') as powerfile:
            lines = powerfile.readlines()
    except IOError as e:
        sys.exit(f"Error opening file {filepath}: {e}")

    # containers for hierarchy, indices and time points
    powerhier = []
    powerindex = []
    timepoint = []

    # patterns for indices and time markers
    tmpl_index = r'.index Pc\(.*?\) [\d]+ Pc'
    tmpl_time = r'^\d+\n$'

    for line in lines:
        if re.match(tmpl_index, line):
            parts = line.split()
            powerhier.append(parts[1])
            powerindex.append(int(parts[2]))
        elif re.match(tmpl_time, line):
            timepoint.append(int(line.strip()))

    timepoint = np.asarray(timepoint)
    powerindex = np.asarray(powerindex)

    # unpack timing parameters
    start_timepoint, timeperiod, inter_timepoint, timescale = timelist
    sum_timepoint = timeperiod + inter_timepoint

    # initialize power trace array
    trace_length = int(timeperiod / timescale)
    powertrace = np.full((len(powerindex), stimunum, trace_length), 1e-11, dtype=np.float32)

    # pattern for power values
    tmpl_power = r'^[\d]+  [\d\.+-eE]+.*'

    tracked_timepoint = 0
    stim_recorded = 0
    for line in lines:
        if stim_recorded >= stimunum:
            break
        # time markers
        if re.match(tmpl_time, line):
            current_tp = int(line.strip())
            tracked_timepoint = current_tp - start_timepoint - (sum_timepoint * stim_recorded)
            if current_tp > start_timepoint + stimunum * sum_timepoint:
                break
            if tracked_timepoint >= sum_timepoint:
                stim_recorded += 1
                tracked_timepoint = current_tp - start_timepoint - (sum_timepoint * stim_recorded)
        # power entries
        elif re.match(tmpl_power, line):
            if 0 <= tracked_timepoint < timeperiod:
                parts = line.split()
                idx = int(parts[0]) - 1
                t_idx = int(tracked_timepoint / timescale)
                powertrace[idx, stim_recorded, t_idx] = float(parts[1])

    # replace default values
    for t in range(powertrace.shape[2]):
        for s in range(stimunum):
            missing = (powertrace[:, s, t] == 1e-11).nonzero()[0]
            if t >= 1:
                powertrace[missing, s, t] = powertrace[missing, s, t-1]

    # collapse first hierarchy level
    result = powertrace[0, ...]
    print(f"The shape of the power trace is {result.shape}\n")

    # save output
    np.save(output_path, result[:1000,:])
    print(f"Power trace saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Process power reports for security analysis.")
    parser.add_argument("--stimunum", type=int, default=1000, help="Number of stimuli to process")
    parser.add_argument("--start", type=int, default=8000, help="Start timepoint")
    parser.add_argument("--period", type=int, default=160, help="Duration of each time period")
    parser.add_argument("--interval", type=int, default=6720, help="Inter timepoint interval")
    parser.add_argument("--scale", type=int, default=1, help="Time scale factor")
    parser.add_argument("--input", type=str, default = 'frontdata/PowerFile0_train.out', help="Path to the input power report file")
    parser.add_argument("--output", type=str, default="Power_trace_train.npy",help="Path to save the output NumPy file")
    args = parser.parse_args()

    timelist = [args.start, args.period, args.interval, args.scale]
    procPowRepo(args.stimunum, timelist, args.input, args.output)

if __name__ == "__main__":
    main()

