#!/usr/bin/env python3

"""
Detailed, GPU-Optimized VCD Processing Script 

Steps:
  1) Split a large VCD into partial VCDs (Step 1).
  2) Parse each partial VCD to extract toggles (Step 2).
  3) Summarize parsed data into a single NumPy array (Step 3).

"""

import argparse
import os
import re
import time
import pickle

from multiprocessing import Process
from tqdm import tqdm

# For GPU arrays (fallback to NumPy if Cupy not installed)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_LIB = 'cupy'
except ImportError:
    import numpy as cp  # fallback
    GPU_AVAILABLE = False
    GPU_LIB = None

import numpy as np


###############################################################################
# STEP 1: ProcessVCD (Splitting a VCD file into multiple partial VCDs)
###############################################################################
class ProcessVCDInMemory:
    """
    Step 1:
      * Reads the entire VCD file into memory.
      * Builds a time -> line_index map by scanning once.
      * Writes out a 'header' file (all lines until time > 0).
      * For each plaintext, slices lines from [left_time, right_time] 
        and writes them to a partial file.
    """

    def __init__(self,
                 vcd_init_path, 
                 vcd_final_path,
                 header_path,
                 start_time_point,
                 num_plaintexts,
                 desired_time_interval,
                 off_time_interval):
        self.vcd_init_path = vcd_init_path
        self.vcd_final_path = vcd_final_path
        self.header_path = header_path
        self.start_time_point = start_time_point
        self.num_plaintexts = num_plaintexts
        self.desired_time_interval = desired_time_interval
        self.off_time_interval = off_time_interval

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.vcd_final_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.header_path), exist_ok=True)

    def run(self):
        """Executes the splitting process."""
        print("=== Step 1: Splitting the large VCD file ===")

        print("Reading entire VCD file into memory...")
        with open(self.vcd_init_path, "r") as f:
            self.lines = f.readlines()

        # Build time -> line_index mapping
        pattern_time = re.compile(r'^#(\d+)\s*$')
        time_to_line = {}
        time_list = []
        header_end_index = len(self.lines)  # default if no time > 0 found

        print("Indexing times in VCD...")
        for idx, line in enumerate(tqdm(self.lines, desc="Indexing Times", unit="lines")):
            m = pattern_time.match(line)
            if m:
                t_val = int(m.group(1))
                time_to_line[t_val] = idx
                time_list.append(t_val)
                if t_val > 0 and header_end_index == len(self.lines):
                    header_end_index = idx

        time_list.sort()

        # Write the header file
        print("Writing header file...")
        with open(self.header_path, 'w', newline='\n') as hf:
            hf.writelines(self.lines[:header_end_index])

        # Helper: get line index at or after t
        def get_line_index(t):
            import bisect
            pos = bisect.bisect_left(time_list, t)
            if pos < len(time_list):
                return time_to_line[time_list[pos]]
            else:
                return len(self.lines) - 1

        # Split into partial VCDs
        print("Splitting into partial VCD files...")
        for i in tqdm(range(self.num_plaintexts), desc="Splitting VCD", unit="file"):
            left_time = self.start_time_point + self.off_time_interval * i
            right_time = left_time + self.desired_time_interval

            start_idx = get_line_index(left_time)
            end_idx = get_line_index(right_time)

            if start_idx >= len(self.lines):
                break

            partial_slice = self.lines[start_idx : end_idx + 1]

            # Adjust times so left_time => #0
            adjusted_slice = []
            for line in partial_slice:
                m = pattern_time.match(line)
                if m:
                    old_t = int(m.group(1))
                    new_t = old_t - left_time
                    adjusted_slice.append(f"#{new_t}\n")
                else:
                    adjusted_slice.append(line)

            out_path = f"{self.vcd_final_path}{i}.vcd"
            with open(out_path, 'w', newline='\n') as out_f:
                out_f.writelines(adjusted_slice)

        print("Done splitting VCD (Step 1).")


###############################################################################
# STEP 2: ParserVCD (Parsing each partial VCD, producing .pkl of toggles)
###############################################################################
class ParserVCD:

    def setup(self, relevant_module, time_period):

        module_symbol_idx = {}
        big_lst = []
        for c, part in enumerate(relevant_module):
            all_parts = part.split(' ')
            symbol = all_parts[3]
            pin = ''.join(all_parts[4:-1])
            module_symbol_idx[symbol] = c

            big_lst.append([symbol, pin])

        for b in big_lst:
            b.extend([[] for _ in range(time_period)])
        return module_symbol_idx, big_lst

    def get_vcd_data(self, switchfile_path):

        data = open(switchfile_path, 'r').read().split('\n')
        return data

    def get_relevant_module(self, header_file_path, module_name):
        
        data = open(header_file_path, 'r').read().split('\n')
        start = 0
        end = 0
        for i, x in enumerate(data):
            if '$scope module' in x:
                if '$scope module {} $end'.format(module_name) == x:
                    start = i
                elif start:
                    end = i
                    break
        relevant_content = [x for x in data[start + 1:end] if any(x)]
        return relevant_content

    def remove_between_dumpvars_dumpon(self, data):
        
        start_irrelevant = 0
        end_irrelevant = 0
        for i, d in enumerate(data):
            if d == '$dumpvars':
                start_irrelevant = i
            elif d == '$dumpon':
                end_irrelevant = i
                break

        data = data[:start_irrelevant] + [''] * (end_irrelevant - start_irrelevant) + data[end_irrelevant:]
        data = data[:start_irrelevant] + data[end_irrelevant:]
        return data

    def get_times(self, data):
        
        hashtag_d_indices = {}
        hashtag_d_keys = []
        for i, x in enumerate(data):
            if x and x[0] == '#':
                hashtag_d_indices[int(x[1:])] = i
                hashtag_d_keys.append(int(x[1:]))
        return hashtag_d_indices, hashtag_d_keys

    def setup_time_periods(self, start_time_point, time_granularity, time_period):
        
        start_t_periods = list(range(start_time_point, start_time_point + time_granularity * (time_period + 1), time_granularity))
        final_t_periods = []
        for i in range(len(start_t_periods) - 1):
            final_t_periods.append(start_t_periods[i: i + 2])
        return final_t_periods

    def get_time_positions(self, time_indices, time_periods, time_keys):
        
        adjusted = []
        for cnt, time_period in enumerate(time_periods):
            start_t, end_t = time_period
            if not start_t in time_indices:
                nearest_after_i = [i for i, x in enumerate(time_keys) if x > start_t][0]
                start_t = time_keys[nearest_after_i]

            begin = time_indices[start_t]
            if not end_t in time_indices:
                nearest_after_i = [i for i, x in enumerate(time_keys) if x > end_t][0]
                end_t = time_keys[nearest_after_i]
            end = time_indices[end_t]
            adjusted.append([begin, end])
        return adjusted

    def get_next_item(self, item, lst):
        
        all_next = [x for x in lst if x > item]
        return all_next[0]

    def replace_range_itm(self, the_list, time_period):
        for x in the_list:
            temp_lst = []
            if '[' in x[0]:
                name, ranges = x[0].split('[')
                range_end, range_start = ranges.split(']')[0].split(':')
                final_range = list(range(int(range_start), int(range_end) + 1))[::-1]
                for r in final_range:
                    this_val = ['{}[{}]'.format(name, r)]
                    this_val.extend([[] for _ in range(time_period)])
                    temp_lst.append(this_val)
                bus_init = []
                for t_stamp_i, t_stamp in enumerate(x[1:]):
                    if t_stamp:
                        t_stamp_divide = list([t_stamp[128 * i:128 * (i + 1)] for i in range(len(t_stamp) // (int(range_end) - int(range_start) + 1))])
                        if bus_init:
                            t_stamp_divide.insert(0, t_stamp)
                        elif len(t_stamp_divide) == 1:
                            bus_init = t_stamp[0]
                            continue
                        t_stamp_divide_divide_by2 = list([t_stamp_divide[i:(i + 2)] for i in range(len(t_stamp_divide) - 1)])
                        different_indices = []
                        for item in t_stamp_divide_divide_by2:
                            for i, (item1, item2) in enumerate(zip(item[0], item[1])):
                                if item1 != item2:
                                    different_indices.append([i, item2])
                        for each_indice in different_indices:
                            temp_lst[each_indice[0]][t_stamp_i + 1].append(each_indice[1])
                        bus_init = t_stamp[-1]
                yield temp_lst
            else:
                yield x

    def main(self, switchfile_path, relevant_module, start_time_point, time_granularity, time_period, pin_switch_pkl,module_symbol_idx, big_lst):

        pin_vals_to_ignore = ['x', 'z']
        time_granularity = time_granularity * 1000

        data = self.get_vcd_data(switchfile_path) 
        time_indices, time_keys = self.get_times(data)

        time_periods = self.setup_time_periods(start_time_point, time_granularity, time_period)

        time_positions = self.get_time_positions(time_indices, time_periods, time_keys)

        for cnt, time_chunk in enumerate(time_positions):
            begin, end = time_chunk
            for val in data[begin + 1:end]:
                if val:
                    if val[0] == 'b':
                        all_bits, symbol = val.split(' ')
                        if symbol in module_symbol_idx:
                            symbol_i = module_symbol_idx[symbol]
                            all_bits = [x for x in all_bits[1:]]
                            for pin_val in all_bits:
                                if pin_val not in pin_vals_to_ignore:
                                    big_lst[symbol_i][cnt + 2].append(pin_val)
                                else:
                                    big_lst[symbol_i][cnt + 2].append('')
                    elif val[0] != '#':
                        pin_val, symbol = val[0], val[1:]
                        if pin_val != pin_vals_to_ignore and symbol in module_symbol_idx:
                            symbol_i = module_symbol_idx[symbol]
                            big_lst[symbol_i][cnt + 2].append(pin_val)

        big_lst = [x[1:] for x in big_lst]
        big_lst = list(self.replace_range_itm(big_lst, time_period))
        big_dict = {}
        for x in big_lst:
            if isinstance(x[0], str):
                big_dict[x[0]] = x[1:]
            else:
                for y in x:
                    if y:
                        big_dict[y[0]] = y[1:]

        pin_value = {}
        for key, value in big_dict.items():
            sublist_counts = []
            for sublist in value:
                count_zeros = sublist.count('0')
                count_ones = sublist.count('1')
                sublist_counts.append([count_zeros + count_ones])
            pin_value[key] = sublist_counts

        with open(pin_switch_pkl + '.pkl', 'wb') as f:
            pickle.dump(pin_value, f)

    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--header_file_path', default='./vcd_files/header.txt')
        parser.add_argument('--module_name', default='uut')
        parser.add_argument('--start_time_point', default=0)
        parser.add_argument('--time_granularity', default=1)
        parser.add_argument('--time_period', default=160)
        parser.add_argument('--pin_switch_pkl', default='./pin_switch/pin_switch')
        parser.add_argument('--stimuli_num', type=int, default=1000, help="Number of VCD files to process")
        parser.add_argument('--skip_split', action='store_true', help="Skip Step 1 (splitting the large VCD).")
        args = parser.parse_args()
        header_file_path = args.header_file_path
        module_name = args.module_name
        start_time_point = args.start_time_point
        time_granularity = args.time_granularity
        time_period = args.time_period
        pin_switch_pkl = args.pin_switch_pkl
        stimuli_num = args.stimuli_num

        os.makedirs(os.path.dirname(pin_switch_pkl), exist_ok=True)
        start_time = time.time()

        relevant_module = self.get_relevant_module(header_file_path, module_name)

        module_symbol_idx, big_lst = self.setup(relevant_module, time_period) 

        # processes = []
        # for i in range(stimuli_num):
        #     switchfile_vcd_path = f'vcd_files/tb_main_{i}.vcd'
        #     pin_switch_pkl_i = f'{pin_switch_pkl}_{i}'
        #     new_process = Process(target= self.main, args =(switchfile_vcd_path, relevant_module, start_time_point, time_granularity, time_period, pin_switch_pkl_i,module_symbol_idx, big_lst,))
        #
        #     processes.append(new_process)

        for batch_start in range(0, stimuli_num, 20):
            batch_end = min(batch_start + 20, stimuli_num)
            processes = []

            for i in range(batch_start, batch_end):
                switchfile_vcd_path = f'vcd_files/tb_main_{i}.vcd'
                pin_switch_pkl_i = f'{pin_switch_pkl}_{i}'
                new_process = Process(target=self.main, args=(
                    switchfile_vcd_path, relevant_module, start_time_point, time_granularity, time_period,
                    pin_switch_pkl_i, module_symbol_idx, big_lst,
                ))
                processes.append(new_process)
            for process in processes:
                process.start()
            for process in processes:
                process.join()

        # for process in processes:
        #     process.start()
        # for process in processes:
        #     process.join()
        print("VCD Parser Running time:", time.time() - start_time, "seconds")
###############################################################################
# STEP 3: Summarizing pin_switch PKLs -> pin_switch_mean.npy
###############################################################################
class SummarizePinSwitch:
    """
    Step 3:
      1. Read each .pkl file (pin_switch_<i>.pkl).
      2. Convert toggles to arrays (CPU or GPU).
      3. Compute average (mean) across signals.
      4. Collect into one final 2D array and save as .npy (shape = (#files, #time_slices)).
    """

    def __init__(self, pin_switch_folder, pin_switch_prefix, start_index, end_index, output_npy):
        self.pin_switch_folder = pin_switch_folder
        self.pin_switch_prefix = pin_switch_prefix
        self.start_index = start_index
        self.end_index = end_index
        self.output_npy = output_npy

        os.makedirs(os.path.dirname(self.output_npy), exist_ok=True)

    def run(self):
        pin_switch_sum_list = []

        print("Summarizing toggle data from .pkl files...")
        for i in tqdm(range(self.start_index, self.end_index), desc="Summarizing toggles", unit="files"):
            pin_switch_file = f'{self.pin_switch_prefix}_{i}.pkl'
            pin_switch_path = os.path.join(self.pin_switch_folder, pin_switch_file)

            if not os.path.exists(pin_switch_path):
                # skip missing .pkl
                continue

            with open(pin_switch_path, 'rb') as pkl_file:
                pin_switch = pickle.load(pkl_file)

            arrays = []
            for value in pin_switch.values():
                arr_np = np.array(value).flatten()  
                arrays.append(arr_np)

            arr_gpu = cp.array(arrays, dtype=cp.float32)  # shape (num_signals, time_period)
            mean_gpu = cp.mean(arr_gpu, axis=0)
            mean_gpu = cp.expand_dims(mean_gpu, axis=0)

            pin_switch_sum_list.append(cp.asnumpy(mean_gpu))

        if not pin_switch_sum_list:
            print("No .pkl files found to summarize. Exiting.")
            return

        pin_switch_sum_array = np.vstack(pin_switch_sum_list)
        np.save(self.output_npy, pin_switch_sum_array)
        print(f"Saved pin_switch_mean data to {self.output_npy}.")


###############################################################################
# Combined main function (optional script entrypoint)
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="GPU-Optimized VCD Processing (Single Process)")

    # Step 1
    parser.add_argument("--vcd_init_path", type=str, default="./frontdata/SwitchFile_train.vcd")
    parser.add_argument("--vcd_final_path", type=str, default="vcd_files/tb_main_")
    parser.add_argument("--header_path", type=str, default="vcd_files/header.txt")
    parser.add_argument("--start_time_point", type=int, default=4560000)
    parser.add_argument("--num_plaintexts", type=int, default=1000)
    parser.add_argument("--desired_time_interval", type=int, default=160000)
    parser.add_argument("--off_time_interval", type=int, default=6880000)

    # Step 2
    parser.add_argument('--header_file_path', default='./vcd_files/header.txt')
    parser.add_argument('--module_name', default='uut')
    parser.add_argument('--time_granularity', type=int, default=1)
    parser.add_argument('--time_period', type=int, default=160)
    parser.add_argument('--pin_switch_pkl', default='./pin_switch/pin_switch')
    parser.add_argument('--stimuli_num', type=int, default=1000)

    # Step 3
    parser.add_argument('--pin_switch_folder', default='./pin_switch/')
    parser.add_argument('--output_npy', default='./pin_switch_mean_train.npy')
    parser.add_argument('--pin_switch_prefix', default='pin_switch')
    parser.add_argument('--pkl_start_index', type=int, default=0)
    parser.add_argument('--pkl_end_index', type=int, default=1000)

    # Step-skipping
    parser.add_argument('--skip_split', action='store_true',
                        help="Skip Step 1 (splitting the large VCD).")
    parser.add_argument('--skip_parse', action='store_true',
                        help="Skip Step 2 (parsing splitted VCDs).")
    parser.add_argument('--skip_post_process', action='store_true',
                        help="Skip Step 3 (post-processing PKL).")

    args = parser.parse_args()
    overall_start = time.time()

    # --- Step 1 ---
    if not args.skip_split:
        splitter = ProcessVCDInMemory(
            vcd_init_path=args.vcd_init_path,
            vcd_final_path=args.vcd_final_path,
            header_path=args.header_path,
            start_time_point=args.start_time_point,
            num_plaintexts=args.num_plaintexts,
            desired_time_interval=args.desired_time_interval,
            off_time_interval=args.off_time_interval
        )
        step1_start = time.time()
        splitter.run()
        step1_end = time.time()
        print(f"--- Step 1 completed in {step1_end - step1_start:.2f}s")
    else:
        print("Skipping Step 1 (Split VCD)...")

    # --- Step 2 ---
    if not args.skip_parse:
        step2_start = time.time()
        print("\n=== Step 2: Parsing splitted VCD files ===")
        ParserVCD()  
        step2_end = time.time()
        print(f"--- Step 2 completed in {step2_end - step2_start:.2f}s")
    else:
        print("Skipping Step 2 (Parse splitted VCD)...")

    # --- Step 3 ---
    if not args.skip_post_process:
        step3_start = time.time()
        summarizer = SummarizePinSwitch(
            pin_switch_folder=args.pin_switch_folder,
            pin_switch_prefix=args.pin_switch_prefix,
            start_index=args.pkl_start_index,
            end_index=args.pkl_end_index,
            output_npy=args.output_npy
        )
        summarizer.run()
        step3_end = time.time()
        print(f"--- Step 3 completed in {step3_end - step3_start:.2f}s")
    else:
        print("Skipping Step 3 (Post-processing PKL)...")

    overall_end = time.time()
    print(f"\n--- Overall time: {overall_end - overall_start:.2f}s")


if __name__ == "__main__":
    main()
