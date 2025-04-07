import argparse
import re
import sys
import time
import pickle
from threading import Thread
from multiprocessing import Process


class ProcessVCD:

    def generate_header_file(self, header_path):

        pattern_time = r'^#\d+\n$'
        header_file = open(header_path, 'w', newline='\n')
        for line in self.vcd_init_data:
            match_time = re.match(pattern_time, line, flags=0)
            if match_time:
                match_line = match_time.group().strip()
                # print(match_line)
                current_time = int(match_line[1:])
                if current_time > 0:
                    break
                else:
                    header_file.write(line)
            else:
                header_file.write(line)
        header_file.close()

    def process_vcd_file(self, start_time_point, num_plaintexts, desired_time_interval, off_time_interval,
                         vcd_final_path):

        start = False
        pattern_time = r'^#\d+\n$'
        current_num = 0
        new_vcd_file = open(vcd_final_path + str(current_num) + '.vcd', 'w', newline='\n')
        left_time_interval = start_time_point + off_time_interval * current_num
        right_time_interval = start_time_point + desired_time_interval + off_time_interval * current_num

        for line in self.vcd_init_data:
            match_time = re.match(pattern_time, line, flags=0)
            if current_num == num_plaintexts:
                break
            elif match_time:
                match_line = match_time.group().strip()
                current_time = int(match_line[1:])
                if current_time > right_time_interval:
                    start = False
                    new_vcd_file.close()
                    current_num += 1
                    left_time_interval = start_time_point + off_time_interval * current_num
                    right_time_interval = start_time_point + desired_time_interval + off_time_interval * current_num
                    # print('Current loop', current_num, 'Time interval (', left_time_interval, ',', right_time_interval, ')')
                    new_vcd_file = open(vcd_final_path + str(current_num) + '.vcd', 'w', newline='\n')
                    if current_time == left_time_interval:
                        start = True
                        tmp_line = '#' + str(current_time - left_time_interval) + '\n'
                        new_vcd_file.write(tmp_line)
                        # Thread(target= lambda operator : new_vcd_file.write(tmp_line)).start()
                elif current_time >= left_time_interval:
                    start = True
                    tmp_line = '#' + str(current_time - left_time_interval) + '\n'
                    new_vcd_file.write(tmp_line)
                    # Thread(target= lambda operator : new_vcd_file.write(tmp_line)).start()
            elif start:
                new_vcd_file.write(line)
                # Thread(target= lambda operator : new_vcd_file.write(line)).start()

    def main(self, start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_final_path,
             header_path):
        threads = [None, None]
        threads[0] = Thread(target=self.generate_header_file, args=(header_path,))
        threads[1] = Thread(target=self.process_vcd_file, args=(
        start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_final_path,))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("Cut off vcd files, finished")

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--vcd_init_path", type=str, default="SwitchFile.vcd",
                            help="Path to the init vcd file, should end in .vcd")
        parser.add_argument("--vcd_final_path", type=str, default="vcd_files/tb_main_",
                            help="Path to the output vcd file")
        parser.add_argument("--header_path", type=str, default="vcd_files/header.txt",
                            help="Path to the header vcd file")
        parser.add_argument("--start_time_point", type=int, default=8000000,
                            help="Start time point for power analysis, timescale 1ns/1ps")
        parser.add_argument("--num_plaintexts", type=int, default=1000,
                            help="Amount of the required plaintexts")
        parser.add_argument("--desired_time_interval", type=int, default=160000,
                            help="Desired time slice for power analysis, timescale 1ns/1ps")
        parser.add_argument("--off_time_interval", type=int, default=6880000,
                            help="Time interval between two-times power analysis, timescale 1ns/1ps")

        args = parser.parse_args()
        vcd_init_path = args.vcd_init_path
        vcd_final_path = args.vcd_final_path
        header_path = args.header_path
        start_time_point = args.start_time_point
        num_plaintexts = args.num_plaintexts
        desired_time_interval = args.desired_time_interval
        off_time_interval = args.off_time_interval

        start_time = time.time()
        self.vcd_init_data = []
        with open(vcd_init_path, "r") as file:
            self.vcd_init_data = file.readlines()

        try:
            self.main(start_time_point, num_plaintexts, desired_time_interval, off_time_interval,
                      vcd_final_path, header_path)
        except KeyboardInterrupt:
            sys.exit()
        finally:
            print("VCD Process Running time:", time.time() - start_time, "seconds")


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

        start_t_periods = list(
            range(start_time_point, start_time_point + time_granularity * (time_period + 1), time_granularity))
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
                        t_stamp_divide = list([t_stamp[128 * i:128 * (i + 1)] for i in
                                               range(len(t_stamp) // (int(range_end) - int(range_start) + 1))])
                        if bus_init:
                            t_stamp_divide.insert(0, t_stamp)
                        elif len(t_stamp_divide) == 1:
                            bus_init = t_stamp[0]
                            continue
                        t_stamp_divide_divide_by2 = list(
                            [t_stamp_divide[i:(i + 2)] for i in range(len(t_stamp_divide) - 1)])
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

    def main(self, switchfile_path, relevant_module, start_time_point, time_granularity, time_period, pin_switch_pkl,
             module_symbol_idx, big_lst):

        pin_vals_to_ignore = ['x', 'z']
        time_granularity = time_granularity * 1000

        data = self.get_vcd_data(switchfile_path)  # 整个vcd_file的内容---vcd文件
        time_indices, time_keys = self.get_times(data)
        # print(time_indices) #时间点以及时间点所在的行数---vcd文件
        # print(time_keys) #time_period个时间点---vcd文件
        time_periods = self.setup_time_periods(start_time_point, time_granularity, time_period)
        # print(time_periods)  #time_period个时间点[0, 1000], [1000, 2000],...---vcd文件
        time_positions = self.get_time_positions(time_indices, time_periods, time_keys)
        # print(time_positions)  #time_period个时间点所在的行数[119583, 156563], [156563, 156832],...---vcd文件
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
        # with open(pin_value_pkl + '.pkl', 'wb') as f:
        # pickle.dump(big_dict, f)

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

        args = parser.parse_args()
        header_file_path = args.header_file_path
        module_name = args.module_name
        start_time_point = args.start_time_point
        time_granularity = args.time_granularity
        time_period = args.time_period
        pin_switch_pkl = args.pin_switch_pkl
        stimuli_num = args.stimuli_num
        # batch_size = 20 #设置了这个batch_size这样就可以避免内存溢出的问题

        start_time = time.time()

        relevant_module = self.get_relevant_module(header_file_path, module_name)
        # print(relevant_module) #uut中记录信号名称和信号索引的部分---header文件
        module_symbol_idx, big_lst = self.setup(relevant_module,
                                                time_period)  # I think this line of code can also be separated from the main function, as it is the processing of the header.txt file.
        # print(module_symbol_idx) #uut中记录信号名称和信号索引的部分---header文件
        # print(big_lst) #uut中每一个信号根据time_period做一个大列表用于后面存放文件---可由header文件产生

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
        print("VCD Parser Running time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
    # print('starting process')
    # print(f'{time.localtime()[4]}:{time.localtime()[5]}')
    # process_vcd = ProcessVCD()
    # print('starting parsering')
    # print(f'{time.localtime()[4]}:{time.localtime()[5]}')
    parser_vcd = ParserVCD()
    # print(f'{time.localtime()[4]}:{time.localtime()[5]}')

