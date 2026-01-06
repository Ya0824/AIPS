import os
import re
import time
import _pickle as pickle
import argparse
import shutil

def mkdir(path):
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)

def Del_line(file_path):
    with open(file_path, "r") as f:
        res = f.readlines()
    res = [x for x in res if x.split()]
    with open(file_path, "w") as f:
        f.write("".join(res))
    return

def main():
    parser = argparse.ArgumentParser(description="Extract connection relationships from netlist files")
    parser.add_argument('--input', type=str, default=os.path.join('.', 'frontdata', 'aes_top.v'), help="Netlist to be processed")
    parser.add_argument('--result_folder', type=str, default=os.path.join('.'))
    parser.add_argument('--temp_folder', type=str, default=None)
    parser.add_argument('--output_pins', type=str, default="Q,QN,Y,CO,S", help="output pin names of device")
    parser.add_argument('--exist_global_logic1', type=int, default=1, help="If there is a global logic 1 in the circuit, set it to 1")
    parser.add_argument('--exist_global_logic0', type=int, default=0, help="If there is a global logic 0 in the circuit, set it to 0")
    parser.add_argument('--clk_rst', type=str, default="clk_48Mhz,reset_p", help="Clock and reset signal name")
    args = parser.parse_args()


    if args.temp_folder is None:
        args.temp_folder = os.path.join(args.result_folder, 'temp_file')

    output_pin = [pin.strip() for pin in args.output_pins.split(',') if pin.strip()]
    clk_rst = [sig.strip() for sig in args.clk_rst.split(',') if sig.strip()]

    start = time.perf_counter_ns()

    origin_netlist_file_path = args.input
    result_folder_path = args.result_folder
    temp_folder_path = args.temp_folder

    mkdir(result_folder_path)
    mkdir(temp_folder_path)

    for file_name in os.listdir(temp_folder_path):
        if file_name in ['use_netlist.v', 'use_netlist2.v']:
            os.remove(os.path.join(temp_folder_path, file_name))

    copy_netlist_path = os.path.join(temp_folder_path, 'use_netlist.v')
    shutil.copyfile(origin_netlist_file_path, copy_netlist_path)

    Del_line(copy_netlist_path)

    use_netlist2_path = os.path.join(temp_folder_path, 'use_netlist2.v')
    with open(copy_netlist_path, 'r') as file1, open(use_netlist2_path, 'w') as file2:
        for file_line in file1.readlines():
            file_line = file_line.lstrip(' ')
            if len(file_line) < 2:
                continue
            if file_line[-2] != ';':
                file_line = file_line.replace("\n", "")
            if 'module' not in file_line:
                file2.write(file_line)

    # Start to extract the connection relationships from netlist
    input_pin_list = []
    output_pin_list = []
    # Create a dictionary from pin to cell，format：[ net_name: [ [cell(inptut) ], [cell(output) ] ] ]
    from_pin_to_inst = {}
    device_list = []
    # Create a dictionary from cell to pin，format：[ [ cell_name, cell_type: [ [pin: net, ...], [...] ] ]
    from_inst_to_all = []

    if args.exist_global_logic1:
        from_pin_to_inst["1'b1"] = [[], []]
    if args.exist_global_logic0:
        from_pin_to_inst["1'b0"] = [[], []]

    with open(use_netlist2_path, 'r') as file3:
        for file3_line in file3.readlines():
            first_word = re.split(r'\s+', file3_line, 1)[0]
            if first_word == 'wire':
                if ':' in file3_line:
                    line_clean = file3_line.replace(' ', '').replace("\n", "").replace(";", "")
                    element_varia = re.split(r'[\[:\];]', line_clean)
                    temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
                    temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
                    for width in range(temp_min_width, temp_max_width + 1):
                        net_name = element_varia[3] + '[' + str(width) + ']'
                        from_pin_to_inst[net_name] = [[], []]
                else:
                    line_clean = file3_line.replace("\n", "").replace(";", "")
                    parts = re.split(r'\s+', line_clean, 1)
                    if len(parts) > 1:
                        nets = parts[1].replace(' ', '')
                        for net in nets.split(','):
                            if net:
                                from_pin_to_inst[net] = [[], []]
            elif first_word == 'input':
                if ':' in file3_line:
                    line_clean = file3_line.replace(' ', '').replace("\n", "").replace(";", "")
                    element_varia = re.split(r'[\[:\];]', line_clean)
                    temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
                    temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
                    for width in range(temp_min_width, temp_max_width + 1):
                        net_name = element_varia[3] + '[' + str(width) + ']'
                        input_pin_list.append(net_name)
                        from_pin_to_inst[net_name] = [[], []]
                else:
                    line_clean = file3_line.replace("\n", "").replace(";", "")
                    parts = re.split(r'\s+', line_clean, 1)
                    if len(parts) > 1:
                        nets = parts[1].replace(' ', '')
                        for net in nets.split(','):
                            if net:
                                input_pin_list.append(net)
                                from_pin_to_inst[net] = [[], []]
            elif first_word == 'output':
                if ':' in file3_line:
                    line_clean = file3_line.replace(' ', '').replace("\n", "").replace(";", "")
                    element_varia = re.split(r'[\[:\];]', line_clean)
                    temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
                    temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
                    for width in range(temp_min_width, temp_max_width + 1):
                        net_name = element_varia[3] + '[' + str(width) + ']'
                        output_pin_list.append(net_name)
                        from_pin_to_inst[net_name] = [[], []]
                else:
                    line_clean = file3_line.replace("\n", "").replace(";", "")
                    parts = re.split(r'\s+', line_clean, 1)
                    if len(parts) > 1:
                        nets = parts[1].replace(' ', '')
                        for net in nets.split(','):
                            if net:
                                output_pin_list.append(net)
                                from_pin_to_inst[net] = [[], []]
            else:
                parts = re.split(r'\s+', file3_line, 1)
                if len(parts) < 2:
                    continue
                device_name = parts[0]
                device_list.append(device_name)
                net_line = parts[1].replace(" ", "")
                net_line = net_line.rstrip('));\n')
                inst_split = re.split(r'[(]', net_line, 1)
                inst_name = inst_split[0]
                import_to_all = [inst_name, device_name]
                pin_and_net_list = re.split(r'[()]', inst_split[1])
                pin_and_net_dict = {}
                for i in range(len(pin_and_net_list) // 2):
                    if pin_and_net_list[2 * i + 1]:
                        temp_name = pin_and_net_list[2 * i].lstrip('.').lstrip(',.')
                        pin_and_net_dict[temp_name] = pin_and_net_list[2 * i + 1]
                import_to_all.append(pin_and_net_dict)
                from_inst_to_all.append(import_to_all)

    device_list = list(set(device_list))

    for inst_info in from_inst_to_all:
        temp_dict = inst_info[2]
        for pin_key in temp_dict.keys():
            if pin_key in output_pin:
                from_pin_to_inst[temp_dict[pin_key]][1] = [inst_info[0], pin_key]
            else:
                from_pin_to_inst[temp_dict[pin_key]][0].append([inst_info[0], pin_key])

    extract_from_inst_to_all = {}
    for inst_info in from_inst_to_all:
        input_dict = {}
        output_dict = {}
        for pin_key in inst_info[2].keys():
            if pin_key in output_pin:
                output_dict[pin_key] = inst_info[2][pin_key]
            else:
                input_dict[pin_key] = inst_info[2][pin_key]
        extract_from_inst_to_all[inst_info[0]] = [inst_info[1], [input_dict, output_dict]]


    cell2pin_pkl = os.path.join(result_folder_path, 'cell2pin.pkl')
    with open(cell2pin_pkl, 'wb') as f:
        pickle.dump(extract_from_inst_to_all, f, -1)


    print('Finish！')

    end = time.perf_counter_ns()
    len_run_time = end - start

    print('Number of nodes in the netlist:', len(from_pin_to_inst))
    print('Number of cells in the netlist:', len(from_inst_to_all))
    print('Running time: {} s'.format(len_run_time // 1000000000))

if __name__ == "__main__":
    main()
