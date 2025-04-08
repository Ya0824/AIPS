# -*- coding: UTF-8 -*-

import os
import re
import time
import _pickle as pickle


start = time.perf_counter_ns()


# Path of the file to be processed
origin_netlist_file_path = '.' + os.path.sep + 'frontdata' + os.path.sep + 'aes_top.v'

# Analyze Setup
# List of names of output pins in all cells
output_pin = ['Q', 'QN', 'Y', 'CO', 'S']    #only for smic18_base


# Sets whether global logic 1/global logic 0 exists in the circuit
exist_global_logic1 = 1
exist_global_logic0 = 0

# Define the clock signal and reset signal name
clk_rst = ['clk_48Mhz', 'reset_p']


# Create a data folder
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

result_folder_path = '.' + os.path.sep + '1-topology-analysis_result'
temp_folder_path = result_folder_path + os.path.sep + 'temp_file'
mkdir(result_folder_path)
mkdir(temp_folder_path)


# Process netlist file
files_list = os.listdir(temp_folder_path)
for file_name1 in files_list:
    if file_name1 == 'use_netlist.v':
        remove_txt_path = temp_folder_path + os.path.sep + r'use_netlist.v'
        os.remove(remove_txt_path)
    elif file_name1 == 'use_netlist2.v':
        remove_txt_path = temp_folder_path + os.path.sep + r'use_netlist2.v'
        os.remove(remove_txt_path)

# Copy the netlist to be processed as use_netlist.v
copy_netlist_path = temp_folder_path + os.path.sep + r'use_netlist.v'
copy_command = 'cp ' + origin_netlist_file_path + ' ' + copy_netlist_path
os.system(copy_command)  # 若是linux系统，不用copy用cp


def Del_line(file_path):
    f = open(file_path, "r")
    res = f.readlines()
    res = [x for x in res if x.split()]
    f.close()
    f = open(file_path, "w")
    f.write("".join(res))
    f.close()
    return

Del_line(temp_folder_path + os.path.sep + 'use_netlist.v')
file1 = open(temp_folder_path + os.path.sep + 'use_netlist.v', 'r')
file2 = open(temp_folder_path + os.path.sep + 'use_netlist2.v', 'w')
for file_line in file1.readlines():
    file_line = file_line.lstrip(' ')
    if file_line[-2] != ';':
        file_line = file_line.replace("\n", "")
    if bool(1 - ('module' in file_line)):
        file2.write(file_line)
file1.close()
file2.close()


# Extract connection relationships in the netlist by pin or instance
input_pin_list = []
output_pin_list = []
# 创建节点_器件pin字典 / 器件列表 / inst_all列表
from_pin_to_inst = {}  # { pin1:[ [[instance1, pin],[instance2, pin],...], [instance1, pin] ] }
device_list = []
from_inst_to_all = []  # [ [instance, device, {A:n1, B:n2, ...}], [...] ]

# 在节点列表导入全局逻辑1/0
if exist_global_logic0:
    from_pin_to_inst['1\'b1'] = [[], []]
if exist_global_logic0:
    from_pin_to_inst['1\'b0'] = [[], []]

# 向节点_器件pin字典导入所有节点 / 导出器件列表、inst_all列表
file3 = open(temp_folder_path + os.path.sep + 'use_netlist2.v', 'r')
for file3_line in file3.readlines():
    first_word = re.split(r'[ ]', file3_line, 1)[0]
    if first_word == 'wire':  # 向节点_器件pin字典导入所有节点
        if ':' in file3_line:
            file3_line = file3_line.replace(' ', '')
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            element_varia = re.split(r'[\[:\];]', file3_line)
            temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
            temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
            for width_varia in range(temp_min_width, temp_max_width + 1):
                net_name = element_varia[3] + '[' + str(width_varia) + ']' #若一个信号为 wire A [3:0],生成，A[0],A[1]...
                from_pin_to_inst[net_name] = [[], []]
        else:
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            file3_line_temp = re.split(r'[ ]', file3_line, 1)[1]
            file3_line_temp = file3_line_temp.replace(' ', '')
            list_net_const = re.split(r'[,]', file3_line_temp)
            for list_net_const1 in list_net_const:
                from_pin_to_inst[list_net_const1] = [[], []]
    elif first_word == 'input':
        if ':' in file3_line:
            file3_line = file3_line.replace(' ', '')
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            element_varia = re.split(r'[\[:\];]', file3_line)
            # print('分割结果为：', element_varia)
            temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
            temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
            for width_varia in range(temp_min_width, temp_max_width + 1):
                net_name = element_varia[3] + '[' + str(width_varia) + ']'
                input_pin_list.append(net_name)
                from_pin_to_inst[net_name] = [[], []]
        else:
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            file3_line_temp = re.split(r'[ ]', file3_line, 1)[1]
            file3_line_temp = file3_line_temp.replace(' ', '')
            list_net_const = re.split(r'[,]', file3_line_temp)
            for list_net_const1 in list_net_const:
                input_pin_list.append(list_net_const1)
                from_pin_to_inst[list_net_const1] = [[], []]
    elif first_word == 'output':
        if ':' in file3_line:
            file3_line = file3_line.replace(' ', '')
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            element_varia = re.split(r'[\[:\];]', file3_line)
            # print('分割结果为：', element_varia)
            temp_max_width = max(int(element_varia[1]), int(element_varia[2]))
            temp_min_width = min(int(element_varia[1]), int(element_varia[2]))
            for width_varia in range(temp_min_width, temp_max_width + 1):
                net_name = element_varia[3] + '[' + str(width_varia) + ']'
                output_pin_list.append(net_name)
                from_pin_to_inst[net_name] = [[], []]
        else:
            file3_line = file3_line.replace("\n", "")
            file3_line = file3_line.replace(";", "")
            file3_line_temp = re.split(r'[ ]', file3_line, 1)[1]
            file3_line_temp = file3_line_temp.replace(' ', '')
            list_net_const = re.split(r'[,]', file3_line_temp)
            for list_net_const1 in list_net_const:
                output_pin_list.append(list_net_const1)
                from_pin_to_inst[list_net_const1] = [[], []]
    else:  # 提取网表信息
        slice_device_and_net = re.split(r'[ ]', file3_line, 1)
        # 分离出device名称，并导出为所有用到的器件类型
        device_name = slice_device_and_net[0]
        device_list.append(device_name)
        # 分离出inst名称，并将device和inst名称放入import_to_all，等待导入inst_all列表
        net_name1 = slice_device_and_net[1].replace(" ", "")
        net_name2 = net_name1.rstrip('));\n')
        net_name3 = re.split(r'[(]', net_name2, 1)
        inst_name = net_name3[0]
        import_to_all = [inst_name, device_name]
        # 再分离出pin_and_net字典，并将其放入import_to_all，等待导入inst_all列表
        pin_and_net_list = re.split(r'[()]', net_name3[1])
        pin_and_net_dict = {}
        for neti in range(len(pin_and_net_list) // 2):
            if pin_and_net_list[2 * neti + 1]:
                temp_name = pin_and_net_list[2 * neti].lstrip('.')
                temp_name2 = temp_name.lstrip(',.')
                pin_and_net_dict[temp_name2] = pin_and_net_list[2 * neti + 1]
        import_to_all.append(pin_and_net_dict)
        # 最终将import_to_all导入inst_all列表
        from_inst_to_all.append(import_to_all)
# 将device_list列表中元素去重
device_list = list(set(device_list))
file3.close()

# 打印所有用到的器件类型
# print('电路使用的所有器件类型：', device_list)
# print(len(from_pin_to_inst.keys())) #11164
# 遍历from_inst_to_all中所有节点，并转化数据格式，存入from_pin_to_inst
for from_inst_to_all_i in from_inst_to_all:
    temp_dict = from_inst_to_all_i[2]
    for temp_dict_i in temp_dict.keys():
        if temp_dict_i in output_pin:  # output_pin设置所有器件的输出pin的名称
            from_pin_to_inst[temp_dict[temp_dict_i]][1] = [from_inst_to_all_i[0], temp_dict_i]
        else:
            from_pin_to_inst[temp_dict[temp_dict_i]][0].append([from_inst_to_all_i[0], temp_dict_i])


# 将from_inst_to_all转为字典，以inst为key
# 实际上此处可在生成from_inst_to_all时直接生成字典而不是生成列表然后在此处再转为字典，后续可以改一改
extract_from_inst_to_all = {}
for from_inst_to_all_i in from_inst_to_all:
    input_dict = {}
    output_dict = {}
    for temp_translatei in from_inst_to_all_i[2].keys():
        if temp_translatei in output_pin:
            output_dict[temp_translatei] = from_inst_to_all_i[2][temp_translatei]
        else:
            input_dict[temp_translatei] = from_inst_to_all_i[2][temp_translatei]
    temp_translate_list = [input_dict, output_dict]
    extract_from_inst_to_all[from_inst_to_all_i[0]] = [from_inst_to_all_i[1], temp_translate_list]


# 网表连接关系存储为文件####################################################################################################
file4 = open(result_folder_path + os.path.sep + 'instance2pin.txt', 'w')
for file_line4 in extract_from_inst_to_all:
    file4.write(str(file_line4))
    file4.write(' ')
    file4.write(str(extract_from_inst_to_all[file_line4]))
    file4.write('\n')
file4.close()


file5 = open(result_folder_path + os.path.sep + 'pin2instance.txt', 'w')
for file_line5 in from_pin_to_inst:
    file5.write(str(file_line5))
    file5.write(' ')
    file5.write(str(from_pin_to_inst[file_line5]))
    file5.write('\n')
file5.close()

pkl_from_inst_to_all_name = result_folder_path + os.path.sep + 'instance2pin.pkl'
pkl_from_inst_to_all = open(pkl_from_inst_to_all_name, 'wb')
pickle.dump(extract_from_inst_to_all, pkl_from_inst_to_all, -1)
pkl_from_inst_to_all.close()

pkl_from_pin_to_inst_name = result_folder_path + os.path.sep + 'pin2instance.pkl'
pkl_from_pin_to_inst = open(pkl_from_pin_to_inst_name, 'wb')
pickle.dump(from_pin_to_inst, pkl_from_pin_to_inst, -1)
pkl_from_pin_to_inst.close()

print('网表信息提取完成！')


end = time.perf_counter_ns()
len_run_time = end - start

print('网表中节点个数', len(from_pin_to_inst))
print('网表中门个数', len(from_inst_to_all))

print('运行时间: ', str(len_run_time//1000000000), 's')