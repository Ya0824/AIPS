from statistics import median
import re
import _pickle as pickle
import time

def find_vals(nl, key, strict=True, return_key=False):
    '''gets values from a provided nested list of 2 elements; [key, value] '''
    if strict:
        result = [x for x in nl if x[0] == key]  # will return values if the list key exactly matches the input string
    else:
        result = [x for x in nl if key in x[0]]  # will return value if the input string is in the list key
    if result:
        if return_key:
            return result
        else:
            return [x[1] for x in result]
    else:
        return []


def find_full_key(nl, partial_key):
    '''searches for full key from a partial key input string. nested list of [key, value]. Example: '''
    return [x[0] for x in nl if partial_key in x[0]]


def inside_parenthesis(text):
    # gets text inside the first pair of parenthesis
    return text.split('(')[1].split(')')[0]


def parse_values_str(values_str):
    # converts a power values string into list of floats
    values_str = inside_parenthesis(values_str).replace('\\', '').replace('\"', '').split(', ')
    return [float(x.strip()) for x in values_str]


def middle_val(values_str):
    # gets the median power value after parsing the input values string
    parsed_vals = parse_values_str(values_str) # parses the string into a list of floats
    return median(parsed_vals) # uses the median function from a built-in library


def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z


def get_dict(data_list):
    # need to run parse(...) function first to get data_list input
    cell_data = {}
    all_cells = find_vals(data_list, 'cell', strict=False, return_key=True)  # find all the cells
    #print(all_cells)

    for cell_name, cell_vals in all_cells:
        in_cell = {}
        cell_name = inside_parenthesis(cell_name)

        # make a nested dictionary for the cell
        cell_data[cell_name] = in_cell

        # find all the pins in the cell
        num_internal = {}
        all_pins = find_vals(cell_vals, 'pin', strict=False, return_key=True)
        #print(all_pins)

        # fill the dictionary with pins information
        for pin_name, pin_data in all_pins:
            pin_name = inside_parenthesis(pin_name)
            in_cell[pin_name] = {
                'direction': find_vals(pin_data, 'direction')[0],
                'capacitance': float(find_vals(pin_data, 'capacitance')[0]),
            }  # add each of the pins to the cell with direction and capacitance values
            #default_internal_power_val = 0.0
            #empty_internal_power_vals = {'internal_power' + str(i): default_internal_power_val for i in range(1, 9)} # sets default internal power values for the pins
            #in_cell[pin_name] = merge_two_dicts(in_cell[pin_name], empty_internal_power_vals)
            internal_power_results = find_vals(pin_data, 'internal_power', strict=False)
            for internal_power in internal_power_results:
                related_pin = find_vals(internal_power, 'related_pin')
                if related_pin:
                    related_pin = related_pin[0].replace('\"', '')  # find all related_pin values in internal_power
                else:
                    related_pin = pin_name  # otherwise, the related pin is set as the current pin

                power_related = find_vals(internal_power, '_power',
                                          strict=False)  # find all power values (rise_power, fall_power) in internal power
                for x in power_related:
                    vals = find_full_key(x, 'values')  # find the values data
                    if vals:
                        if related_pin not in num_internal:
                            num_internal[related_pin] = 1
                        else:
                            num_internal[related_pin] += 1
                        #  counts the number of internal_power results for each pin so far
                        in_cell[related_pin]['internal_power' + str(num_internal[related_pin])] = middle_val(vals[0]) # adds the next power value result
                        # get the median from the values list; go to the middle_val function for more
        # find all the cell_leakage_power in the cell
        all_leakage = find_vals(cell_vals, 'cell_leakage_power', strict=False, return_key=True)
        # print(all_leakage)
        # add cell_leakage_power
        in_cell['cell_leakage_power'] = all_leakage


    # input = 0 ouput = 1
    for cell in cell_data:
        for pin in cell_data[cell]:
            if pin != 'cell_leakage_power':
                # gets the values from the dict
                cell_data[cell][pin] = list(cell_data[cell][pin].values())
                # replaces direction with number
                if cell_data[cell][pin][0] == 'input':
                    cell_data[cell][pin][0] = 0
                elif cell_data[cell][pin][0] == 'output':
                    cell_data[cell][pin][0] = 1
            else:
                if cell_data[cell][pin]:
                    cell_data[cell][pin] = cell_data[cell][pin][0][1]
                else:
                    cell_data[cell][pin] = '0.0'

    return cell_data


def parse(data):
    '''parses the lib file by using special characters such as {, }, :, and ; to figure out data relationships'''
    data = [*data]  # converst text to characters
    prev_str = ''
    all_d = []
    current_d = all_d  # binds current_d to all_d, another empty list
    old_current = []
    for c in data:
        if c == '{':
            new_d = []
            current_d.append([prev_str.strip(), new_d])  # appends the new values
            old_current.append(current_d)  # saves the current data layer in a list of old data layers
            current_d = new_d  # binds current_d to the new, inner layer list we just defined
            prev_str = ''
        elif c == ':':
            prev_str = prev_str.strip()
            current_d.append([prev_str, ''])  # add a key to the current data layer
            prev_str = ''
        elif c == ';':
            prev_str = prev_str.strip()
            if current_d and current_d[-1][1] == '':  # if a key was added
                current_d[-1][1] = prev_str  # add its value
            else:
                current_d.append([prev_str,
                                  ';'])  # otherwise, if there was a semicolon but no colon, add the key with a default value (not '')
            prev_str = ''
        elif c == '}':
            if old_current:  # of the list of old layers has not been depleted

                current_d = old_current[
                    -1]  # set the current data layer we're editing to be higher up (like [here] instead of [[here]])
                old_current.pop(-1)  # get rid of the last layer
            prev_str = ''
        else:
            prev_str += c  # add the character to the string
    #print(all_d)
    result = get_dict(all_d)  # using these values, get the final dictionary
    return result

def Process_lib_first():
    start_regex = r'^cell \([a-zA-Z0-9]+\) {'   #start regular expression
    stop_regex = r'cell_leakage_power'          #stop  regular expression
    lib_file = 'fast.lib'
    txt_file = 'fast_lib_extract.txt'
    # Open the lib file in read mode
    with open(lib_file, 'r') as lib:
        extracting = False                      # Initialize a flag to track extraction mode
        extracted_lines = []                    # Initialize an empty list to store the extracted lines

        for line in lib:                        # Iterate over each line in the lib file
            if re.match(start_regex, line):     # Check if the line matches the start_regex
                extracting = True               # Start extraction mode
            if extracting:                      # Check if extraction mode is active
                extracted_lines.append(line)    # Add the line to the extracted lines list
            if re.match(stop_regex, line):      # Check if the line matches the stop_regex
                extracting = False              # Stop extraction mode

    final_extracted_lines = [line for line in extracted_lines if '*' not in line]
    with open(txt_file, 'w') as txt:
        txt.writelines(final_extracted_lines)
    return txt_file


def extract_power(parsed_file):
    extract_power = {}
    for key, sub_dict in parsed_file.items():
        extract_power[key] = [float(sub_dict['cell_leakage_power'])]
        for sub_key, values in sub_dict.items():
            if sub_key != 'cell_leakage_power':
                if len(values) > 2:
                    extract_power[key].extend(values[2:])
    return extract_power


start_time = time.time()


file_data = open(Process_lib_first(), 'r').read()
# parses the file into the requested dictionary
parsed_data = parse(file_data)
#print(parsed_data)
extract_power = extract_power(parsed_data)
#print(extract_power)
# 将字典逐行存储为文本文件
with open('parsed_lib.txt', 'w') as file:
    for key, value in parsed_data.items():
        file.write(f'{key}: {value}\n')

# 将字典存储为 pickle 文件
with open('parsed_lib.pkl', 'wb') as file:
    pickle.dump(parsed_data, file)

with open("cell_power.pkl", "wb") as file:
    pickle.dump(extract_power, file)

with open('cell_power.txt', 'w') as f:
    for key, value in extract_power.items():
        f.write(f'{key}: {value}\n')

end_time = time.time()
print(end_time-start_time)