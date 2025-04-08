import re
import time
import _pickle as pickle
import argparse
from statistics import median


def find_vals(nl, key, strict=True, return_key=False):
    if strict:
        result = [x for x in nl if x[0] == key]
    else:
        result = [x for x in nl if key in x[0]]
    if result:
        if return_key:
            return result
        else:
            return [x[1] for x in result]
    else:
        return []


def find_full_key(nl, partial_key):
    return [x[0] for x in nl if partial_key in x[0]]


def inside_parenthesis(text):
    return text.split('(')[1].split(')')[0]


def parse_values_str(values_str):
    values_str = inside_parenthesis(values_str).replace('\\', '').replace('\"', '').split(', ')
    return [float(x.strip()) for x in values_str]


def middle_val(values_str):
    parsed_vals = parse_values_str(values_str)
    return median(parsed_vals)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def get_dict(data_list):
    cell_data = {}
    all_cells = find_vals(data_list, 'cell', strict=False, return_key=True)

    for cell_name, cell_vals in all_cells:
        in_cell = {}
        cell_name = inside_parenthesis(cell_name)
        cell_data[cell_name] = in_cell

        num_internal = {}
        all_pins = find_vals(cell_vals, 'pin', strict=False, return_key=True)

        for pin_name, pin_data in all_pins:
            pin_name = inside_parenthesis(pin_name)
            in_cell[pin_name] = {
                'direction': find_vals(pin_data, 'direction')[0],
                'capacitance': float(find_vals(pin_data, 'capacitance')[0]),
            }
            internal_power_results = find_vals(pin_data, 'internal_power', strict=False)
            for internal_power in internal_power_results:
                related_pin = find_vals(internal_power, 'related_pin')
                if related_pin:
                    related_pin = related_pin[0].replace('\"', '')
                else:
                    related_pin = pin_name

                power_related = find_vals(internal_power, '_power', strict=False)
                for x in power_related:
                    vals = find_full_key(x, 'values')
                    if vals:
                        if related_pin not in num_internal:
                            num_internal[related_pin] = 1
                        else:
                            num_internal[related_pin] += 1
                        in_cell[related_pin]['internal_power' + str(num_internal[related_pin])] = middle_val(vals[0])
        all_leakage = find_vals(cell_vals, 'cell_leakage_power', strict=False, return_key=True)
        in_cell['cell_leakage_power'] = all_leakage

    for cell in cell_data:
        for pin in cell_data[cell]:
            if pin != 'cell_leakage_power':
                cell_data[cell][pin] = list(cell_data[cell][pin].values())
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
    data = [*data]
    prev_str = ''
    all_d = []
    current_d = all_d
    old_current = []
    for c in data:
        if c == '{':
            new_d = []
            current_d.append([prev_str.strip(), new_d])
            old_current.append(current_d)
            current_d = new_d
            prev_str = ''
        elif c == ':':
            prev_str = prev_str.strip()
            current_d.append([prev_str, ''])
            prev_str = ''
        elif c == ';':
            prev_str = prev_str.strip()
            if current_d and current_d[-1][1] == '':
                current_d[-1][1] = prev_str
            else:
                current_d.append([prev_str, ';'])
            prev_str = ''
        elif c == '}':
            if old_current:
                current_d = old_current[-1]
                old_current.pop(-1)
            prev_str = ''
        else:
            prev_str += c
    result = get_dict(all_d)
    return result


def Process_lib_first(lib_file, start_regex, stop_regex):
    with open(lib_file, 'r') as lib:
        extracting = False
        extracted_lines = []
        for line in lib:
            if re.match(start_regex, line):
                extracting = True
            if extracting:
                extracted_lines.append(line)
            if re.match(stop_regex, line):
                extracting = False

    final_extracted_lines = [line for line in extracted_lines if '*' not in line]
    return "".join(final_extracted_lines)


def extract_power(parsed_file):
    extract_power_dict = {}
    for key, sub_dict in parsed_file.items():
        extract_power_dict[key] = [float(sub_dict['cell_leakage_power'])]
        for sub_key, values in sub_dict.items():
            if sub_key != 'cell_leakage_power':
                if len(values) > 2:
                    extract_power_dict[key].extend(values[2:])
    return extract_power_dict


def main():
    parser = argparse.ArgumentParser(description="Parses the lib file, extracts the cell and its power information, and outputs the results in pickle format")
    parser.add_argument('--lib_file', type=str, default="./frontdata/fast.lib", help="input lib file")
    parser.add_argument('--cell_power_pkl', type=str, default="cell_power.pkl")
    parser.add_argument('--start_regex', type=str, default=r'^cell \([a-zA-Z0-9]+\) {')
    parser.add_argument('--stop_regex', type=str, default=r'cell_leakage_power')
    args = parser.parse_args()

    start_time = time.time()


    file_data = Process_lib_first(args.lib_file, args.start_regex, args.stop_regex)

    parsed_data = parse(file_data)
    cell_power_data = extract_power(parsed_data)

    with open(args.cell_power_pkl, 'wb') as file:
        pickle.dump(cell_power_data, file)

    end_time = time.time()
    print("Running time：", end_time - start_time)


if __name__ == "__main__":
    main()
