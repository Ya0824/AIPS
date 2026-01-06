#!/usr/bin/env python3
"""
Standalone script to perform correlation power analysis (CPA).
"""

import scared
import numpy as np
# from scared import aes
import matplotlib.pyplot as plt


# Attack Settings

# Decide Real Key in hexadecimal format.
key = '0123456789abcdef123456789abcdef0'
# Decide file path of Plaintext Data.
plain_file = './StimuliFile_test.txt'
# Decide file path of Power Trace.
trace = np.load('../2-Diffusion Model/power_trace_test_pre.npy')
# Decide Trace Number of targeted power data.
trace_num = 1000


def read_txt(file,trace_num):
    """ read and process plaintext data from StimuliFile.txt """
    with open(file, 'r') as f:
        datas = f.read().splitlines()
    datas = datas[0:trace_num]
    data = []
    for j in datas:
        for i in range(0,len(j), 2):
            data.append(int(j[i:i+2], 16))
    data = np.array(data).reshape(-1, 16)
    
    data = data[range(1, 2*trace_num, 2)]
    return data

# process plaintext data from StimuliFile.txt
plaintext = read_txt(plain_file, trace_num)
plaintext = plaintext.astype('uint8')

# convert real key from hexadecimal format to decimal format
real_key = []
for i in range(0, len(key), 2):
    real_key.append(int(key[i:i + 2], 16))
# construct matrix of real key (traces num x bytes num)  
real_key = np.array([real_key for i in range(trace_num)], dtype='uint8')
# convert above data into scared readable format (numpy.ndarray → ets)
ths = scared.traces.formats.read_ths_from_ram(samples = trace, plaintext = plaintext, key = real_key)

# Leakage Model: HammingWeight model
# Attack Point: KeyAdd operation before first round
@scared.attack_selection_function
def first_add_key(plaintext, guesses):
    res = np.empty((plaintext.shape[0], len(guesses), plaintext.shape[1]), dtype='uint8')
    for i, guess in enumerate(guesses):
        res[:, i, :] = np.bitwise_xor(plaintext, guess)
    # the shape of res array is trace_num x 256 x time_point
    return res

# Leakage Model: HammingWeight model
# Attack Point: SubBytes operation at first round
@scared.attack_selection_function
def first_bytes(guesses, plaintext):
    out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
    for guess in guesses:
        out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))
    return out.swapaxes(0, 1)

# Leakage Model: HammingDistance model
# Attack Point: SubBytes operation at first round
@scared.attack_selection_function
def first_bytes_hd(guesses, plaintext):
    res = np.empty((plaintext.shape[0], len(guesses), 16), dtype='uint8')
    for guess in guesses:
        subbyte_input = np.bitwise_xor(plaintext, guess)
        subbyte_out = scared.aes.sub_bytes(subbyte_input)
        res[:, guess, :] = np.bitwise_xor(subbyte_input,subbyte_out)
    return res

# Create an analysis CPA
cpa_attack = scared.CPAAttack(
        selection_function=first_bytes_hd,
        model=scared.HammingWeight(),
        discriminant=scared.maxabs)

# create a container including required data
container = scared.Container(ths)
# run CPA attack using object container and cpa_attack
cpa_attack.run(container)
# obtain the guessed key from cpa attack
cpa_first_round_key = np.argmax(cpa_attack.scores, axis=0)
# check the cpa attack result
isright = []
for index, gk in enumerate(cpa_first_round_key):
    if(ths[0].key[index] == gk):
        isright.append(1)
    else:
        isright.append(0)
correctbyte = [i for i, x in enumerate(isright) if x == 1]

print('# ========================================================================== #')
print('#    Guess result: {}'.format(cpa_first_round_key))
print('#    Right result: {}'.format(ths[0].key))
print('#    The CPA attack recover {} key bytes from this cipher module'.format(sum(isright)))
print('#    The following key bytes are recovered: {}'.format(correctbyte))
print('# ========================================================================== #\n')

for byte in range(16):
    correct_tag = ths[0].key[byte-1]

    fig = plt.figure(figsize=(14, 9))
    for current_num in range(cpa_attack.results.shape[0]):
        if current_num == 0:
            plt.plot(cpa_attack.results[current_num, byte-1, :].T, linewidth=1, color='darkgray', label='Incorrect Keys')
        else:
            plt.plot(cpa_attack.results[current_num, byte-1, :].T, linewidth=1, color='darkgray')
    plt.plot(cpa_attack.results[correct_tag, byte-1, :].T, linewidth=2, color='red', label='Correct Key')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel("Correlation", fontsize=30)
    plt.xlabel("Time Points", fontsize=30)
    plt.legend(loc='best', fontsize=25, frameon=False)
    plt.show()
    plt.savefig('CpaAttackByte{}.png'.format(str(byte)))
