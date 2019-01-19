import subprocess
import sys
import os
import time

NR_THREAD = 10

"""
cmd = './libffm/ffm-predict ./data/train_ffm.txt ./model/model_ffm tr_ffm.out'.format(nr_thread=NR_THREAD)
cmd_res = os.popen(cmd).readlines()
for line in cmd_res:
    print(line)

cmd = './libffm/ffm-predict ./data/valid_ffm.txt ./model/model_ffm va_ffm.out'.format(nr_thread=NR_THREAD)
cmd_res = os.popen(cmd).readlines()
for line in cmd_res:
    print(line)
"""

cmd = './libffm/ffm-predict ./data/test_ffm.txt ./model/model_ffm te_ffm.out'.format(nr_thread=NR_THREAD)
cmd_res = os.popen(cmd).readlines()
for line in cmd_res:
    print(line)

