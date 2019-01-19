import subprocess
import sys
import os
import time


NR_THREAD = 10
cmd = './libffm/ffm-train --auto-stop -r 0.1 -t 32 -s {nr_thread} -p ./data/valid_ffm.txt ./data/train_ffm.txt ./model/model_ffm'.format(nr_thread=NR_THREAD)
cmd_res = os.popen(cmd).readlines()
print(cmd_res)

