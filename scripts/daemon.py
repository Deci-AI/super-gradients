import sys
import time
from multiprocessing import Process

import psutil


def f():
    i = 0
    while True:
        i += 1
        mem = ' - '.join([f"{name}: {value}" for name, value in dict(psutil.virtual_memory()._asdict()).items()])
        print(mem)
        time.sleep(1)



process = Process(daemon=True, target=f)
process.start()

print('Ok')
time.sleep(3)
print('middle')
sys.exit(-2)
time.sleep(5)
print('end')
