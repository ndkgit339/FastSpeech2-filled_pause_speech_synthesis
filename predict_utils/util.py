import torch
import numpy as np

def convert_str_to_time_ms(s: str):
    t = int(s.split(":")[0])*60*60*1000 + int(s.split(":")[1])*60*1000 + float(s.split(":")[2])*1000
    return int(t)

def floor_num(n, k=5):
    """
    入力を，ある値の整数倍に丸める
    """
    return (n // k) * k

