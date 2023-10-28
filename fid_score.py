import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math

from utils import *

def calculate_fid_score(args=args):
    # load model
    _, _, trainer = load_model(args=args)  # remember to set args.calculate_fid=True!

    # fid score
    trainer.cal_fid()


if __name__ == '__main__':
    calculate_fid_score(args)
