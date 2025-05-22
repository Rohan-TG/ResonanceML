import os

os.environ["OMP_NUM_THREADS"] = "30"
os.environ["MKL_NUM_THREADS"] = "30"
os.environ["OPENBLAS_NUM_THREADS"] = "30"
os.environ["TF_NUM_INTEROP_THREADS"] = "30"
os.environ["TF_NUM_INTRAOP_THREADS"] = "30"

import pandas as pd
import keras
import datetime
import numpy as np
import random
import periodictable
import itertools
import scipy
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import mean_absolute_error