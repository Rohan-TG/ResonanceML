import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"

import hyperopt.early_stop
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pickle
from sklearn.metrics import mean_absolute_error
import keras
import numpy as np
import pandas as pd











def build_model(hp):
	pass