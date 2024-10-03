import pandas as pd
import numpy as np
import os
import periodictable

isotope = [27, 59]

master_directory = f'/Users/rntg/PycharmProjects/ResonanceML/fudge/broadening_tests/n+{periodictable.elements[isotope[0]]}{isotope[1]}'
master_heated_directory = f'{master_directory}/heated'

