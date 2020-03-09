import csv
import pdb
import pandas as pd
from pathlib import Path

# Define paths
DIR_CURRENT = Path(__file__).parent
DATA_DIR = DIR_CURRENT.parents[0] / 'data' / 'raw' / 'avatar' / 'eeg'
D_NOT_SEP = DATA_DIR / 'eeg_raw_peri_only'
D_YES_SEP = DATA_DIR / 'peri_hinf_delta'

# Get file info
f_not_sep = [x for x in D_NOT_SEP.glob('*.csv') if x.is_file()]
f_yes_sep = [x for x in D_YES_SEP.glob('*.csv') if x.is_file()]

# Read one file
thisone = f_yes_sep[0]
print(f"Going to process: {thisone}")
df_yes = pd.read_csv(str(thisone))

pdb.set_trace()
