import sys 
sys.path.append('/home/monte.flora/python_packages/wofs_ml_severe')
from wofs_ml_severe.data.ml_2to6_data_pipeline import (GridPointExtracter,
                                                       subsampler, 
                                                       load_dataset)

from wofs_ml_severe.common.util import Emailer

from os.path import join , exists
from WoF_post.wofs.post.multiprocessing_script import run_parallel, to_iterator
import os
from glob import glob
import pandas as pd 

""" usage: stdbuf -oL python -u run_2to6hr_ml_data_pipeline.py  2 > & log_2to6hr_data_pipeline & """

n_jobs = 30

# Workflow script. 
def worker(path):
    print(path)
    X_env, X_strm, ncfile  = load_dataset(path)
    #print(ncfile)
    extracter = GridPointExtracter(ncfile, env_vars=X_env.keys(), strm_vars=X_strm.keys())
    df = extracter(X_env, X_strm)

    ys = [f for f in df.columns if 'severe' in f]
    y_df = df[ys].sum(axis='columns')

    # Sampling all grid points with an event, but only 25% of 
    # grid points with no events. 
    inds = subsampler(y_df, pos_percent=1.0, neg_percent=0.15)

    df_sub = df.iloc[inds, :]
    df_sub.reset_index(drop=True, inplace=True)

    out_name = join(path, 'wofs_ML2TO6.feather')
    print(f'Saving {out_name}...')
    df_sub.to_feather(out_name)
    
    return None

emailer = Emailer()

start_time = emailer.get_start_time()

base_path = '/work/mflora/SummaryFiles'
dates = [d for d in os.listdir(base_path) if '.txt' not in d]

paths = []
for d in dates:
    if d[4:6] != '05':
        continue
    
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t]
    for t in times:
        path = join(base_path,d,t)
        files = glob(join(path, f'wofs_ENS_[2-7]*'))
        
        all_nc_files = [f for f in files if f.endswith('.nc')]
        
        if len(all_nc_files) == len(files):
            if len(files) == 53:
                #if not exists(join(path,'wofs_ML2TO6.feather')):
                paths.append(path)
     
print(f'Number of paths : {len(paths)}')
emailer.send_message('Starting process for wofs_ML2to6', start_time)

"""
run_parallel(
                func = worker,
                nprocs_to_use = n_jobs,
                iterator = to_iterator(paths),
                mode='mp',
                )
"""

emailer.send_message('Individual dataframes for the 2-6 hr dataset are complete', start_time)

# Create the ML and BL datasets
ml_files = []
for d in dates:
    if d[4:6] != '05':
        continue
        
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t]
    for t in times:
        path = join(base_path,d,t)
        filename = join(path,'wofs_ML2TO6.feather')
        if exists(filename):
            ml_files.append(filename)
    
dfs = [pd.read_feather(f) for f in ml_files]
        
df = pd.concat(dfs)

METADATA = ['Run Date', 'Init Time']

baseline_features = [f for f in df.columns if 'nmep' in f]
targets = [f for f in df.columns if 'severe' in f]

baseline_df = df[baseline_features+METADATA+targets].reset_index(drop=True) 
features = [f for f in df.columns if f not in baseline_features]

ml_df = df[features].reset_index(drop=True)  

out_path = join('/work/mflora/ML_2TO6HR', 'data') 

baseline_df.to_feather(join(out_path, f'wofs_ml_severe__2to6hr__baseline_data.feather'))
ml_df.to_feather(join(out_path, f'wofs_ml_severe__2to6hr__data.feather'))

emailer.send_message('The 2-6 hr ML and BL datasets are built and ready to go!', start_time)
