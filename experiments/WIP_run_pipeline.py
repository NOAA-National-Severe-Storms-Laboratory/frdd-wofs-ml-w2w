import sys
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe')
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/experiments')
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/monte.flora/python_packages/WoF_post')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
#for any prefix Wof_post, I have to remove the WoF_Post prefix for some reason

import pandas as pd
from WIP_pipeline import (GridPointExtracter,
                                   subsampler, 
                                   load_dataset)

from wofs_ml_severe.common.emailer import Emailer

from os.path import join , exists

from skexplain.common.multiprocessing_utils import run_parallel, to_iterator

import os
from glob import glob 


""" usage: stdbuf -oL python -u run_2to6hr_ml_data_pipeline.py  2 > & log_2to6hr_data_pipeline & """
""" usage: stdbuf -oL python -u run_2to6hr_ml_data_pipeline.py > log_2to6hr_data_pipeline.txt 2>&1 &"""

n_jobs = 7 #7

#####################################
##Framework and Time Scale Settings##
#####################################
FRAMEWORK='POTVIN' #Framework to use when creating the dataset. Valid options: POTVIN or ADAM
TIMESCALE='0to3' #Forecast windows to use when creating the data set. Valid Options: 0to3 or 2to6

################################
##Input and Output Directories##
################################
OUT_PATH = '/work/samuel.varga/data/{}_hr_severe_wx/{}'.format(TIMESCALE, FRAMEWORK) #Output directory
SUMMARY_FILE_OUT_PATH = '/work/samuel.varga/data/{}_hr_severe_wx/{}/SummaryFiles'.format(TIMESCALE, FRAMEWORK) #Output directory for Summary files
base_path = '/work/mflora/SummaryFiles' #Directory of WOFS ENS. Files


print('Using Sam\'s version of the data pipeline')
print('Framework: {}'.format(FRAMEWORK))
print('Time scale: {}'.format(TIMESCALE))

###################
##Workflow script##
###################

def worker(path, FRAMEWORK=FRAMEWORK, TIMESCALE=TIMESCALE):
    print(path)
    X_env, X_strm, ncfile, ll_grid  = load_dataset(path, TIMESCALE=TIMESCALE) #Load the files for the time scale
    #print(ncfile)
    extracter = GridPointExtracter(ncfile, env_vars=X_env.keys(), strm_vars=X_strm.keys(), ll_grid=ll_grid, TIMESCALE=TIMESCALE, FRAMEWORK=FRAMEWORK) #Def GPE-- pass timescale and framework through
    df = extracter(X_env, X_strm) #Apply GPE to the env and storm

    ys = [f for f in df.columns if 'severe' in f]
    y_df = df[ys].sum(axis='columns')

    # Sampling all grid points with an event, but only 15% of 
    # grid points with no events. 
    inds = subsampler(y_df, pos_percent=1.0, neg_percent=0.15)

    df_sub = df.iloc[inds, :]
    df_sub.reset_index(drop=True, inplace=True)

    path = path.replace(base_path, SUMMARY_FILE_OUT_PATH) #replace the base path with the output path
    if not exists(path):
        os.makedirs(path)
       
    out_name = join(path, 'wofs_ML{}.feather'.format(TIMESCALE.upper()))
    print(f'Saving {out_name}...')
    df_sub.to_feather(out_name)
    
    return None

emailer = Emailer()

start_time = emailer.get_start_time()

dates = [d for d in os.listdir(base_path) if '.txt' not in d]


##########################
##Get Paths of ENS files##
##########################

paths = [] #list of valid paths for worker function
for d in dates:
    if d[4:6] != '05': #Skips all months other than May
        continue
    
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t] #initialization time
    for t in times: #For every init time on that day
        path = join(base_path,d,t)
        if TIMESCALE=='0to3':
            files = glob(join(path, f'wofs_ENS_[0-3]*')) #For 0-200 minutes into the forecast- gets changed to 0-180 in get_files
        elif TIMESCALE=='2to6':
            files = glob(join(path, f'wofs_ENS_[2-7]*')) #For 100-360 minutes into the forecast- gets changed to 120-360 in get_files
        all_nc_files = [f for f in files if f.endswith('.nc')] #list of every ENS file that ends with nc for that init time
        
        if len(all_nc_files) == len(files):
            if TIMESCALE=='2to6' and len(files) == 53: #If files are available for all time steps btwn 20-72:
               #if not exists(join(path,'wofs_ML2TO6.feather')):
                paths.append(path) #If all ENS files are nc files, append the path to the active list
            elif TIMESCALE=='0to3' and len(files)==40: #If files are available for all time steps between 0-40
                paths.append(path)

#############################
##Create the Daily Datasets##
#############################
print(f'Number of paths : {len(paths)}')
emailer.send_email('Starting process for wofs_ML2to6', start_time)

run_parallel(
                func = worker,
                n_jobs = n_jobs,
                args_iterator = to_iterator(paths),
              
                )

emailer.send_email('Individual dataframes for the 2-6 hr dataset are complete', start_time)

# Create the ML and BL datasets
ml_files = []
for d in dates:
    if d[4:6] != '05':
        continue
        
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t]
    for t in times:
        path = join(SUMMARY_FILE_OUT_PATH,d,t)
        filename = join(path,'wofs_ML{}.feather'.format(TIMESCALE.upper())) #Make a list of the individual ML frames for each day
        if exists(filename):
            ml_files.append(filename)
    
dfs = [pd.read_feather(f) for f in ml_files]
        
df = pd.concat(dfs) #Concatenates all daily DFs

METADATA = ['Run Date', 'Init Time']

baseline_features = [f for f in df.columns if 'nmep' in f] #neighborhood estimation
targets = [f for f in df.columns if 'severe' in f] #Storm reports

baseline_df = df[baseline_features+METADATA+targets].reset_index(drop=True) 
features = [f for f in df.columns if f not in baseline_features] 

ml_df = df[features].reset_index(drop=True)  

baseline_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{TIMESCALE}hr__baseline_data.feather'))
ml_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{TIMESCALE}hr__data.feather'))

emailer.send_email('The 2-6 hr ML and BL datasets are built and ready to go!', start_time)
