import sys
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe')
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/experiments')
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/samuel.varga/python_packages/WoF_post')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
#for any prefix Wof_post, I have to remove the WoF_Post prefix for some reason

import pandas as pd
from ml_2to6_data_pipeline import (GridPointExtracter,
                                   subsampler, 
                                   load_dataset, random_subsampler)

from wofs_ml_severe.common.emailer import Emailer

from os.path import join , exists

from skexplain.common.multiprocessing_utils import run_parallel, to_iterator

import os
from glob import glob 
import numpy.random as npr #Used for date selection


""" usage: stdbuf -oL python -u run_2to6hr_ml_data_pipeline.py  2 > & log_2to6hr_data_pipeline & """
""" usage: stdbuf -oL python -u run_2to6hr_ml_data_pipeline.py > log_2to6hr_data_pipeline.txt 2>&1 &"""

##############
##############
##User Input##
##############
##############



#####################################
##Framework and Time Scale Settings##
#####################################
FRAMEWORK=['POTVIN','ADAM'] #Framework to use when creating the dataset. Valid options: POTVIN or ADAM
TIMESCALE='0to3' #Forecast windows to use when creating the data set. Valid Options: 0to3 or 2to6
n_jobs=5 #Number of jobs for parallel processing

################################
##Input and Output Directories##
################################
OUT_PATH_BASE = '/work/samuel.varga/data/{}_hr_severe_wx'.format(TIMESCALE) #Output directory
SUMMARY_FILE_OUT_PATH = '/work/samuel.varga/data/{}_hr_severe_wx'.format(TIMESCALE) #Output directory for Summary files
base_path = '/work/mflora/SummaryFiles' #Directory of WOFS ENS. Files


print('Using Sam\'s version of the data pipeline')
print('Framework: {}'.format(FRAMEWORK))
print('Time scale: {}'.format(TIMESCALE))


###################
###################
##Workflow script##
###################
###################

def worker(path, FRAMEWORK=FRAMEWORK, TIMESCALE=TIMESCALE):
    print(path)
    
    X_env, X_strm, ncfile, ll_grid  = load_dataset(path, TIMESCALE=TIMESCALE) #Load the files for the time scale
    inds=None #set to none
    
    for framework in FRAMEWORK:
        #print(ncfile)
        extracter = GridPointExtracter(ncfile, env_vars=X_env.keys(), strm_vars=X_strm.keys(), ll_grid=ll_grid, TIMESCALE=TIMESCALE, FRAMEWORK=framework) #Def GPE-- pass timescale and framework through
        df = extracter(X_env, X_strm) #Apply GPE to the env and storm

        #ys = [f for f in df.columns if 'severe' in f]
        #y_df = df[ys].sum(axis='columns')

        # Sampling all grid points with an event, but only 15% of 
        # grid points with no events. 
        #inds = subsampler(y_df, pos_percent=1.0, neg_percent=1.0) #Loken et. didn't resample, so use 1

        if inds is None: #Inds will be none on the first call. For the second framework, inds will already be assigned
            inds = random_subsampler(len(df), percent=0.3)
        
        df_sub = df.iloc[inds, :] #Selects subset based on inds-- will choose the same indices for both frameworks
        df_sub.reset_index(drop=True, inplace=True)

        out_path = path.replace(base_path, join(SUMMARY_FILE_OUT_PATH, f'{framework}/SummaryFiles')) #replace the base path with the output path
        if not exists(out_path):
            os.makedirs(out_path)

        out_name = join(out_path, 'wofs_ML{}.feather'.format(TIMESCALE.upper()))
        print(f'Saving {out_name}...')
        df_sub.to_feather(out_name)
        
    
    return None

emailer = Emailer()

start_time = emailer.get_start_time()

dates = [d for d in os.listdir(base_path) if '.txt' not in d]

##################################
##Changes for making big dataset##
##################################
#valInit=['2000','2100','2200','2300','0000','0100','0200','0300'] #List of init times to keep
#dates=[d for d in dates if d[4:6]=='05'] #Removes all Months other than May
#dates=[d for d in dates if d[0:4]!='2017'] #Removes 2017
#print(f'Number of cases in May: {len(dates)}')
#randState=npr.RandomState(42) #Set random state for reproducibility
#dates=randState.choice(dates, 40, replace=False) #Randomly choose 40 days in May



##########################
##Get Paths of ENS Files##
##########################


paths = [] #list of valid paths for worker function
for d in dates:
    if d[4:6] != '05': #Skips all months other than May-- should we change this?
        continue
    
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t] #initialization time
    #times = [t for t in times if t in valInit] #only keeps init times between 22-03
    
    for t in times: #For every init time on that day
        path = join(base_path,d,t)
        if TIMESCALE=='0to3':
            files = glob(join(path, f'wofs_ENS_[0-3]*')) #For 0-200 minutes into forecast, gets changed to 0-180 in get_files
        elif TIMESCALE=='2to6':    
            files = glob(join(path, f'wofs_ENS_[2-7]*')) #For 100-360 minutes into the forecast- gets changed to 120-360 in get_files
        
        all_nc_files = [f for f in files if f.endswith('.nc')] #list of every ENS file that ends with nc for that init time
        
        if len(all_nc_files) == len(files):
            if TIMESCALE=='2to6' and len(files) == 53: #If files are available for all time steps btwn 20-72:
               #if not exists(join(path,'wofs_ML2TO6.feather')):
                paths.append(path) #If all ENS files are nc files, append the path to the active list
            elif TIMESCALE=='0to3' and len(files)==40: #If files are available for all timesteps between 0-40:
                paths.append(path)

############################                
##Create the Daily Dataset##
############################

print(f'Number of paths : {len(paths)}')
emailer.send_email(f'Starting process for wofs_ML{TIMESCALE}', start_time)

run_parallel(
                func = worker,
                n_jobs = n_jobs,
                args_iterator = to_iterator(paths),
                )

emailer.send_email(f'Individual dataframes for the {TIMESCALE} hr dataset are complete', start_time)

#################################
##Create the ML and BL datasets##
#################################
for framework in FRAMEWORK:
    OUT_PATH = join(OUT_PATH_BASE, f'{framework}') #Output directory
    SUMMARY_FILE_OUT_PATH = '/work/samuel.varga/data/{}_hr_severe_wx/{}/SummaryFiles'.format(TIMESCALE, framework) 

    ml_files = []
    for d in dates:
        if d[4:6] != '05':
            continue

        times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t]
        #times = [t for t in times if t in valInit] #only keeps init times between 22-03

        for t in times:
            path = join(SUMMARY_FILE_OUT_PATH,d,t)
            filename = join(path,f'wofs_ML{TIMESCALE.upper()}.feather') #Make a list of the individual ML frames for each day
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

    emailer.send_email(f'The {TIMESCALE} hr {framework} ML and BL datasets are built and ready to go!', start_time)
