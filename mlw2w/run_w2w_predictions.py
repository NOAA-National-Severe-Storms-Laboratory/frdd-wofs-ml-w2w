import sys
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe')
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/mlw2w')
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/samuel.varga/python_packages/WoF_post')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')


import pandas as pd
from ml_2to6_data_pipeline import (GridPointExtracter,
                                   subsampler, 
                                   get_files, random_subsampler)

from wofs_ml_severe.common.emailer import Emailer


from os.path import join , exists

from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
from wofs_ML_2to6_op_pipeline import wofs_ml_2to6

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
FRAMEWORK=['POTVIN'] #Framework to use when creating the dataset. Valid options: POTVIN or ADAM
TIMESCALE='2to6' #Forecast windows to use when creating the data set. Valid Options: 0to3 or 2to6
n_jobs=1 #Number of jobs for parallel processing

################################
##Input and Output Directories##
################################
OUT_PATH_BASE = f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/sfe_prep/SummaryFiles' #Output directory
base_path = '/work/mflora/SummaryFiles' #Directory of WOFS ENS. Files
ml_dir = '/work/samuel.varga/projects/2to6_hr_severe_wx/sfe_prep/mlModels/'
model_dics = [{'name':'hist','prefix':'sfe','train':'all','hazard':'all','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'all','hazard':'wind','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'all','hazard':'hail','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'all','hazard':'tornado','target':'36km','suffix':'control','severity':'Sev'},
                          {'name':'hist','prefix':'sfe','train':'45km','hazard':'wind','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'45km','hazard':'hail','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'45km','hazard':'tornado','target':'36km','suffix':'control','severity':'Sev'},
                          {'name':'hist','prefix':'sfe','train':'27km','hazard':'wind','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'27km','hazard':'hail','target':'36km','suffix':'control','severity':'Sev'},
             {'name':'hist','prefix':'sfe','train':'27km','hazard':'tornado','target':'36km','suffix':'control','severity':'Sev'}]


print('Using Sam\'s version of the data pipeline')
print(f'Framework: {FRAMEWORK}')
print(f'Time scale: {TIMESCALE}')


###################
###################
##Workflow script##
###################
###################

def worker(path, FRAMEWORK=FRAMEWORK, TIMESCALE=TIMESCALE):
    print(path)
   
    try:
        files = get_files(path, TIMESCALE='2to6')[0] #Load the files for the time scale
        outdir = join(OUT_PATH_BASE, '/'.join(files[0].split('/')[4:-1]))
        print(outdir)
        wofs_ml_2to6(files, ml_dir, outdir, model_dics=model_dics, verbose=0,
                     baseline_dir = None, save_predictors=False).run_pipeline() 
    except:
        print(f'Something went wrong for {files[0]}')
        files = get_files(path, TIMESCALE='2to6')[0]
        file_in = files[0].replace('mflora','samuel.varga/data/2to6_hr_severe_wx/sfe_prep')
        file_in = f"{'/'.join(file_in.split('/')[:-1])}/wofs_ML2TO6_full.feather"
        file_out =files[0].split('/')[-1].replace('ENS_24', 'ML2TO6')
        outdir = join(OUT_PATH_BASE, '/'.join(files[0].split('/')[4:-1]))
        print(outdir)
        try:
           
            df = pd.read_feather(file_in)
            metadata = ['Run Date', 'Init Time','NX','NY']
            targets = [f for f in df.columns if 'severe' in f]
            nmep = [f for f in df.columns if 'nmep' in f]
            features = [f for f in df.columns if f not in targets+metadata]
            df = df.drop(metadata, axis=1)
            df = df.drop(targets, axis=1)
            df = df.drop(nmep, axis=1)
            ml, bl = wofs_ml_2to6(files, ml_dir, outdir, model_dics=model_dics, verbose=0,
                     baseline_dir = None, save_predictors=False,
                                  out_file = file_out, **{'forecast_shape':(100,100)}).get_predictions(df, None)
            wofs_ml_2to6(files, ml_dir, outdir, model_dics=model_dics, verbose=0,
                     baseline_dir = None, save_predictors=False, out_file = file_out).save_ml_predictions(ml, {})
        except:
            print(f'Something went wrong again for {files[0]}')
    return None



test_dates = pd.read_pickle('/work/samuel.varga/data/dates_split_deep_learning.pkl')['test_dates'] 
dates=[d for d in os.listdir(base_path) if '.txt' not in d] 
gen_dates=[]
for d in dates:
    if d[:8] in test_dates:
        gen_dates.append(d)
print(gen_dates)

##########################
##Get Paths of ENS Files##
##########################


paths = [] #list of valid paths for worker function
for d in gen_dates:
    if d[4:6] != '05' or int(d[:4]) <=2018:   #<=2018:
        continue
    
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t] #initialization time
    #times = [t for t in times if t in valInit] #only keeps init times between 22-03
    
    for t in times: #For every init time on that day
        path = join(base_path,d,t)
        if TIMESCALE=='0to3':
            files = glob(join(path, f'wofs_{"ALL" if int(path.split("/")[4][:4]) >= 2021 else "ENS"}_[0-3]*.nc')) #For 0-200 minutes into forecast, gets changed to 0-180 in get_files
        elif TIMESCALE=='2to6':    
            files = glob(join(path, f'wofs_{"ALL" if int(path.split("/")[4][:4]) >= 2021 else "ENS"}_[2-7]*.nc')) #For 100-360 minutes into the forecast- gets changed to 120-360 in get_files
        
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

run_parallel(
                func = worker,
                n_jobs = n_jobs,
                args_iterator = to_iterator(paths),
                )