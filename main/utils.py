import sys
sys.path.append('/home/monte.flora/python_packages/ml_workflow')

from ml_workflow.io.cross_validation_generator import DateBasedCV
import random
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join, exists
import pandas as pd
import pickle

# Splitting the data into training and testing. 
def _train_test_split():
    """
    Randomly split the full dataset into training and testing 
    based on the date. 
    """
    FRAMEWORK=['POTVIN']
    TIMESCALE='2to6'
    
    if exists('/work/samuel.varga/data/dates_split.pkl'):
        date_pkl = pd.read_pickle('/work/samuel.varga/data/dates_split.pkl')
        print('Using previous T-T split')
        train_dates, test_dates = date_pkl['train_dates'], date_pkl['test_dates']
    else:
        train_dates, test_dates = None, None
    for framework in FRAMEWORK:
        basePath = f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/{framework}/' #Base path to data


        path = join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__data_full.feather')
        df = pd.read_feather(path)

        baseline_path = join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__baseline_data_full.feather')
        baseline_df = pd.read_feather(baseline_path)

        # Get the date from April, May, and June 
        df['Run Date'] = df['Run Date'].apply(str)
        baseline_df['Run Date'] = baseline_df['Run Date'].apply(str)

        # Limit data to the Spring/Summer 
        df = df[pd.to_datetime(df['Run Date']).dt.strftime('%B').isin(['March', 'April', 'May', 'June', 'July'])]
        df = df[pd.to_datetime(df['Run Date']).dt.strftime('%Y').isin(['2018','2019','2020','2021'])]
        
        baseline_df = baseline_df[
        pd.to_datetime(baseline_df['Run Date']).dt.strftime('%B').isin(['March', 'April', 'May', 'June', 'July'])]
        baseline_df = baseline_df[
        pd.to_datetime(baseline_df['Run Date']).dt.strftime('%Y').isin(['2018','2019','2020','2021'])]

        if train_dates is None and test_dates is None:
            all_dates = list(df['Run Date'].unique())
            random.Random(42).shuffle(all_dates)
            train_dates, test_dates = train_test_split(all_dates, test_size=0.3)
            print(test_dates)
            with open(f'/work/samuel.varga/data/dates_split.pkl', 'wb') as date_file:
                    pickle.dump({'train_dates':train_dates,'test_dates':test_dates}, date_file)

        train_df = df[df['Run Date'].isin(train_dates)] 
        test_df  = df[df['Run Date'].isin(test_dates)] 

        train_base_df = baseline_df[baseline_df['Run Date'].isin(train_dates)] 
        test_base_df  = baseline_df[baseline_df['Run Date'].isin(test_dates)] 

        print(f'{train_df.shape=}')
        print(f'{test_df.shape=}')

        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        train_base_df.reset_index(inplace=True, drop=True)
        test_base_df.reset_index(inplace=True, drop=True)
        
        
        ##3km only
        if False: #Simple Random Sampling
            np.random.seed(42)
            new_sample=np.random.choice(np.arange(0, len(train_df)), 2000000, replace=False) #Generate 2 million random indices
            train_df=train_df.loc[new_sample] #Select the same indices from both dataframes
            train_base_df = train_base_df.loc[new_sample]
            train_df.reset_index(inplace=True, drop=True)
            train_base_df.reset_index(inplace=True, drop=True)
        elif False: #Date-Based Random Sampling
            np.random.seed(42)
            train_out=pd.DataFrame() #subsampled ML data
            train_base_out=pd.DataFrame() #subsampled BL data 

            for day in train_dates: #For every day in the training data set
                
                #Create subset of df only consisting of data from this date
                train_sub_df = train_df[train_df['Run Date'] == day]
                train_base_sub_df= baseline_df[baseline_df['Run Date']==day]
                
                
                for init_time in list(train_sub_df['Init Time'].unique()): #For every init time on that date
                    
                    #Create subset of df only consisting of data from this date & init time
                    train_sub_init=train_sub_df[train_sub_df['Init Time']==init_time]
                    train_base_sub_init=train_base_sub_df[train_base_sub_df['Init Time']==init_time]
                    
                    #Reset indices to be [0, len(train_sub_init)]
                    train_sub_init.reset_index(inplace=True, drop=True) 
                    train_base_sub_init.reset_index(inplace=True, drop=True)
                    
                    #Sample n points
                    #Draw samples from each init time such that we end up with 2000000 points. 
                    #There are 430 combos of run date and init time in the training dataset
                    new_sample_inds=np.random.choice(np.arange(0,len(train_sub_init)), int(2000000/430), replace=False)
                    
                    #Append to data frame 
                    train_out=train_out.append(train_sub_init.loc[new_sample_inds])
                    train_base_out=train_base_out.append(train_base_sub_init.loc[new_sample_inds])
                    
            #Copy the subsampled data to train_df and baseline_df, then reset the index
            train_df=train_out
            print(len(train_df))
            baseline_df=train_base_out
            train_df.reset_index(inplace=True, drop=True)
            train_base_df.reset_index(inplace=True, drop=True)
        ##            
        
        
        train_df.to_feather(join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__train_data_full.feather'))
        test_df.to_feather(join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__test_data_full.feather'))

        train_base_df.to_feather(join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__baseline_train_data_full.feather'))
        test_base_df.to_feather(join(basePath, f'wofs_ml_severe__{TIMESCALE}hr__baseline_test_data_full.feather'))

# Execute the code. 
_train_test_split()
    
    
    