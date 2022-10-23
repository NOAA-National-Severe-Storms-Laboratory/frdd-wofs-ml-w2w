import sys
sys.path.append('/home/monte.flora/python_packages/ml_workflow')

from ml_workflow.io.cross_validation_generator import DateBasedCV
import random
from sklearn.model_selection import train_test_split
from os.path import join
import pandas as pd

# Splitting the data into training and testing. 
def _train_test_split():
    """
    Randomly split the full dataset into training and testing 
    based on the date. 
    """
    basePath = '/work/samuel.varga/data/2to6_hr_severe_wx' #Base path to data
    
    path = join(basePath, f'wofs_ml_severe__2to6hr__data.feather')
    df = pd.read_feather(path)
    
    baseline_path = join(basePath, f'wofs_ml_severe__2to6hr__baseline_data.feather')
    baseline_df = pd.read_feather(baseline_path)
        
    # Get the date from April, May, and June 
    df['Run Date'] = df['Run Date'].apply(str)
    baseline_df['Run Date'] = baseline_df['Run Date'].apply(str)
        
    # Limit data to the Spring/Summer 
    df = df[pd.to_datetime(df['Run Date']).dt.strftime('%B').isin(['March', 'April', 'May', 'June', 'July'])]
    baseline_df = baseline_df[
    pd.to_datetime(baseline_df['Run Date']).dt.strftime('%B').isin(['March', 'April', 'May', 'June', 'July'])]
        
    all_dates = list(df['Run Date'].unique())
    random.shuffle(all_dates)
    train_dates, test_dates = train_test_split(all_dates, test_size=0.3)
    
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
        
    train_df.to_feather(join(basePath, f'wofs_ml_severe__2to6hr__train_data.feather'))
    test_df.to_feather(join(basePath, f'wofs_ml_severe__2to6hr__test_data.feather'))
        
    train_base_df.to_feather(join(basePath, f'wofs_ml_severe__2to6hr__baseline_train_data.feather'))
    test_base_df.to_feather(join(basePath, f'wofs_ml_severe__2to6hr__baseline_test_data.feather'))
    
# Execute the code. 
_train_test_split()
    
    
    