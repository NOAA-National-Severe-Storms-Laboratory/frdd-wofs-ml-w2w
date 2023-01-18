# Scripts for loading the ML and baseline dataset. 
import pandas as pd
from os.path import join

# Load the data in a scikit-learn-ready input. 
def load_ml_data(base_path, target_col=None, date = None, mode=None, bl_column=None, FRAMEWORK=None, TIMESCALE=None):
    """Load the ML dataframe into a X,y-ready scikit-learn input
    Parameters
    ---------------
    base_path : path-like str
        Path to where the training or testing dataset is stored. 
    
    target_col : str 
        The name of the target column 
    
    mode : 'train', 'test', or None
        Indicating whether to load the training or testing dataset.
        If None, then the original, unsplit dataset is loaded. 
    
    Returns
    ---------------
    X, y, metadata 
    
    metadata contains informatoin like the run dates and initialization times. 
    Run dates is used for cross-validation. 
    """
    # Load the feather file.
    if mode is None:
        if date is not None:
            if FRAMEWORK and TIMESCALE:
                ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__{date}_data.feather'))
            else:
                ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__{date}_data.feather'))
        else:
            ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__data.feather'))
    
    elif FRAMEWORK and TIMESCALE:
        ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__{mode}_data.feather'))
    
    else:
        ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__{mode}_data.feather'))
      
    # The two columns are additional metadata. They tell us the run date and 
    # initialization for a given example.
    metadata = ['Run Date', 'Init Time']
    
    if date is None:
        # All the target columns will have "severe" in them.
        targets = [f for f in ml_df.columns if 'severe' in f]
        # The features we will be using for training. 
        features = [f for f in ml_df.columns if f not in targets+metadata]
    else:
        nmep_vars = [f for f in ml_df.columns if 'nmep' in f]
        features = [f for f in ml_df.columns if f not in nmep_vars+metadata]
    
    X = ml_df[features]
    
    if date is None:
        y = ml_df[target_col]
        return X, y, ml_df[metadata]
    else:
        X_bl = ml_df[bl_column]
        return X, X_bl 


# Load the baseline data into a scikit-learn ready input. 
def load_bl_data(base_path, target_col, mode, feature_col=None, FRAMEWORK=None, TIMESCALE=None):
    """
    Load the baseline dataset.
    
    Parameters
    ----------------
    base_path : path-like str
        Path to where the training or testing dataset is stored.
    
    target_col : str 
        The name of the target column 
        
     mode : 'train', 'test', or None
        Indicating whether to load the training or testing dataset.
        If None, then the original, unsplit dataset is loaded.    
        
    feature_col : str (default=None)
        The name of the feature column. If None, then return the full dataframe. 
        Useful for training the baseline model. 

    """
    if TIMESCALE and FRAMEWORK:
        
        bl_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__baseline_{mode}_data.feather'))
    
    else:
        bl_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__baseline_{mode}_data.feather'))
    
    y = bl_df[target_col]
    dates = bl_df['Run Date'].apply(str)
    
    return bl_df, y, dates 