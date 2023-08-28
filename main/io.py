# Scripts for loading the ML and baseline dataset. 
import pandas as pd
from os.path import join

# Load the data in a scikit-learn-ready input. 
def load_ml_data(base_path, target_col=None, date = None, mode=None, bl_column=None, FRAMEWORK=None, TIMESCALE=None, appendUH=False, Three_km=False, return_targets=False, full_9km=False):
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
            if TIMESCALE:
                ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__{date}_data{"_DBRS" if Three_km else ""}{"_target" if return_targets else ""}{"_full" if full_9km else ""}.feather'))
            else:
                ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__{date}_data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
        else:
            ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
    
    elif TIMESCALE:
        ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__{mode}_data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
    else:
        ml_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__{mode}_data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
      
    # The two columns are additional metadata. They tell us the run date and 
    # initialization for a given example.
    metadata = ['Run Date', 'Init Time','NX','NY']
    
    if date is None or return_targets:
        # All the target columns will have "severe" in them.
        targets = [f for f in ml_df.columns if 'severe' in f]
        # The features we will be using for training. 
        features = [f for f in ml_df.columns if f not in targets+metadata]
    else:
        nmep_vars = [f for f in ml_df.columns if 'nmep' in f]
        features = [f for f in ml_df.columns if f not in nmep_vars+metadata]
    if FRAMEWORK=='ADAM':
        print('Appending init time to predictors')
        features.append('Init Time')
        
    X = ml_df[features]
    
    if FRAMEWORK=='POTVIN' and appendUH:
        print('Appending UH')
        temp = pd.read_feather(join(f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/ADAM', f'wofs_ml_severe__{TIMESCALE}hr__{mode}_data.feather'))
        cols = [col for col in temp.columns if 'uh_2to5_instant__time_max__9km__smoothed_' in col]
        X[cols]=temp[cols]
        
    if date is None:
        y = ml_df[target_col]
        return X, y, ml_df[metadata]
    elif return_targets:
        y=ml_df[target_col]
        X_bl=ml_df[bl_column]
        
        return X, X_bl, y
    else:
        X_bl = ml_df[bl_column] if FRAMEWORK=='POTVIN' else None
        return X, X_bl, ml_df[metadata] 


# Load the baseline data into a scikit-learn ready input. 
def load_bl_data(base_path, target_col, mode, feature_col=None, TIMESCALE=None, Big=False, Three_km=False, full_9km=False):
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
    if TIMESCALE:
        if Big:
            bl_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__baseline_{mode}_data_Big.feather'))
        else:
            bl_df = pd.read_feather(join(base_path, f'wofs_ml_severe__{TIMESCALE}hr__baseline_{mode}_data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
    
    else:
        bl_df = pd.read_feather(join(base_path, f'wofs_ml_severe__2to6hr__baseline_{mode}_data{"_DBRS" if Three_km else ""}{"_full" if full_9km else ""}.feather'))
    
    y = bl_df[target_col]
    dates = bl_df['Run Date'].apply(str)
    
    return bl_df, y, dates 
