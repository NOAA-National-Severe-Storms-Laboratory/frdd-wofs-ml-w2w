####################################################################################
#
# DATA PIPELINE SCRIPT FOR THE 2-6 HR WOFS-ML-SEVERE PRODUCTS
#
# Git Author: monte-flora (monte.flora@noaa.gov)
#PMMWIP Branch: svarga 
####################################################################################


#Varga version
#These are from the wofs_post package

from wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)

from wofs_ml_severe.data_pipeline.storm_report_loader import StormReportLoader

from wofs.plotting.util import decompose_file_path


from glob import glob
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter
from collections import ChainMap
import numpy as np
import xarray as xr
from os.path import join
import itertools
import pyresample 
import pandas as pd
import datetime as dt

# These are the list of variables that are pulled from the WoFS
# summary files. The list is informed by previous work (Flora et al. 2021, MWR) 

ml_config = { 'ENS_VARS':  ['uh_2to5_instant',
                            'uh_0to2_instant',
                            'wz_0to2_instant',
                            'comp_dz',
                            'ws_80',
                            'hailcast',
                            'w_up',
                            'okubo_weiss',
                    ],
             
              'ENV_VARS' : ['mid_level_lapse_rate', 
                            'low_level_lapse_rate', 
                           ],
             
              'SVR_VARS': ['shear_u_0to1', 
                        'shear_v_0to1', 
                        'shear_u_0to6', 
                        'shear_v_0to6',
                        'shear_u_3to6', 
                        'shear_v_3to6',
                        'srh_0to3',
                        'cape_ml', 
                        'cin_ml', 
                        'stp',
                        'scp',
                       ]
            }

def get_files(path, TIMESCALE):
    """Get the ENS, ENV, and SVR file paths for the 0-3 || 2-6 hr forecasts"""
    # Load summary files between time step 00-36 || 24-72. 
    if TIMESCALE=='0to3':
        ens_files = glob(join(path,'wofs_ENS_[0-3]*')) 
        ens_files.sort()
        ens_files = ens_files[:37] #Drops the last 4 files, so we have 0-36
    elif TIMESCALE=='2to6':
        ens_files = glob(join(path,'wofs_ENS_[2-7]*'))
        ens_files.sort()
        ens_files = ens_files[4:] #Drops the first 4 files, so we have 24-72 instead of 20-72
    
    svr_files = [f.replace('ENS', 'SVR') for f in ens_files]
    env_files = [f.replace('ENS', 'ENV') for f in ens_files]
    
    return ens_files, env_files, svr_files
    
def load_dataset(path, TIMESCALE):
    """Load the 0-3|| 2-6 hr forecasts"""
    ens_files, env_files, svr_files = get_files(path, TIMESCALE)
    
    coord_vars = ["xlat", "xlon", "hgt"]
    
    X_strm, coords, _, _  = load_multiple_nc_files(
                ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS'])

    X_env, _, _, _  = load_multiple_nc_files(
                env_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENV_VARS'])

    X_svr, _, _, _ = load_multiple_nc_files(
                svr_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['SVR_VARS'])

    X_env = {**X_env, **X_svr}

    X_env = {v : X_env[v][1] for v in X_env.keys()}
    X_strm = {v : X_strm[v][1] for v in X_strm.keys()}
    
    ll_grid = (coords['xlat'][1].values, coords['xlon'][1].values)
    
    return X_env, X_strm, ens_files[0], ll_grid

class GridPointExtracter:
    """Upscale X, compute time-composites, compute ensemble statistics.
    
    Attributes
    -------------------------
    
    ncfile : path-like 
        The summary file path for either the ENS, ENV, or SVR file at the beginning of the 
        4-hr period. Used for determining getting the correct reports for labelling. 
    
    env_vars : list of strs 
        The environmental variables. These variables are 4-hr time-averaged and 
        are upscaled using a spatial average filter. 
    
    strm_vars : list of strs
        The intra-storm variables. These variables are 4-hr time-maxed and 
        are upscaled using a spatial maximum filter. 
    
    ll_grid : 2-tuple of 2D arrays 
        original latitude and longitude grids prior to the resampling. 
    
    upscale_size : int (default = None)
        The initial upscaling radius (in grid points). This initial upscaling 
        coarsens the grid resolution to improve processing times. 
        
    forecast_sizes : list of ints
        The radius of the forecast upscalings (in grid points). Internally, the
        radius are converted to the diameters for the 
        scipy.ndimage.uniform_filter and scipy.ndimage.maximum_filter. 
        d = 2*r 
        
    target_sizes : list of ints
        The radius of the targets upscalings (in grid points). Internally, the
        radius are converted to the diameters for the scipy.ndimage.uniform_filter. 
        d = 2*r 
        
    grid_spacing : int 
        Grid spacing (in km) of the original grid. 
        
    TIMESCALE: string
        Forecast window of the data in hours after the init time
        
    FRAMEWORK: string
        Framework used for preprocessing the data. See respective papers for more detail.
    """
    def __init__(self, ncfile, env_vars, strm_vars, ll_grid, TIMESCALE, FRAMEWORK,
                 forecast_sizes=[1,3,5], #[1,3,5] #upscale_zie*grid_spacing*forecast size = predictor radius (km)
                 target_sizes = [1,2,4,6], #[1,2,4,6] #upscale_size*grid_spacing*target size = target radius (km)
                 upscale_size=None, #Scales from 3-> 9 km grid
                 grid_spacing=3, #WOFS grid spacing (km)
                 reports_path = '/work/mflora/LSRS/STORM_EVENTS_2017-2022.csv',
                 report_type = 'NOAA'
                ):
        
        #Change parameters based on Framework:
        if FRAMEWORK.upper()=='ADAM':
            self._upscale_size = 3 #3 #No upscaling of data -->  Change this to 1, for comparison, set to 3
            self._TARGET_SIZES = np.array([1, 2, 4]) * 2 #Converts radius to diameter, 3 km x 12,13 boxes == 36 km, 39 km target radius
            self._SIZES = np.array([1])*2 #Diameter of Gaussian Smoother only used for smoothed mean of storm fields
            #From Loken et.: SD is 18 km -> grid_spacing * forecast_size * 2 =18km -> forecast_size=3
            #With upscaling: SD of 18 km-> grid_spacing * upscale * forecast * 2 =18 -> f_size=1
            #But is that really valid when the initial data is 3x as coarse?
        else:
            self._upscale_size = 3 #3km smoothing of data before everything else #1
            self._TARGET_SIZES =  np.array(target_sizes)*2 #*3
            self._SIZES = np.array(forecast_sizes)*2 #*3
        
        
        #Constant Parameters
        self._n_ens = 18     
        self._env_vars = env_vars
        self._strm_vars = strm_vars      
        self._ncfile = ncfile
        self._DX = grid_spacing * self._upscale_size  
        self._reports_path = reports_path
        self._report_type = report_type
        self._TIMESCALE=TIMESCALE
        self._FRAMEWORK=FRAMEWORK
        self._deltat=5 #Time step in minutes
        
        
        if np.max(np.absolute(ll_grid[0]))>90:
            raise ValueError('Latitude values for ll_grid > 90 and are likely longitude values. Switch the input.')
        
        self._original_grid = ll_grid 

        self._target_grid = (ll_grid[0][::self._upscale_size, ::self._upscale_size], 
                             ll_grid[1][::self._upscale_size, ::self._upscale_size]
                            ) 

        # Baseline variables and their corresponding thresholds. 
        self._BASELINE_VARS = ['hailcast', 'uh_2to5_instant', 'ws_80']
        self._NMEP_THRESHS = {'hailcast' : [0.5, 0.75, 1.0, 1.25, 1.5], 
                             'uh_2to5_instant' : [50, 75, 100, 125, 150, 175, 200],
                             'ws_80' : [30, 40, 50, 60], 
                            }
        
    def __call__(self, X_env, X_strm, predict=False):
        
        # TODO: Pre-processor. Get rid of super high updraft speeds, replace NaNs, etc. 
         
        
        # This X has had a 3-grid point gaussian smoother applied to it. -- Identical to original fields when upscale_size==1
        X_env_upscaled = {v  : self.upscaler(X_env[v], 
                                     func=uniform_filter,
                                     size=self._upscale_size) for v in self._env_vars}
        
        # This X has had a 3-grid point maximum filter applied to it. (See above note)
        X_strm_upscaled = {v : self.upscaler(X_strm[v], 
                                     func=maximum_filter,
                                     size=self._upscale_size) for v in self._strm_vars}
        
        
        
        if self._FRAMEWORK=='POTVIN':
            #rint(f'{self._FRAMEWORK} Block')
            # For the environment, 
            # 1. Time-average 
            # 2. Spatial ensemble statistics at different scales. 
            X_env_time_comp = self.calc_time_composite(X_env_upscaled, 
                                                        func=np.nanmean, name='time_avg', keys=self._env_vars)


            X_env_stats = self.calc_spatial_ensemble_stats(X_env_time_comp, environ=True)

            # For the storm, 
            # 1. Time-maximum 
            # 2. Spatial ensemble statistics at different scales. 
            # 3. Amplitude Statistics
            X_strm_time_comp = self.calc_time_composite(X_strm_upscaled, 
                                                        func=np.nanmax, name='time_max', keys=self._strm_vars)

            X_strm_stats = self.calc_spatial_ensemble_stats(X_strm_time_comp, environ=False)
        elif self._FRAMEWORK=='ADAM':
        
            #For the environment,
            #1. Ensemble mean at each time index
            #2. Time-average of ensemble means 
            X_ens_stats=self.calc_spatial_ensemble_stats(X_env_upscaled, environ=True, FRAMEWORK=self._FRAMEWORK)
            X_env_stats=self.calc_time_composite(X_ens_stats, func=np.nanmean, name='time_avg', keys=X_ens_stats.keys())
            
            #For the storm,
            #1. Time-maximum 
            #2. Ensemble Statistics at each grid point
            X_strm_time_comp = self.calc_time_composite(X_strm_upscaled, 
                                                        func=np.nanmax, name='time_max', keys=self._strm_vars)
            
            X_strm_stats = self.calc_spatial_ensemble_stats(X_strm_time_comp, environ=False, FRAMEWORK=self._FRAMEWORK)
        
        
        X_all = {**X_strm_stats, **X_env_stats}
        
        if predict:
            data = X_all
        else:
            # IF not predicting, then get the target values. 
            y = self.get_targets(TIMESCALE=self._TIMESCALE)
            data = {**X_all, **y}

        
        # Stack the data and convert to dataframe. 
        data = {v : (['NY', 'NX'], data[v]) for v in data.keys()}
        ds = xr.Dataset(data)
        df = ds.stack(z=('NY', 'NX')).to_dataframe()
        
        ds.close()
        
        # Convert target variable to binary values. 
        # In the to_grid code, it gives each report a unique label
        # which is used for the object matching. 
        ys = [f for f in df.columns if 'severe' in f]
        
        new_df = df.copy()
        for y_var in ys:
            new_df[y_var] = np.where(df[y_var]>0, 1, 0)
        
        # Add date and init time
        comps = self._ncfile.split('/')
        date, init_time = comps[-3], comps[-2]
        new_df["Run Date"] = [date]*len(df)
        new_df["Init Time"] = [init_time]*len(df)
        
        return new_df
    
    def get_nmep(self, X, size):
        """Compute the Neighborhood Maximum Ensemble Probability baseline"""
        X_nmep = {}
        for v in self._BASELINE_VARS:
            for t in self._NMEP_THRESHS[v]:
                data = X[f'{v}__time_max__{self._DX*size/2:.0f}km'] 
                data_bin = np.where(data>t,1,0)
                ens_prob = np.nanmean(data_bin, axis=0)
                X_nmep[f"{v}__nmep_>{str(t).replace('.','_')}_{self._DX*size/2:.0f}km"] = ens_prob 
                
        return X_nmep 

    def get_targets(self, TIMESCALE):
        """Convert storm reports to a grid and apply different upscaling"""
        comps = decompose_file_path(self._ncfile)
        #start_time = comps['VALID_DATE']+comps['VALID_TIME']
        start_time=(pd.to_datetime(comps['VALID_DATE']+comps['INIT_TIME'])+dt.timedelta(minutes=int(comps['TIME_INDEX'])*self._deltat)).strftime('%Y%m%d%H%M')

        forecast_length = 180 if TIMESCALE=='0to3' else 240
        report = StormReportLoader(
                reports_path = '/work/mflora/LSRS/StormEvents_2017-2022.csv',
                report_type='NOAA',
                initial_time=start_time, 
                forecast_length=forecast_length, 
                err_window=15,               
            )
        
        ds = xr.load_dataset(self._ncfile, decode_times=False)
        report_ds = report.to_grid(dataset=ds, size=self._upscale_size)
        
        keys = list(report_ds.data_vars)
        
        y = {v : report_ds[v].values[::self._upscale_size, ::self._upscale_size] 
             for v in keys}#Here, we take every 3rd point to reproject from 3km to 9km. What happens if we use resample, so it's consistent with the reprojection of predictors?
        
        # Upscale the targets. 
        y_final = [] 
        for size in self._TARGET_SIZES:
            y_nghbrd = {f'{v}__{self._DX*size/2:.0f}km' : self.neighborhooder(y[v], 
                                                                      func=maximum_filter,
                                                                     size=size, is_2d=True) for v in keys}
            y_final.append(y_nghbrd)
            
        y_final = dict(ChainMap(*y_final)) 
        
        return y_final
    
    
    def resample(self, variable):
        '''
        Resamples (i.e., re-projects, re-grid) the original grid to the target grid 
        using a nearest neighborhood approach
        
        Parameters
        --------------------
            target_grid: 2-tuple of 2D arrays 
                target latitude and longitude grids for the resampling. 
                
            original_grid : 2-tuple of 2D arrays 
                original latitude and longitude grids prior to the resampling. 
                
            variable : 2D array to be resampled
                The grid to be resampled. 
                
        Return
        -----------------------
            variable_nearest, 2D array of variable resampled to the target grid
        '''
        # Create a pyresample object holding the original grid
        orig_def = pyresample.geometry.SwathDefinition(lons=self._original_grid[1], lats=self._original_grid[0])

        # Create another pyresample object for the target grid
        targ_def = pyresample.geometry.SwathDefinition(lons=self._target_grid[1], lats=self._target_grid[0])

        variable_nearest = pyresample.kd_tree.resample_nearest(orig_def, variable, \
                    targ_def, radius_of_influence=50000, fill_value=None)

        return variable_nearest
    
    def neighborhooder(self, X, func, size, is_2d=False, AdamEnv=False):
        """Apply neighborhood function to X. For any grid points with NaN values, 
           replace it with a generic, full-domain spatial average value."""
        new_X = X.copy()
        fill_value = np.nanmean(X)
        if is_2d:
            for n in range(self._n_ens):
                X_ = np.nan_to_num(X[:,:], nan=fill_value)
                new_X[:,:] = func(X_, size)
        else:
            if AdamEnv:
                for t,n in itertools.product(range(new_X.shape[0]), range(self._n_ens)): #Every time step and ens member
                    X_ = np.nan_to_num(X[t, n, :,:], nan=fill_value)
                    new_X[t,n,:,:] = func(X_, size) 
            else:
                for n in range(self._n_ens): #Every Ens Member
                    X_ = np.nan_to_num(X[n,:,:], nan=fill_value) 
                    new_X[n,:,:] = func(X_, size)
        return new_X 
    
    def upscaler(self, X, func, size, remove_nans=False):
        """Applies a spatial filter per ensemble member and timestep and then 
        subsamples the grid to reduce the number of grid points."""
        new_X = np.zeros((X.shape[0], X.shape[1], 
                          self._target_grid[0].shape[0], self._target_grid[0].shape[1] ))
        
        fill_value = np.nanmean(X)
        for t,n in itertools.product(range(new_X.shape[0]), range(self._n_ens)):
            X_ = np.nan_to_num(X[t,n,:,:], nan=fill_value)
            new_X[t,n,:,:] = self.resample(func(X_, size)) #Time, Ens Member, lat/lon
            
        return new_X
    
    def calc_time_composite(self, X, func, name, keys):
        """Compute the time-maximum or time-average"""
        X_time_comp = {f'{v}__{name}' : func(X[v], axis=0) for v in keys }
        return X_time_comp
        
    def calc_spatial_ensemble_stats(self, X, environ=True, FRAMEWORK='POTVIN'):
        """Compute the spatial ensemble mean and standard deviation if environ = True,
        else compute the ensemble 90th. Ensemble statistics are computed in multiple different 
        neighborhood sizes"""
        print(f'{environ} and {FRAMEWORK}')
        keys = X.keys()
        
        X_final = []
        
        for size in self._SIZES:
            if environ:
                if FRAMEWORK=='POTVIN':
                    X_nghbrd = {f'{v}__{self._DX*size/2:.0f}km' : self.neighborhooder(X[v], 
                                                                          func=uniform_filter,
                                                                         size=size, 
                                                                               ) for v in keys}

                    X_ens_mean = {f'{v}__ens_mean' : np.nanmean(X_nghbrd[v], axis=0) for v in X_nghbrd.keys()}
                    X_ens_std = {f'{v}__ens_std' : np.nanstd(X_nghbrd[v], axis=0, ddof=1) for v in X_nghbrd.keys()}   
                    X_ens_stats = {**X_ens_mean, **X_ens_std}
                elif FRAMEWORK == 'ADAM':
                    X_nghbrd = {f'{v}__{self._DX*1/2:.0f}km' : self.neighborhooder(X[v], 
                                                                          func=uniform_filter,
                                                                         size=1, AdamEnv=True  
                                                                               ) for v in keys} #Returns X[t, n, x, y]

                    X_ens_mean = {f'{v}__ens_mean' : np.nanmean(X_nghbrd[v], axis=1) for v in X_nghbrd.keys()} #Returns X[t,x,y]
                    X_ens_stats = {**X_ens_mean} 
            
            else:
                #Block for storm variables
                if FRAMEWORK=='POTVIN':
                    X_nghbrd = {f'{v}__{self._DX*size/2:.0f}km' : self.neighborhooder(X[v], 
                                                                          func=maximum_filter,
                                                                         size=size,      
                                                                               ) for v in keys}
                  
                    X_ens_mean = {f'{v}__ens_mean' : np.nanmean(X_nghbrd[v], axis=0) for v in X_nghbrd.keys()} #Change 

                   
                    X_ens_16th = {f'{v}__ens_16th' : np.nanpercentile(X_nghbrd[v],
                                                                    16/18*100, axis=0, method='higher') for v in X_nghbrd.keys()} 
                    X_ens_2nd = {f'{v}__ens_2nd' : np.nanpercentile(X_nghbrd[v],
                                                                    2/18*100, axis=0, method='lower') for v in X_nghbrd.keys()}
                    X_strm_iqr={f'{v}__ens_IQR' : np.nanpercentile(X_nghbrd[v],
                                                                    75, axis=0, method='higher')-np.nanpercentile(X_nghbrd[v], 25, axis=0, method='lower') for v in X_nghbrd.keys()}


                    # Compute the baseline stuff. 
                    X_baseline = self.get_nmep(X_nghbrd, size)

                    X_ens_stats = {**X_baseline, **X_ens_mean, **X_ens_2nd, **X_strm_iqr, **X_ens_16th} 
                    
                elif FRAMEWORK=='ADAM':
                    X_nghbrd={f'{v}__{self._DX*1/2:.0f}km':self.neighborhooder(X[v],func=maximum_filter, size=1) for v in keys}
                    
                    #Ensemble max at each grid point
                    X_ens_max={f'{v}__ens_max': np.nanmax(X_nghbrd[v], axis=0) for v in X_nghbrd.keys()}
                    
                    #Ensemble 90th %ile at each grid point (No extrapolation)
                    X_ens_90th={f'{v}__ens_16th': np.nanpercentile(X_nghbrd[v], 16/18*100, axis=0, method='higher') for v in X_nghbrd.keys()}
                    
                    #Mean of ensemble at each grid point
                    
                    X_ens_mean={f'{v}__ens_mean':np.nanmean(X_nghbrd[v], axis=0) for v in X_nghbrd.keys()}
                    
                    X_gaussian={f'{v}__smoothed':self.neighborhooder(X_ens_mean[v], func=gaussian_filter, size=size, is_2d=True) for v in X_ens_mean.keys()}
                    
                    
                    X_smoothed_UH={f'{v}__{self._DX*size/2:.0f}km__smoothed' : self.neighborhooder(X[v], func=gaussian_filter, size=size) for v in keys if 'uh_2to5_instant' in v} 
                    #print(X_smoothed_UH.keys())
                    
                    for v in X_smoothed_UH.keys():
                        X_indiv_UH={f'{v}_{n}' : X_smoothed_UH[v][n,:,:] for n in range(self._n_ens)}
                    #print(X_nghbrd.keys())
                    X_baseline=self.get_nmep(X_nghbrd, 1)
                    
                    X_ens_stats={**X_baseline, **X_ens_90th, **X_ens_max, **X_gaussian,**X_indiv_UH}
                                                           
            
            X_final.append(X_ens_stats)
            
                                                          
            
        X_final = dict(ChainMap(*X_final))    
            
        return X_final 
    
def subsampler(y, pos_percent=1.0, neg_percent=0.25):

    pos_inds = np.where(y>0)[0]
    neg_inds = np.where(y==0)[0]
    
    if len(pos_inds) > 0:
        pos_inds_sub = np.random.choice(pos_inds, size=int(pos_percent*(len(pos_inds))), replace=False)
    else:
        pos_inds_sub = [] 
    
    neg_inds_sub = np.random.choice(neg_inds, size=int(neg_percent*(len(neg_inds))), replace=False)

    inds = np.concatenate([pos_inds_sub, neg_inds_sub])
    
    return inds     

def random_subsampler(length, percent=0.5):
    '''Selects a random sample of indices consisting of n% of the dataset'''
    if percent<=0:
        inds=[]
    else:
        n_inds = int(percent*length) #Number of samples to keep
        inds = np.random.choice(np.arange(0,length), size=n_inds, replace=False) #Draw random selection between 0-length-1
    return inds