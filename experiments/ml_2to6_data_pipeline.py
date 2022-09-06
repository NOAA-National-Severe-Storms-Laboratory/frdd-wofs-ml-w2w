from WoF_post.wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)
from glob import glob
from scipy.ndimage import uniform_filter, maximum_filter 
from collections import ChainMap
import numpy as np

from WoF_post.wofs.verification.lsrs.get_storm_reports import StormReports
from WoF_post.wofs.plotting.util import decompose_file_path
import xarray as xr
from glob import glob
from os.path import join
import itertools

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

def get_files(path):
    """Get the ENS, ENV, and SVR file paths for the 2-6 hr forecasts"""
    # Load the X. 
    ens_files = glob(join(path,'wofs_ENS_[2-7]*'))
    ens_files.sort()
    ens_files = ens_files[4:]
    
    svr_files = [f.replace('ENS', 'SVR') for f in ens_files]
    env_files = [f.replace('ENS', 'ENV') for f in ens_files]
    
    return ens_files, env_files, svr_files
    
def load_dataset(path):
    """Load the 2-6 hr forecasts"""
    ens_files, env_files, svr_files = get_files(path)
    
    coord_vars = ["xlat", "xlon", "hgt"]
    X_strm, _, _, _  = load_multiple_nc_files(
                ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS'])

    X_env, _, _, _  = load_multiple_nc_files(
                env_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENV_VARS'])

    X_svr, _, _, _ = load_multiple_nc_files(
                svr_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['SVR_VARS'])

    X_env = {**X_env, **X_svr}

    X_env = {v : X_env[v][1] for v in X_env.keys()}
    X_strm = {v : X_strm[v][1] for v in X_strm.keys()}
    
    return X_env, X_strm, ens_files[0]

class GridPointExtracter:
    """Upscale X, compute time-composites, compute ensemble statistics."""
    def __init__(self, ncfile, env_vars, strm_vars, upscale_size=3):
        self._n_ens = 18
        self._env_vars = env_vars
        self._strm_vars = strm_vars
        
        self._upscale_size = upscale_size
        self._SIZES = [1,3,5]
        self._TARGET_SIZES = [1,2,4,6]
        self._ncfile = ncfile
        self._DX = 9 
        
        self._BASELINE_VARS = ['hailcast', 'uh_2to5_instant', 'ws_80']
        
        self._NMEP_THRESHS = {'hailcast' : [0.5, 0.75, 1.0, 1.25, 1.5], 
                             'uh_2to5_instant' : [50, 75, 100, 125, 150, 175, 200],
                             'ws_80' : [30, 40, 50, 60], 
                            }
        
    def __call__(self, X_env, X_strm, predict=False):
        
        # Pre-processor. Get rid of super high updraft speeds, replace NaNs, etc. 
         
        
        # This X has had a 3-grid point gaussian smoother applied to it. 
        X_env_upscaled = {v  : self.upscaler(X_env[v], 
                                     func=uniform_filter,
                                     size=self._upscale_size) for v in self._env_vars}
        
        # This X has had a 3-grid point maximum filter applied to it. 
        X_strm_upscaled = {v : self.upscaler(X_strm[v], 
                                     func=maximum_filter,
                                     size=self._upscale_size) for v in self._strm_vars}
        
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
        
        X_all = {**X_strm_stats, **X_env_stats}
        
        if predict:
            data = X_all
        else:
            # IF not predicting, then get the target values. 
            y = self.get_targets()
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
        """Compute the NMEP baseline"""
        X_nmep = {}
        for v in self._BASELINE_VARS:
            for t in self._NMEP_THRESHS[v]:
                data = X[f'{v}__time_max__{self._DX*size}km']
                data_bin = np.where(data>t,1,0)
                ens_prob = np.nanmean(data_bin, axis=0)
                X_nmep[f"{v}__nmep_>{str(t).replace('.','_')}_{self._DX*size}km"] = ens_prob
                
        return X_nmep 

    def get_targets(self):
        """Convert storm reports to a grid and apply different upscaling"""
        comps = decompose_file_path(self._ncfile)
        start_time = comps['VALID_DATE']+comps['VALID_TIME']
        report = StormReports(start_time, 
            forecast_length=240,
            err_window=15, 
            )

        ds = xr.load_dataset(self._ncfile, decode_times=False)
        report_ds = report.to_grid(dataset=ds, size=self._upscale_size)
        
        keys = list(report_ds.data_vars)
        
        y = {v : report_ds[v].values[::self._upscale_size, ::self._upscale_size] 
             for v in keys}
        
        # Upscale the targets. 
        y_final = [] 
        for size in self._TARGET_SIZES:
            y_nghbrd = {f'{v}__{self._DX*size}km' : self.neighborhooder(y[v], 
                                                                      func=maximum_filter,
                                                                     size=size, is_2d=True) for v in keys}
            y_final.append(y_nghbrd)
            
        y_final = dict(ChainMap(*y_final)) 
        
        return y_final
    
    def neighborhooder(self, X, func, size, is_2d=False):
        """Apply neighborhood function to X. For any grid points with NaN values, 
           replace it with a generic, full-domain spatial average value."""
        new_X = X.copy()
        fill_value = np.nanmean(X)
        if is_2d:
            for n in range(self._n_ens):
                X_ = np.nan_to_num(X[:,:], nan=fill_value)
                new_X[:,:] = func(X_, size)
        else:
            for n in range(self._n_ens):
                X_ = np.nan_to_num(X[n,:,:], nan=fill_value)
                new_X[n,:,:] = func(X_, size)
    
        return new_X 
    
    def upscaler(self, X, func, size, remove_nans=False):
        """Applies a spatial filter per ensemble member and timestep and then 
        subsamples the grid to reduce the number of grid points."""
        new_X = X.copy()
        fill_value = np.nanmean(X)
        for t,n in itertools.product(range(new_X.shape[0]), range(self._n_ens)):
            X_ = np.nan_to_num(X[t,n,:,:], nan=fill_value)
            new_X[t,n,:,:] = func(X_, size)
        return new_X[:,:,::size, ::size]
    
    def calc_time_composite(self, X, func, name, keys):
        """Compute the time-maximum or time-average"""
        X_time_comp = {f'{v}__{name}' : func(X[v], axis=0) for v in keys }
        
        return X_time_comp
        
    def calc_spatial_ensemble_stats(self, X, environ=True):
        """Compute the spatial ensemble mean and standard deviation if environ = True,
        else compute the ensemble 90th. Ensemble statistics are computed in multiple different 
        neighborhood sizes"""
        
        keys = X.keys()
        
        X_final = []
        
        for size in self._SIZES:
            if environ:
                X_nghbrd = {f'{v}__{self._DX*size}km' : self.neighborhooder(X[v], 
                                                                      func=uniform_filter,
                                                                     size=size, 
                                                                           ) for v in keys}
                
                X_ens_mean = {f'{v}__ens_mean' : np.nanmean(X_nghbrd[v], axis=0) for v in X_nghbrd.keys()}
                X_ens_std = {f'{v}__ens_std' : np.nanstd(X_nghbrd[v], axis=0, ddof=1) for v in X_nghbrd.keys()}   
                X_ens_stats = {**X_ens_mean, **X_ens_std}
            
            else:
                X_nghbrd = {f'{v}__{self._DX*size}km' : self.neighborhooder(X[v], 
                                                                      func=maximum_filter,
                                                                     size=size,      
                                                                           ) for v in keys}
                
                X_ens_90th = {f'{v}__ens_90th' : np.nanpercentile(X_nghbrd[v],
                                                                90, axis=0) for v in X_nghbrd.keys()} 
                
                # Compute the baseline stuff. 
                X_baseline = self.get_nmep(X_nghbrd, size)

                X_ens_stats = {**X_baseline, **X_ens_90th}
            
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



