import sys
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe')
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/mlw2w')
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/samuel.varga/python_packages/WoF_post')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
sys.path.append('/home/samuel.varga/python_packages/ml_workflow/')

# Import packages 
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely
import cartopy
import datetime as dt

# We add the github package to our system path so we can import python scripts for that repo. 
import pandas as pd
from os.path import join, exists
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from wofs.plotting.wofs_colors import WoFSColors
from wofs_ml_severe.data_pipeline.storm_report_loader import StormReportLoader
from wofs_ml_severe.data_pipeline.storm_report_downloader import StormReportDownloader
import os
from glob import glob 

base_path = f'/work/samuel.varga/data/2to6_hr_severe_wx/sfe_prep/SummaryFiles'
dates=[d for d in os.listdir(base_path) if '.txt' not in d] 

##########################
##Get Paths of Files##
##########################


paths = [] #list of valid paths for worker function
for d in dates:
    
    times = [t for t in os.listdir(join(base_path,d)) if 'basemap' not in t] #initialization time
    
    for t in times: #For every init time on that day
        path = join(base_path,d,t)   
        files = glob(join(path, f'wofs_ML2TO6_*Deep.nc'))  
        if len(files)>0:
            paths.append(files[0])

print(f'Num. Paths: {len(paths)}')
print(paths)

def add_map_stuff(ax, states, shape_feature):
    ax.add_feature(states, linewidth=.1, facecolor='none', edgecolor="black")
    ax.add_feature(cfeature.LAKES, linewidth=.1, facecolor='none', edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=.1, facecolor='none', edgecolor="black")        
    ax.add_feature(shape_feature)
    
    
def set_extent(ax, projection , crs, lats, lons,):
    """ Set the Map extent based the WoFS domain """
    # Set the extent. 
    xs, ys, _ = projection.transform_points(
            crs,
            np.array([lons.min(), lons.max()]),
            np.array([lats.min(), lats.max()])).T
    _xlimits = xs.tolist()
    _ylimits = ys.tolist()

    # The limit is max(lower bound), min(upper bound). This will create 
    # a square plot and make sure there is no white spaces between the map
    # the bounding box created by matplotlib. This also allows us to set the
    # WoFS domain boundaries in cases where we aren't plotting WoFS data 
    # (e.g., storm reports, warning polygons, etc.) 
    lims = (max([_xlimits[0]]+[_ylimits[0]]),min([_xlimits[-1]]+[_ylimits[-1]]))
        
    ax.set_xlim(lims)
    ax.set_ylim(lims) 
    
    return ax #0000 is 0000 the next day


def worker(path):
    preds = xr.load_dataset(path)
    date, init_time = path.split('/')[7], path.split('/')[8]
    outpath = path.replace('.nc', '_2.png')

    try:
        indir = glob(f'/work/mflora/SummaryFiles/{date}/{init_time}/wofs_ALL_24*.nc')[0]
        ds = xr.load_dataset(indir, decode_times=False)
    except:
        indir = glob(f'/work/mflora/SummaryFiles/{date}/{init_time}/wofs_ENS_24*.nc')[0]
        ds = xr.load_dataset(indir, decode_times=False)
        
    lats = ds['xlat'][::3, ::3]
    lons = ds['xlon'][::3, ::3]

    shape = (len(lons), len(lats))

    central_longitude = ds.attrs['STAND_LON']
    central_latitude = ds.attrs['CEN_LAT']

    standard_parallels = (ds.attrs['TRUELAT1'], ds.attrs['TRUELAT2'])
    projection=ccrs.LambertConformal(central_longitude=central_longitude,
                                     central_latitude=central_latitude,
                                     standard_parallels=standard_parallels)
    crs = ccrs.PlateCarree()
    data_path = '/home/samuel.varga/python_packages/WoF_post/wofs/data/'
    states = NaturalEarthFeature(category="cultural", scale="10m",
                                 facecolor="none",
                                 name="admin_1_states_provinces")

    county_file = join(data_path,'COUNTIES', 'countyl010g.shp')
    reader = shpreader.Reader(county_file)
    shape_feature = ShapelyFeature(reader.geometries(),
                                   crs, facecolor='none', linewidth=0.2, edgecolor='black', )
    
    deltat=5
    from scipy.ndimage import uniform_filter, maximum_filter 
    from wofs.plotting.util import decompose_file_path
    from wofs.plotting.wofs_colors import WoFSColors
    from wofs.verification.lsrs.get_storm_reports import StormReports

    # Get the storm reports. 
    comps = decompose_file_path(indir)


    print(init_time)
    print(comps['VALID_DATE']+comps['VALID_TIME'])

    start_time=(pd.to_datetime(comps['VALID_DATE']+comps['INIT_TIME'])+dt.timedelta(minutes=int(comps['TIME_INDEX'])*deltat)).strftime('%Y%m%d%H%M')
    print(start_time)

    forecast_length= 240



    report = StormReportLoader(initial_time=start_time, 
                forecast_length=forecast_length, 
                err_window=15, 
                reports_path='/work/mflora/LSRS/STORM_EVENTS_2017-2023.csv', #.format(str(date)[0:4]), #change this
                report_type='NOAA'
                )
    points = report()
    start_date=(pd.to_datetime(comps['VALID_DATE']+comps['INIT_TIME'])+dt.timedelta(minutes=int(comps['TIME_INDEX'])*deltat)) #Beginning of forecast window
    end_date=(start_date+dt.timedelta(minutes=forecast_length)).strftime('%Y%m%d%H%M') #End of forecast window
    print((start_date, end_date))
    title = f'Valid: {start_time[:4]}-{start_time[4:6]}-{start_time[6:8]} {start_time[8:10]}:{start_time[10:12]} - {end_date[8:10]}:{end_date[10:12]} UTC'
    #This will show as start_date start_time - end_time, even if end_time is on the next day        
    print(title)
    titles = {'hail' : 'Severe Hail', 
         'wind' : 'Severe Wind', 
         'tornado' : 'Tornado',
         'all': 'All Severe'}
    
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(figsize=(10,8), facecolor='w',
                     dpi=170, nrows=2, ncols=2, subplot_kw={'projection': projection},
                        constrained_layout=True)

    h_names=['all','all','all','all'] #Change here
    model_names=[r'Calibrated NMEP (45 km) of UH > 150 $m^2 s^{-2}$', 'Any-Severe', 'Grouped Any-Severe', 'U-Net'] #Change here
    hazard_color = {'hail' : 'g', 'wind': 'b', 'tornado': 'r'}
    levels = np.arange(0, 1.01, 0.1)  
    
    # All 3 Models and BL
    for ax, h, name, in zip(axes.flat, h_names, model_names):
        if name =='Any-Severe':
            pred = preds[f'Sev_all_predictor_scale_all_predictor_type_control'][:]
        elif name =='Grouped Any-Severe':
            pred = preds[f'grouped_any'][:]
        elif name =='U-Net':
            pred = preds[f'deep_learning_full'][:]
        else:
            pred = preds[f'all_severe_baseline'][:]
            
        add_map_stuff(ax, states, shape_feature)
        pred = np.ma.masked_where(pred<=0.025, pred)
        cf = ax.contourf(lons,lats, pred, cmap=WoFSColors.varga_cmap, alpha=0.95, levels=levels,
                    transform = crs) 
        if h=='all':
            for Lorem in ['hail','wind','tornado']:
                _points=points[Lorem]
                ax.scatter(_points[:,1],_points[:,0], s=10, linewidth=0.5, color=hazard_color[Lorem], alpha=0.8, zorder=1, transform=crs, edgecolors='k')
        else:
            _points = points[h]
            ax.scatter(_points[:,1],_points[:,0], linewidth=0.5, s=10, color=hazard_color[h], alpha=0.8, zorder=1, transform=crs, edgecolors='k') #s=10


        ax = set_extent(ax, projection , crs, lats, lons,)
        fontsize = 10 if len(name) > 5 else 12
        ax.set_title(name, fontsize=fontsize)

    fig.tight_layout()
    axes[0,1].annotate(title, (0.3, 1.1), xycoords='axes fraction', fontsize=10, color='k')

    cax = fig.add_axes([0.25, -0.05, 0.5, 0.05])
    fig.colorbar(cf, 
               cax=cax, 
               label=f'Probability of report\n within 36 km of a point\nin the next 2-6 hours', 
              orientation='horizontal', drawedges=True)



    plt.savefig(outpath, transparent=False, bbox_inches='tight')
    plt.close()
    return None

run_parallel(
                func = worker,
                n_jobs = 1,
                args_iterator = to_iterator(paths),
                )
