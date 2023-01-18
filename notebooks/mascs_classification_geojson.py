# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Title 
#
# **Automated surface mapping via unsupervised learning and classification of Mercury Visible--Near-Infrared reflectance spectra**
#

# %% [markdown]
# **Abstract**
#
# In this work we apply unsupervised learning techniques for  dimensionality reduction and clustering to remote sensing  hyperspectral Visible-Near Infrared (VNIR) reflectance spectra  datasets of the planet Mercury obtained by the MErcury Surface, Space  ENvironment, GEochemistry, and Ranging (MESSENGER) mission.
# This  approach produces cluster maps, which group different regions of the  surface based on the properties of their spectra as inferred during  the learning process.
# While results depend on the choice of model  parameters and available data, comparison to expert-generated geologic  maps shows that some clusters correspond to expert-mapped classes such  as smooth plains on Mercury.
# These automatically generated maps can  serve as a starting point or comparison for traditional methods of  creating geologic maps based on spectral patterns.
#
#
# The code and data  used in this work is available as python jupyter notebook on the  github public repository  [MESSENGER-Mercury-Surface-Cassification-Unsupervised_DLR](https://github.com/epn-ml/MESSENGER-Mercury-Surface-Cassification-Unsupervised_DLR) funded by the European Union's Horizon 2020 grant No 871149.
#
# Authors:
# - Mario D'Amore$^1$
# - Sebastiano Padovan$^{1,2,3}$
#
# Affiliations : 
#
# -  $^1$German Aerospace Center (DLR), Rutherfordstraße 2, 12489 Berlin,Germany
# -  $^2$EUMETSAT, Eumetsat Allee 1, 64295 Darmstadt, Germany
# -  $^3$WGS, Berliner Allee 47, 64295 Darmstadt, Germany
#
#

# %% [markdown] tags=[]
# ## Imports
#
# Generic imports

# %% tags=[]
import geopandas as gpd
from shapely.geometry import Polygon
import fiona

import matplotlib
# matplotlib.use('Agg') # non interactive
# #%matplotlib qt # for not-notebook
# %matplotlib inline
from   matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns


matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['font.size'] = 16

pd.set_option('display.width',150)
pd.set_option('display.max_colwidth',150)
pd.set_option('display.max_rows',150)

from IPython.display import display

# %% tags=[]
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hds
import hvplot.pandas
# hv.extension('bokeh','matplotlib')

# %% [markdown]
# Ignore watnings, some holoviews calls should be updated.

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Define auxiliary functions & data

# %% tags=[]
base_path = pathlib.Path('..')                     # local base path
input_data_path = base_path / 'data/processed'  # input data location 
out_figure_path = base_path / 'reports/figures' # output location <- CHANGE THIS TO YOUR LIKING
out_models_path = base_path / 'models' # output location <- CHANGE THIS TO YOUR LIKING

print(f'{base_path=}')
print(f'{input_data_path=}')
print(f'{out_models_path=}')


# this globally saves all generated plot with save_plot defind below
save_plots_bool = 0


# %% tags=[]
# make a figure
def make_map(in_data,alpha,norm,interpolation=None):
    '''
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    '''
    
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    map_crs = ccrs.PlateCarree(central_longitude=0.0)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[18,18],subplot_kw={'projection': map_crs})
    background = ax.imshow(img,
                            cmap=plt.cm.gray,
                            extent=img_extent,
                            origin='upper',
                            transform=ccrs.PlateCarree(central_longitude=0.0)
                          );
    # Here we are using the numpy reshape because we now the final image shape: it is not always the case!!
    # This is FASTER then Geopandas.GeoDataFrame.plot!!!!
    im = ax.imshow(outdf_gdf.sort_index()['R'].values.reshape(360,180).T,
               interpolation= interpolation,
               extent= data_img_extent,
               cmap=plt.cm.Spectral_r,
               transform=ccrs.PlateCarree(central_longitude=0.0),
               origin='upper',
               alpha=alpha,
               # vmax=0.065,
               norm=norm,
                  );
    return im

def save_plot(out_file,
              output_dir = out_figure_path,
              dpi=150,
              out_format='jpg',
              save=False):
    ''' helper function to save previous plot
    '''
    
    out_path = output_dir / (out_file +f'.{out_format}' )
    if save :
        plt.savefig(out_path,dpi=dpi)
        return (f'Saving image to {out_path}')
    else:
        return (f'NOT saving image to {out_path}')  

def df_shader(in_data,**kwargs):
    '''
    accept input : spectral_df[in_wav] 
    add it to outdf_gdf:
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    
    returns shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,'R']))
    '''
    vdims = 'R'
    kdims=[('x','longitude'),('y','latitude')]
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    return shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims),**kwargs)


def df_rasterer(in_data,**kwargs):
    '''
    accept input : spectral_df[in_wav] 
    add it to outdf_gdf:
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    
    returns shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,'R']))
    '''
    vdims = 'R'
    kdims=[('x','longitude'),('y','latitude')]
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    return rasterer(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims).opts(),**kwargs).opts()


def shader(ppoints,aggregator=ds.mean(),x_sampling=1,y_sampling=1,cmap=plt.cm.Spectral_r, dynamic=True):
    return hds.shade(
                hds.rasterize(ppoints,
                              aggregator=aggregator,
                              x_sampling=x_sampling,
                              y_sampling=y_sampling),
                              cmap=cmap,
                              dynamic=dynamic,
                              )

def rasterer(ppoints,aggregator=ds.mean(),x_sampling=2,y_sampling=2,dynamic=True):
    return hds.rasterize(ppoints,
                         aggregator=aggregator,
                         x_sampling=x_sampling,
                         y_sampling=y_sampling,
                         dynamic=dynamic,
                        )


def colorbar_img_shader(in_data,cmap=plt.cm.Spectral_r,**kwargs):
    
    raster = df_rasterer(in_data,**kwargs).opts(alpha=1,colorbar=True,cmap=cmap)
    
    return hv.Overlay([raster,
                        hds.shade(raster,cmap=cmap,group='datashaded'),
                        background.opts(cmap='Gray',alpha=0.5),
                      ])
# .collate()

def find_nearest_in_array(
                 value,
                 array,
                 return_value=False):
    """Find nearest value in a numpy array

    Parameters
    ----------

    value : value to search for
    array : array to search in, should be sorted.
    return_value : bool, if to return the actual values

    Returns
    -------

    """

##TODO add sorting check/option to sort ? make sense?
    import numpy as np

    closest_index = (np.abs(array - value)).argmin()

    if value > np.nanmax(array):
        import warnings
        warnings.warn(f'value > np.nanmax(array) : {value} > {np.nanmax(array)}')

    if value < np.nanmin(array):
        import warnings
        warnings.warn(f'value < np.nanmin(array) : {value} < {np.nanmax(array)}')

    if not return_value: 
        return closest_index
    else:
        return closest_index, array[closest_index]


def get_mascs_geojson(file, gzipped=True):
    """return a geopandas.DataFrame from a geojson, could be compressed.

    Parameters
    ----------

    file : path of geojson compressed file to geopandas dataframe
    gzipped : bool, if compressed

    Returns
    -------

    geopandas.DataFrame
    """

    import geopandas as gpd
    import numpy as np
    import json
    import gzip
    from geopandas import GeoDataFrame

    if gzipped:
        # get the compressed geojson
        with gzip.GzipFile(file, 'r') as fin:
            geodata = json.loads(fin.read().decode('utf-8'))
    else:
        # get the uncompressed geojson
        with open(file, 'r') as fin:
            geodata = json.load(fin)

    import shapely
    # extract geometries
    geometries = [shapely.geometry.Polygon(g['geometry']['coordinates'][0]) for g in geodata['features']]
    # extract id
    ids = [int(g['id']) for g in geodata['features']]
    # generate GeoDataFrame
    out_gdf = gpd.GeoDataFrame(data=[g['properties'] for g in geodata['features']], geometry=geometries, index=ids).sort_index()
    # cast arrays to numpy
    if 'array' in out_gdf:
        out_gdf['array'] = out_gdf['array'].apply(lambda x: np.array(x).astype(float))

    return out_gdf

#CRS from  https://github.com/Melown/vts-registry/blob/master/registry/registry/srs.json
mercury_crs = {
    "geographic-dmercury2000": {
        "comment": "Geographic, DMercury2000 (iau2000:19900)",
        "srsDef": "+proj=longlat +a=2439700 +b=2439700 +no_defs",
        "type": "geographic"
    },
    "geocentric-dmercury2000": {
        "comment": "Geocentric, Mercury",
        "srsDef": "+proj=geocent +a=2439700 +b=2439700 +lon_0=0 +units=m +no_defs",
        "type": "cartesian"
    },
    "eqc-dmercury2000": {
        "comment": "Equidistant Cylindrical, DMercury2000 (iau2000:19911)",
        "srsDef": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "merc-dmercury2000": {
        "comment": "Mercator, DMercury2000 (iau2000:19974)",
        "srsDef": "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "steren-dmercury2000": {
        "comment": "Polar Sterographic North, DMercury2000 (iau2000:19918)",
        "srsDef": "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "steres-dmercury2000": {
        "comment": "Polar Stereographic South, DMercury2000 (iau2000:19920)",
        "srsDef": "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
}}


# %% [markdown] tags=[]
# ## Load Data
#
# Data are too big to be included in this repo, user can find it on [Zenodo](https://zenodo.org/record/7433033) at [https://zenodo.org/record/7433033](https://zenodo.org/record/7433033).
#
# Download the datafile `grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm.geojson.gz` in  `data/processed` with some variation of
#
# ```bash
# curl https://zenodo.org/record/7433033/files/grid_2D_0_360_-90_%2B90_1deg_st_median_photom_iof_sp_2nm.png --output data/processed/grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm.geojson.gz
# ```
#
# This is a preview of the data cube from Zenodo.
#
# ![Preview od the data cube from Zenodo](https://zenodo.org/api/iiif/v2/c98bb0bc-cfa1-449e-94f7-95f9d074543e:f4cc114a-fac9-42b8-b5a8-380901fe8dba:grid_2D_0_360_-90_%2B90_1deg_st_median_photom_iof_sp_2nm.png/full/750,/0/default.png)
#
# Create the wavelenghts array and its helper functions

# %% tags=[]
# create the wavelenghts array
wav_grid_2nm = np.arange(260,1052,2)

# define find_nearest, with wav_grid_2nm as default array
find_nearest = lambda x : find_nearest_in_array(x,wav_grid_2nm)

# this is to index an array based on wav_grid_2nm
wavelenght = 415
print(f'         wavelenght = {wavelenght:5d} <-- number we search for')
print(f'find_nearest({wavelenght:5d}) = {find_nearest(wavelenght):5d} <-- index of {wavelenght} in wav_grid_2nm')

# %% [markdown] tags=[]
#
# Select input data. From MASCS documentation :  
#
# ```
# PHOTOM_IOF_SPECTRUM_DATA : 
#    Derived column of photometrically normalized 
#    reflectance-at-sensor spectra. One row per spectrum. NIR spectrum has up 
#    to 256 values (depending on binning and windowing), VIS has up to 512. 
#    Reflectance is a unitless parameter. Reflectance from saturated pixels, 
#    or binned pixels with one saturated element, are set to 1e32. PER 
#    SPECTRUM column."
#
# IOF_SPECTRUM_DATA : 
#    DESCRIPTION = "Derived column of reflectance-at-sensor spectra. One 
#    row per spectrum. NIR spectrum has up to 256 values (depending on binning 
#    and windowing), VIS has up to 512. Reflectance is a unitless parameter. 
#    Reflectance from saturated pixels, or binned pixels with one saturated 
#    element, are set to 1e32. PER SPECTRUM column."
# ```
#
# define the datafile with filename structure:
#
# ```
# [description from database]_[function applied to the spectra for each pixel]_[data array used]
#     [description from database] = grid_2D_0_360_-90_+90
#     [function applied to the spectra for each pixel] = avg or st_median
#     [data array used] = iof_sp_2nm or photom_iof_sp_2nm
# ```

# %% tags=[]
input_data_name = 'grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm'

# %% [markdown]
# The data are in a gzipped geojson to reduce size, but geopandas doesn't like it.
#
# The function below accept a path and return a GeoDataFrame.
#
# An optional `gzipped[=True default]` keywords take care of compressed geojson.

# %% tags=[]
outdf_gdf = get_mascs_geojson( input_data_path / (input_data_name+'.geojson.gz'), gzipped=True)
# outdf_gdf = get_mascs_geojson( input_data_path / (input_data_name+'.geojson.gz'), gzipped=True, cast_to_numeric=False)

# this is to be sure that the cells are ordered in natural way == reshape with numpy
outdf_gdf = outdf_gdf.set_index('natural_index',drop=True).sort_index()
import fiona.crs

# set Mercury Lat/Lon as crs
outdf_gdf.crs = fiona.crs.from_string(mercury_crs['geographic-dmercury2000']['srsDef'])

# %% [markdown]
# Unravel spectral reflectance data

# %% tags=[]
# create wavelenghts columns: this create empy columns with np.nan (nice!)
# use a separate df, because mixed types columns are crazy. and buggy
spectral_df = pd.DataFrame(index=outdf_gdf.index,columns = wav_grid_2nm).fillna(np.nan)
## assign single wavelenght to columns, only where array vectors len !=0
spectral_df.loc[outdf_gdf['array'].apply(lambda x : len(x)) != 0, wav_grid_2nm] = np.stack(outdf_gdf.loc[outdf_gdf['array'].apply(lambda x : len(x)) != 0,'array'], axis=0).astype(np.float64)
## drop array column
outdf_gdf.drop(columns=['array'], inplace=True)

# %% tags=[]
# create x and y cols = lon and lat
outdf_gdf['x'] = outdf_gdf.apply(lambda x: x['geometry'].centroid.x , axis=1)
outdf_gdf['y'] = outdf_gdf.apply(lambda x: x['geometry'].centroid.y , axis=1)

# %% tags=[]
# drop outlier : this clean up further noisy data, instrumental effect, etc
print(spectral_df.shape)
low = .02
high = .999
quant_df = spectral_df.quantile([low, high])
spectral_df =  spectral_df[spectral_df >= 0].apply(lambda x: x[(x>quant_df.loc[low,x.name]) &\
                                       (x < quant_df.loc[high,x.name])], axis=0)\
                                       
print(spectral_df.shape)

# %% tags=[]
# cut to stop_wav
# iloc doesn't support int columns indexing!!!!
start_wav = 268 # below all NaN
stop_wav  = 975 # above a bump in NaN 

spectral_df = spectral_df.iloc[:,find_nearest(start_wav):find_nearest(stop_wav)+1]

# %% tags=[]
# count nan
print(f'{spectral_df.shape=}')

# %% tags=[]
fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(19,1.5))

spectral_df.isna().sum(axis=0).plot(ax=axs[0]);
spectral_df.isna().sum(axis=1).plot(ax=axs[1]);

# %% tags=[]
import seaborn as sns
print(spectral_df.shape, spectral_df.dropna(axis=0,how='any').shape)

display(spectral_df.sample(10))

# %% tags=[]
# whole data distribution 
fig, axarr = plt.subplots(nrows=1, ncols=2,figsize=(19,2))
sns.histplot(spectral_df.dropna(how='any').values.flatten(), ax= axarr[0],bins = 255,alpha=0.4, kde=True,edgecolor='none');
sns.histplot(spectral_df.dropna(how='any').values.flatten(), ax= axarr[1],bins = 255,alpha=0.4, kde=True,edgecolor='none',log_scale=(False, True),);

save_plot('whole_data_distribution_seaborn',save=save_plots_bool)


# %% tags=[]
img = spectral_df.dropna(how='any').values.T

from skimage import exposure

# Equalization : paramterless
plt.figure(figsize=[24,8]);
plt.imshow(exposure.equalize_hist(img),
           interpolation='bicubic',
           aspect='auto',
           cmap=plt.cm.Spectral_r);
# [Colorbar Tick Labelling Demo — Matplotlib 3.1.2 documentation](https://matplotlib.org/3.1.1/gallery/ticks_and_spines/colorbar_tick_labelling_demo.html)
cbar = plt.colorbar(ticks=[0.01, 0.5, 1], orientation='vertical')
cbar.ax.set_yticklabels(
    np.around([np.percentile(img,0.01), np.median(img), np.nanmax(img)],decimals=3)
    );

plt.tight_layout()

save_plot('spectrogram',save=save_plots_bool)

# %% tags=[]
# keep the old name? naa
spectral_df_nona_index = spectral_df.dropna(how='any').index

print(f'        outdf_gdf : {outdf_gdf.shape}')
print(f'      spectral_df : {spectral_df.shape}')
print(f'spectral_df_nonan : {spectral_df_nona_index.shape}') 

# %% tags=[]
# define 2 wavelenghts and calculate something 
in_wav = 450
en_wav = 1050

idx_in = find_nearest(in_wav)
idx_en = find_nearest(en_wav)

outdf_gdf.loc[spectral_df_nona_index,'refl'] = spectral_df[in_wav]

print(f'(wav[{idx_in}],wav[{idx_en}]) = ({wav_grid_2nm[idx_in]}, {wav_grid_2nm[idx_en]}) \nspectral[:,idx_in:idx_en].shape : {spectral_df.loc[:,in_wav:en_wav].shape}')


# %% tags=[]
# calculate rows & cols from grid properties files, assuming regular grid
rows, cols = 360, 180
rows_half_step, cols_half_step = 1, 1

data_img_extent = [-180.0, 180.0, -90.0, 90.0]

# extent = [outdf_gdf.total_bounds[i] for i in [0,2,1,3]]

# %% tags=[]
# specific wavelengths data distribution 

fig, axarr = plt.subplots(nrows=1, ncols=1,figsize=(20,4))

plot_wav = 300
ax = sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');
ax.set_xlim([0.005,0.075])
plot_wav = 700
sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');
plot_wav = 900
sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');


plt.tight_layout()

save_plot('global_300nm_700nm_900nm_distribution', out_format='png',save=save_plots_bool)

# %% [markdown]
# ## Plotting section

# %% tags=[]
import cartopy.crs as ccrs
import cartopy
# from scipy import misc
import imageio
from skimage import transform 
# read background image
img = imageio.imread( input_data_path / '1280x640_20120330_monochrome_basemap_1000mpp_equirectangular.png')
img_extent = (-180, 180, -90, 90)

# %% tags=[]
hv.extension('bokeh')

from PIL import Image, ImageEnhance
kdims=[('x','latitude'),('y','longitude')]

backimage = Image.open( input_data_path / '1280x640_20120330_monochrome_basemap_1000mpp_equirectangular.png')
## from help :  as four-tuple defining the (left, bottom, right and top) edges.
hv_img_extent = (-180, -90,180, 90)

background = hv.Image(
            np.array(backimage),
            bounds=hv_img_extent,
            kdims=kdims,
            group='backplane',
            ).opts(cmap='Gray',clone=False)

# # show the background image
# background.options(width=800,height=400)

# %% tags=[]
map_crs = ccrs.PlateCarree(central_longitude=0.0)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[18,18],subplot_kw={'projection': map_crs})
background_mpl = ax.imshow(img,
                        cmap=plt.cm.gray,
                        extent=img_extent,
                        origin='upper',
                        transform=ccrs.PlateCarree(central_longitude=0.0)
                      );

# %% tags=[]
hv.extension('matplotlib')
# hv.output(fig='png')

# %% tags=[]

# %% tags=[]
#     vdims = 'R'
#     kdims=[('x','longitude'),('y','latitude')]
#     outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

#     return shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims).opts(),**kwargs).opts()

spectral_df[970]

# %% tags=[]
hv.extension('matplotlib')

# spectral slope R[970]-R[270] / 270 -970
(background.opts(cmap='Gray',alpha=0.5)*\
 df_shader((spectral_df[970]-spectral_df[270])/700.,cmap=plt.cm.Spectral_r).opts(interpolation='bilinear',alpha=0.5)).\
    opts(
    fig_inches=4, aspect=2,fig_size=200
)

# %% tags=[]
hv.extension('matplotlib')
plot_wav = 700

out = colorbar_img_shader(spectral_df[plot_wav])
_ = out.DynamicMap.II.opts(interpolation='None',aspect=1.8,fig_size=200,alpha=0.7)
out.collate()
# hv.save(out,out_figure_path / '1b_mascs_700nm_refl.png')

# %% tags=[]
hv.extension('bokeh')
background.opts(cmap='Gray')*df_shader(spectral_df[plot_wav],cmap=plt.cm.inferno).opts(height=600,width=1000,alpha=0.7)

out = colorbar_img_shader(spectral_df[plot_wav])
_ = out.DynamicMap.II.opts(height=400,width=800,alpha=0.75)
out

# %% [markdown] tags=[]
# ## Machine Learning

# %% tags=[]
# small_data == spectral_df[spectral_df_nonan]
X = spectral_df.loc[spectral_df_nona_index]

print(f'        outdf_gdf : {outdf_gdf.shape}')
print(f'      spectral_df : {spectral_df.shape}')
print(f'spectral_df_nonan : {spectral_df_nona_index.shape}') 


# %% [markdown]
# ### Dimensionality reduction
#
# [2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 0.20.2 documentation](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)

# %%
from sklearn import decomposition
from sklearn import model_selection
from sklearn import pipeline, preprocessing

# %% [markdown] heading_collapsed=true
# #### PCA

# %% hidden=true tags=[]
# Principal components analysis : which is the data dimensionality?

# explicitely set number of PCA components to comput
n_components = 8
pca = decomposition.PCA(n_components=n_components)

# # let PCA decide the number of components to reconstruct 95% of the total variance 
# pca = decomposition.PCA(0.99)

pca.fit(X)
# n_components = pca.n_components_
X_pca = pca.transform(X)

print('X.shape               : {}\n'
      'X_pca.shape           : {}\n'
      'pca.components_.shape : {}'.format(X.shape, X_pca.shape, pca.components_.shape))

print("              variance       var_ratio      cum_var_ratio")
for i in range(n_components):
    print("Component %2s: %12.10f   %12.10f   %12.10f" % (i, pca.explained_variance_[i], pca.explained_variance_ratio_[i], np.cumsum(pca.explained_variance_ratio_)[i]))


# As we can see, only the 2 first components are useful
# pca.n_components = 2
# small_data_pca = pca.fit_transform(small_data)

# %% hidden=true tags=[]

# %% [markdown] heading_collapsed=true hidden=true
# ##### PCA residual error estimation

# %% hidden=true tags=[]
pca_error = []

cmp_index = np.linspace(0, pca.n_components_, num=pca.n_components_//2, endpoint=True, dtype=int)
cmp_index[-1] -= 1

for i in cmp_index:
# for i in range(pca.n_components):
    print("Component %2s: %12.10f   %12.10f   %12.10f" % (i, pca.explained_variance_[i], pca.explained_variance_ratio_[i], np.cumsum(pca.explained_variance_ratio_)[i]))
    img = (X-np.dot(X_pca[:,:i],pca.components_[:i])-X.mean()).values
    pca_error.append({
        'min': img.min(),
        'max': img.max(),
        'mean' : img.mean(),
        'median' : np.median(img),
        'std' : np.std(img),
        'pca_scores' : np.mean(model_selection.cross_val_score(decomposition.PCA(n_components=n_components).fit(X), X,cv=2))
    })
    print(pca_error[-1])

pca_errors_df = pd.DataFrame.from_dict(pca_error)
pca_errors_df['delta'] = pca_errors_df['max']-pca_errors_df['min']
pca_errors_df['explained_variance'] = pca.explained_variance_[cmp_index]
pca_errors_df['explained_variance_ratio'] = pca.explained_variance_ratio_[cmp_index]

# %% hidden=true tags=[]
pca_errors_df.describe()

# %% hidden=true
import hvplot.pandas
a = (pca_errors_df[['std']]).hvplot().opts(width=1200,height=400)
b = (pca_errors_df['delta']).hvplot().opts(width=1200,height=400)
c = pca_errors_df['pca_scores'].hvplot().opts(width=1200,height=400)
d = pca_errors_df[['max','min']].hvplot().opts(width=1200,height=400)

c
# ((a+b+c)*hv.HLine(10000*(0.005*0.005)).opts(color='red')).cols(1)

# %% hidden=true tags=[]
# %matplotlib inline

from skimage import exposure
img = (X-np.dot(X_pca,pca.components_)-X.mean()).values

plt.figure(figsize=[20,8]);
plt.imshow(exposure.equalize_hist(img,nbins=512),
           aspect='auto',
           interpolation='bilinear',
           cmap=plt.cm.Spectral_r,
           extent = [X.columns.min(),X.columns.max(),  X.index.min(), X.index.max()],
          );
# cbar = plt.colorbar(ticks=[0.01, 0.5, 1], orientation='vertical')
# cbar.ax.set_yticklabels(
#     np.around([np.percentile(img,0.01), np.median(img), np.nanmax(img)],decimals=3)
#     );
plt.show()

# %% [markdown] hidden=true tags=[]
# reconstruct initial data with choosen PCA components  and look at the difference

# %% hidden=true tags=[]
diff_img = (X-np.dot(X_pca,pca.components_)-X.mean())
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[20,8])

ax = axs.flatten()

ax[0].set_title('max')
diff_img.max().plot(ax=ax[0])

ax[1].set_title('median')
diff_img.median().plot(ax=ax[1])

ax[2].set_title('min')
diff_img.min().plot(ax=ax[2])

ax[3].set_title('std')
diff_img.std().plot(ax=ax[3])

# %% hidden=true tags=[]
# hv.extension('matplotlib')
# plot_wav = 500
# out = colorbar_img_shader((X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav])
# _ = out.DynamicMap.II.opts(interpolation='bicubic',aspect=2,fig_size=400,alpha=0.7)
# out

# spectral_df.columns.min(),np.quantile(spectral_df.columns,0.25),np.quantile(spectral_df.columns,0.75),spectral_df.columns.max()
# (268, 444, 797.5, 974)

hv.extension('matplotlib')
    
NdLayout = hv.NdLayout(
            {f'{plot_wav}nm': df_shader( (X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav],x_sampling=2,y_sampling=2).\
            opts(interpolation='bicubic',aspect=2,alpha=0.8)
                 for ind,plot_wav in enumerate([270, 470, 770, 970])}
            ,kdims='Wav')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(2)).opts(fig_size=200,tight=True)

# %% [markdown] heading_collapsed=true hidden=true
# ##### PCA visualisation

# %% hidden=true tags=[]
# %matplotlib inline

components_shift = 0

fig = plt.figure(figsize=(12,4))
ax = plt.subplot()
ax.plot(np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:])
ax.plot(np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:], "r.")
ax.set_title("PCA explained variance")
ax.set_yscale('log')
# ax.set_xlim([components_shift-1, pca.n_components_])
# ax.set_ylim([np.min(pca.explained_variance_ratio_),np.max(pca.explained_variance_ratio_)])
plt.show()

# %% tags=[]
hv.Curve(
                   (np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:]),
                   kdims='components',vdims='var. ratio')


# %% hidden=true tags=[]
components_shift = 0
hv.extension('bokeh')

overlay = hv.Curve(
                   (np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:]),
                   kdims='components',vdims='var. ratio')

overlay2 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.cumsum(pca.explained_variance_ratio_[components_shift:])),
                    kdims='components',vdims='var. cumsum')

overlay3 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.gradient(np.cumsum(pca.explained_variance_ratio_[components_shift:]))),
                    kdims='components',vdims='gradient(var. cumsum)')

overlay4 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.gradient(np.gradient(np.cumsum(pca.explained_variance_ratio_[components_shift:])))) ,
                    kdims='components',vdims='$gradient^2$(var. cumsum)' )


layout = overlay+overlay2+overlay3+overlay4

layout.opts(
    hv.opts.Curve(line_width=3,height=500, width=600),
    hv.opts.Points(alpha=0.5, size=10),
).cols(2)

# %% hidden=true
hv.extension('bokeh')

vdims = ['~Reflectance']
kdims=['Wavelenght (um)']

width=1100
height=500

shift = 0.3
overlay_dict = {'PCA.{}'.format(ind):hv.Curve((spectral_df.columns.to_numpy(),cmp + shift*(ind+1)),vdims=vdims,kdims=kdims) for ind,cmp in 
                enumerate(preprocessing.MinMaxScaler().fit_transform(pca.components_.T).T[:4,:])}

overlay_dict['Mean'] = hv.Curve((spectral_df.columns.to_numpy(),
                  preprocessing.MinMaxScaler().fit_transform(spectral_df.mean().values.reshape(-1, 1)).squeeze()
                                ),vdims=vdims,kdims=kdims).opts(line_width=1,line_dash='solid',color='black',alpha=1)

# shift_dict = {'PCA.{}-shift'.format(ind):hv.HLine(0.5*ind).opts(
#                 line_width=0.25,line_dash='dashed',color='black') for ind in range(4)}


hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4, show_grid=True),
                hv.opts.NdOverlay(width=width,height=height,legend_cols=4,legend_position='bottom')
                ) #* hv.NdOverlay(shift_dict)

# %% hidden=true
hv.extension('bokeh')

max_pca_comp = 4

NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=2,y_sampling=2).\
             options(height=350,width=550,alpha=0.8)
                 for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T)}
            ,kdims='Component')

background.options(cmap='Gray',alpha=0.9)*NdLayout.cols(2)

# %% hidden=true
hv.extension('matplotlib')

max_pca_comp = 16

for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T):
    print(f'PCA.{ind:02} min:{cmp.min():10.5} , max:{cmp.max():10.5}, delta:{cmp.max()-cmp.min():10.5}')

NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=2,y_sampling=2).\
            opts(interpolation='bicubic',aspect=2,alpha=0.7)
                 for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T)}
            ,kdims='Component')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(4)).opts(tight=True, vspace=0.01, hspace=0.01, fig_size=150).cols(4)

# %% hidden=true
hv.extension('bokeh')

import itertools
gridplot = {}
for x,y in itertools.combinations(range(3), 2):
    print(x,y,X_pca[:,[x,y]].shape)
    gridplot[f'PCA.{x} vs PCA.{y}'] = hv.Points(X_pca[:,[x,y]],kdims=[f'PCA.{x}',f'PCA.{y}'])


# # hds.dynspread(
hds.datashade(hv.NdLayout(gridplot),aggregator=ds.count(),cmap=plt.cm.viridis, x_sampling=0.003, y_sampling=0.003).\
opts(height=1000,width=1200,tight=True).cols(3)
# opts(fig_size=200,aspect_weight=True,tight=True).cols(3)


# %% [markdown]
# #### ICA
#
# [2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 0.20.2 documentation](https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica)
#
# From the documentation :
#
# Independent component analysis separates a multivariate signal into additive subcomponents that are maximally independent. It is implemented in scikit-learn using the Fast ICA algorithm. Typically, ICA is not used for reducing dimensionality but for separating superimposed signals. Since the ICA model does not include a noise term, for the model to be correct, whitening must be applied. This can be done internally using the whiten argument or manually using one of the PCA variants.
#
# It is classically used to separate mixed signals (a problem known as blind source separation).
#
# Calculate the reconstruction error for increasing number of ICA components :
#
#     print(np.concatenate((np.arange(1,5),np.arange(0,160,20)[1:])))
#     array([  1,   2,   3,   4,  20,  40,  60,  80, 100, 120, 140])

# %% [markdown] heading_collapsed=true hidden=true
# ##### ICA residual error estimation

# %% tags=[]
ica_rec_error_path = pathlib.Path(out_models_path / 'ica_rec_error_df.csv')

# check if we already run and stored this 
if ica_rec_error_path.is_file():
    # load reconstruction error
    ica_rec_error_df = pd.read_csv(ica_rec_error_path, index_col='ICA components n.')
else:
    # calculate reconstruction error
    ica_rec_error = {}
    for ica_n_components in np.concatenate((np.arange(1,5),np.arange(0,160,20)[1:])) : 
    #     print(ica_n_components)
        ica = decomposition.FastICA(n_components=ica_n_components,random_state=4)
        S_  = ica.fit_transform(X)
        # evaluate overall reconstruction error
        ica_rec_error[ica_n_components] = np.std((X-ica.inverse_transform(S_)).values.flatten())
        print(ica_n_components,ica_rec_error[ica_n_components])

    ica_rec_error_df = pd.DataFrame.from_dict(ica_rec_error,orient='index')
    ica_rec_error_df.index.name = 'ICA components n.'
    ica_rec_error_df.columns = ['reconstruction error']
    ica_rec_error_df.to_csv(ica_rec_error_path)

# %%
hv.extension('bokeh')
display(ica_rec_error_df.sort_index())
import hvplot.pandas

(hv.HLine(0.0015,group='line')*\
 hv.VLine(4,group='line')*\
 ica_rec_error_df.sort_index().hvplot()*\
 ica_rec_error_df.sort_index().hvplot(kind='scatter')).\
opts(
    hv.opts.Points(marker='circle',alpha=0.5),
    hv.opts.Curve(color='red') 
     )

# (hv.RGB(np.random.rand(10, 10, 4), group='A') * hv.RGB(np.random.rand(10, 10, 4), group='B')).opts(
#     hv.opts.RGB('A', alpha=0.1), hv.opts.RGB('B', alpha=0.5)
# )

# # ica_rec_error_df
# hv.extension('bokeh')

# (ica_rec_error_df/X.min().min()).hvplot()

# %% tags=[]
# %matplotlib inline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

fig, ax = plt.subplots(figsize=[12,6])
ica_rec_error_df.sort_index().plot(ax = ax,label=False,marker='o',legend=False)
ax.hlines(0.0015,xmin = ica_rec_error_df.index.min(), xmax = ica_rec_error_df.index.max(), color='red')
ax.vlines(4,ymin = ica_rec_error_df.min(), ymax = ica_rec_error_df.max(), color='green')
ax.set_ylim( [ ica_rec_error_df.min().values[0], ica_rec_error_df.max().values[0]] )

axin = inset_axes(ax, width='60%', height='60%', loc=1)
ica_rec_error_df.sort_index().plot(ax = axin,label=False,marker='o',legend=False)
axin.hlines(0.0015,xmin = ica_rec_error_df.index.min(), xmax = ica_rec_error_df.index.max(), color='red')
axin.vlines(4,ymin = ica_rec_error_df.min(), ymax = ica_rec_error_df.max(), color='green')

axin.set_ylim([0.00144, 0.00165])
axin.set_xlim([2.5, 5])
axin.axes.get_xaxis().set_visible(False)
axin.axes.get_yaxis().set_visible(False)

mark_inset(ax, axin, loc1=2, loc2=3, fc="gray", alpha=0.3, ec="0.5");
plt.tight_layout()
plt.show()
save_plot('ICA_reconstruction_error_zoom_included', out_format='png',save=save_plots_bool)


# %%
# Compute ICA

ica_n_components= 4

# for ica_n_components in range(2,16):
ica = decomposition.FastICA(n_components=ica_n_components,random_state=4)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_           # Get estimated mixing matrix

# coefficients matrix
# S_ /= S_.std(axis=0)

# vector components = signal
# A_ -= A_.mean(axis=0)
# X ~= S_ x A_.T

print(f'ica_n_components : {ica_n_components}')
print(f'X : {X.shape}')
print(f'A_ : {A_.shape}')
print(f'S_ : {S_.shape}')
# print(f'square(sum(X-ICA^-1(X)) : {np.sum((X-ica.inverse_transform(S_))**2)}')
# print(f'np.norm(X-ICA^-1(X),2) : {np.linalg.norm(X-ica.inverse_transform(S_),2)}')
print(((X-ica.inverse_transform(S_)).max()-(X-ica.inverse_transform(S_)).min()).describe().T)

# %%
## reconstruction error at specific wav ma
# h.extension('matplotlib')
# plot_wav = 500
# out = colorbar_img_shader((X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav])
# _ = out.DynamicMap.II.opts(interpolation='bicubic',aspect=2,fig_size=400,alpha=0.7)
# out

# spectral_df.columns.min(),np.quantile(spectral_df.columns,0.25),np.quantile(spectral_df.columns,0.75),spectral_df.columns.max()
# (268, 444, 797.5, 974)

hv.extension('matplotlib')
    
NdLayout = hv.NdLayout(
#           difference maps
            {f'{plot_wav}nm': df_shader( (X-ica.inverse_transform(S_))[plot_wav],x_sampling=1,y_sampling=1).\
#           only reoconstructed vectors maps
#              {plot_wav: df_shader( ica.inverse_transform(S_)[:,plot_wav//2-X.columns[0]],x_sampling=2,y_sampling=2).\

            opts(interpolation='bicubic',aspect=2,alpha=0.8)
                 for ind,plot_wav in enumerate([270, 470, 770, 970])}
            ,kdims='Wav')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(2)).opts(fig_size=200,tight=True)

# %% tags=[]
# %matplotlib inline
np.sqrt(((X-ica.inverse_transform(S_)).max()-(X-ica.inverse_transform(S_))).apply(np.square).sum()/X.size).plot(figsize=[20,3])
plt.show()

# %%
hv.extension('bokeh')

vdims = ['Reflectance']
kdims=['wavelenght (nm)']
width=1100
height=500

overlay_dict = {f'ICA comp. n.{ind}':hv.Curve((spectral_df.columns.to_numpy(),cmp),vdims=vdims,kdims=kdims) for ind,cmp in enumerate(A_.T) }

overlay_dict['Mean'] = hv.Curve((spectral_df.columns.to_numpy(),
                  preprocessing.MinMaxScaler().fit_transform(spectral_df.mean().values.reshape(-1, 1)).squeeze()
                                ),vdims=vdims,kdims=kdims).opts(line_width=1,line_dash='dashed',color='black',alpha=1)


hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4, show_grid=True),
                hv.opts.NdOverlay(width=width,height=height)
                )

# %% tags=[]
hv.extension('matplotlib')

overlay_dict['Mean'].opts(linewidth=1, color='black',alpha =0.25)

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.NdOverlay(fig_size=500, aspect=2.5))
out
save_plot('ICA_components', out_format='png',save=save_plots_bool)


# %%
# [python - Holoviews change datashader colormap - Stack Overflow](https://stackoverflow.com/a/59837074)
from holoviews.plotting.util import process_cmap
# [Colormaps — HoloViews 1.12.7 documentation](http://holoviews.org/user_guide/Colormaps.html)
# process_cmap("Plasma")

# %%
hv.extension('matplotlib')

max_comp = ica_n_components

# sampling = 2
# cmp = S_[:,2]
# out = colorbar_img_shader(cmp[:,np.newaxis],x_sampling=sampling,y_sampling=sampling)
# _ = out.DynamicMap.II.opts(interpolation='bilinear',aspect=2,fig_size=400,alpha=1)
# out

for ind,cmp in enumerate(S_[:,:max_comp].T):
    print(f'ICA.{ind:02} min:{cmp.min():10.5} , max:{cmp.max():10.5}, delta:{cmp.max()-cmp.min():10.5}')

sampling = 1
NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=sampling,y_sampling=sampling).\
            opts(interpolation='bicubic',aspect=2,alpha=0.85)
                for ind,cmp in enumerate(S_.T)}
            ,kdims='ICA Component')

out = (background.opts(cmap='Gray',alpha=1)*NdLayout.cols(4)).opts(tight=True, vspace=0.01, hspace=0.01, fig_size=200).cols(2)
out
# hv.save(out, out_figure_path / 'ICA_components_map.png')

# %% tags=[]
# hv.extension('bokeh')
hv.extension('matplotlib')

df = pd.DataFrame(
    data = S_,
    columns=[f'ICA.{ica_c}' for ica_c in range(ica_n_components)]
)

# # limit to the two "major components"
# df_ds = hv.Dataset(df[['ICA.1','ICA.2']])
df_ds = hv.Dataset(df)

sampling = (df.max()-df.min()).describe().mean()/250

def local_datashade ( X,
                     aggregator=ds.count(),
                     cmap=plt.cm.Spectral_r,
                     x_sampling=sampling,
                     y_sampling=sampling,
                     **kwargs):
    return hds.datashade(X,aggregator=aggregator,cmap=cmap,x_sampling=x_sampling,y_sampling=y_sampling, **kwargs)

point_grid = hv.operation.gridmatrix(df_ds, diagonal_operation=hv.operation.histogram.instance(num_bins=50) ).map(local_datashade, hv.Scatter)
# .opts(hv.opts.RGB(interpolation='bilinear',aspect=1,fig_size=200))

# %% tags=[]
out = point_grid.opts(fig_size=200).opts(hv.opts.RGB(interpolation='bilinear',aspect=1))
out
# hv.save(out, out_figure_path / 'ICA_coefficients_gridplot_density_C1_C2.png', dpi=200)
# hv.save(out, out_figure_path / 'ICA_coefficients_gridplot_density.png', dpi=200)

# %% [markdown]
# ### Manifold Learning
#
# [2.2. Manifold learning — scikit-learn 0.22.1 documentation](https://scikit-learn.org/stable/modules/manifold.html)
#
# Let's load cached embedding data.
# Filenames are splitted to retrive some paramters.
#
# **WARNING :those methods are time/CPU consuming!**

# %%
def split_name(path):
    name=path.stem.split('_')
    out_dict = {'path':path,'kind':name[1],'algorithm':name[0]}
    for el in name[2:]:
        out_dict[el.split('-')[0]]=el.split('-')[1]
    return out_dict

cache_df = pd.DataFrame.from_dict(
                [
                split_name(p) for p in pathlib.Path('../models/').glob('*.npy')
                ]
            )

cache_df=cache_df.sort_values(by=cache_df.columns[1:].tolist())

cache_df

# %% [markdown] heading_collapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# #### T-distributed stochastic neighbor (T-SNE) embedding

# %%
# select T-SNE cache and sample 1 random 
cache_df.query('algorithm == "tsne"').dropna(axis=1)

# %%
cache_df.query('algorithm == "tsne"').dropna(axis=1).columns

# %% hidden=true
# set cache file pathlib.Path
cached_tsne_dict = cache_df.query('algorithm == "tsne"').iloc[1].to_dict() 

cached_tsne   = cached_tsne_dict['path']
pcacomponets  = cached_tsne_dict['path']
pcacomponents = cached_tsne_dict['pcacomponents']
perplexity    = cached_tsne_dict['perplexity']

# load file 
if cached_tsne.is_file():
    print(f'Loading: {cached_tsne}')
    X_tsne = np.load(cached_tsne)
else:
    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(X)
    np.save(cached_tsne,X_tsne)    

print(X.shape, X_tsne.shape)

# %% hidden=true
hv.extension('matplotlib')

vdims = ['label']
kdims=['TSNE.0','TSNE.1']

perplexities = [5, 30, 50, 100]

### RANDOM SAMPPLING
# randomize = np.random.choice(X_pca.shape[0], size=1000)
# label=labels[randomize]
# X = X_pca[randomize,:]

overwrite=True
label= None
X = X_pca
print(f'TSN X.shape : {X.shape}')

# %% hidden=true
filenameextra ='_pcacomponents-{}'.format(n_components)

def get_tsne(perplexity=None,filename_extra=filenameextra, overwrite=False):
    from sklearn.manifold import TSNE
    cached_tsne = pathlib.Path('../models/tsne_embedding_perplexity-{}{}.npy'.format(perplexity,filename_extra))
    if cached_tsne.is_file():
        print(f'Loading: {cached_tsne}')
        x_tsne = np.load(cached_tsne)
    else:
        print(f'file not found: {cached_tsne} - Calculating!')
        if overwrite:
            print(f'Not found, calculating {cached_tsne}')
            x_tsne = TSNE(n_components=2).fit_transform(X)
            np.save(cached_tsne,x_tsne)
        else:
            print(f'overwrite set to {overwrite} stopping')
    print('X.shape : {}, X_tsne.shape : {}, perplexity : {}'.format(X.shape, x_tsne.shape,perplexity))
    return x_tsne


def tsne_to_holocurve(*argv, **kwargs):
    xtsne = get_tsne(perplexity=kwargs['perplexity'],filename_extra=kwargs['filename_extra'], overwrite=kwargs['overwrite'])
    print(kwargs)
    if 'label' in kwargs and kwargs.get('label') is not None:
#         print('Label')
#         print(np.unique(kwargs.get('label'),return_counts=True))
        return hv.Scatter(np.hstack([xtsne ,kwargs.get('label')[:,np.newaxis]]),vdims=kwargs['vdims'],kdims=kwargs['kdims'])
    else:
        print('Nolabel')
        return hv.Scatter(xtsne,kdims=kwargs['kdims'][0], vdims=kwargs['kdims'][1])
    
curve_dict = {p:tsne_to_holocurve(label=label,
                                  perplexity=p,
                                  overwrite=overwrite,
                                  filename_extra=filenameextra,
                                  vdims=vdims,
                                  kdims=kdims)
              for p in perplexities}

# %% hidden=true
label= None
min_size, max_size = 20, 30

if label is not None:
    print('labels = ',label.shape)
    classes , s = np.unique(label, return_counts=True)
    print('classes size:',dict(zip(classes,s)))
    # marker size inverse proportional to population size
    sizes = ((1-(s-s.min())/(s.max()-s.min()))*(max_size-min_size))+min_size
    NdLayout = hv.NdLayout(curve_dict, kdims='perplexity').opts(hv.opts.Scatter(s= [dict(zip(classes,sizes)).get(l) for l in label]),hv.opts.NdLayout(fig_inches=6))
    NdLayout.opts(hv.opts.Scatter(color=vdims[0]))
else:
    print('no labels = ')
    NdLayout = hv.NdLayout(curve_dict, kdims='perplexity').opts(hv.opts.Scatter(s=min_size),hv.opts.NdLayout(fig_inches=6))

NdLayout.opts(hv.opts.Scatter(alpha=0.25, cmap='Set1'))

# %% hidden=true
perplexity = 30
# X_tsne = get_tsne(perplexity=perplexity,filename_extra=filenameextra, overwrite=False)
X_tsne = curve_dict.get(perplexity).data[:,:-1]
print('X.shape : {}, X_tsne.shape : {}, perplexity : {}'.format(X.shape, X_tsne.shape,perplexity))

# %% [markdown]
# #### UMAP embedding
#
# [Basic UMAP Parameters — umap 0.3 documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html)

# %%

import umap

vdims = ['label']
kdims=['UMAP.0','UMAP.1']

X_embedd = X.values

filenameextra ='_icacomponents-{}'.format(ica_n_components)

### RANDOM SAMPPLING
randomize = np.random.choice(X.shape[0], size=100)
try:
    label=labels[randomize]
except NameError:
    label=None
# X_embedd = X_embedd[randomize,:]

overwrite=True
label= None

# neighbors = np.arange(5,X_embedd.shape[0]//6,X_embedd.shape[0]//20) # 5 to a quarter of the data each 1/8 o the data
neighbors     = [   100, 4000, 7000]
min_distances = (0.0, 0.5, 0.99)

print('X_embedd.shape : ',X_embedd.shape)
print('     neighbors : ',neighbors)
print(' min_distances : ',min_distances)

import itertools
print(list(itertools.product(neighbors,min_distances)))

def get_umap(neighbors, mindistances,filename_extra=filenameextra, label=None, overwrite=False):

    cached_umap = pathlib.Path('../models/umap_embedding_neighbors-{}_mindist-{}{}.npy'.format(neighbors,mindistances,filename_extra))
    if cached_umap.is_file():
        print(f'Loading: {cached_umap}')
        x_umap = np.load(cached_umap)
    else:
        print(f'file not found: {cached_umap} - Calculating!')
        if overwrite: 
            x_umap = umap.UMAP(n_neighbors=neighbors, min_dist = mindistances).fit_transform(X_embedd)
            np.save(cached_umap,x_umap)    
        else:
            print(f'overwrite set to {overwrite} stopping')
    print('X_embedd.shape : {}, x_umap.shape : {}, neighbors : {}, min_dist: {}'.format(X_embedd.shape, x_umap.shape,neighbors, mindistances))
    if label is not None:
        return hv.Scatter(np.hstack([x_umap,label[:,np.newaxis]]),vdims=vdims,kdims=kdims)
    else:
        return hv.Scatter(x_umap)



# %%
hv.extension('matplotlib')

curve_dict_2D = {(n,d):get_umap(n,d,label=label, overwrite=overwrite) for n in neighbors for d in min_distances}

gridspace = hv.GridSpace(curve_dict_2D, kdims=['neighbors, local > global structure', 'minimum distance in representation']).opts(hv.opts.Scatter(s=25,alpha=0.25),hv.opts.GridSpace(fig_inches=16))

# if labels.any(): 
#     gridspace.opts(hv.opts.Scatter(cmap=plt.cm.Spectral_r,c='label'))

# gridspace

# %%
gridspace

hv.extension('matplotlib')
# hv.extension('bokeh')

out = hds.dynspread(
hds.datashade(gridspace,
              aggregator=ds.count(),
              cmap=plt.cm.Spectral_r,
              x_sampling=0.25,
              y_sampling=0.25,
             ).\
opts(aspect=1,fig_size=50).opts(hv.opts.RGB(interpolation='bilinear')))
out

# save_plot(f'UMAP_gridspace_ICA_{ica_n_components}components', out_format='png',save=save_plots_bool)

# %%
neigh_mindist =  (4000, 0.99)

# print('X.shape : {}, X_umap.shape : {}, (n_neighbors,min_dist) : {}'.format(X.shape, X_umap.shape,neigh_mindist))
X_umap_scatter = get_umap(neighbors=neigh_mindist[0], mindistances=neigh_mindist[1],filename_extra=filenameextra, label=None, overwrite=True)

X_umap = curve_dict_2D.get(neigh_mindist).data

# %%
vdims = ['label']
kdims=['UMAP.0','UMAP.1']

hv.extension('matplotlib')
# hv.extension('bokeh')
hds.datashade(X_umap_scatter,
              aggregator=ds.count(),
              cmap=plt.cm.Spectral_r,
              x_sampling=0.4,
              y_sampling=0.3,
             ).\
opts(interpolation='bilinear',aspect=1,fig_size=200)
# opts(height=600,width=600)


# %%
a = hds.rasterize(X_umap_scatter,aggregator=ds.count(),dynamic=False,x_sampling=0.4, y_sampling=0.3).data.to_dataframe()
display(a.describe())

ax = a.plot.hist(bins=45)
ax.set_yscale('log')

# %% [markdown]
# ### Classification

# %% tags=[]
from sklearn import cluster
from sklearn import preprocessing

# data for classificatiom from different sources 
# X_classification = X_pca # PCA
# X_classification = S_ # ICA
# X_classification = X_tsne # tsne embedding
X_classification = X_umap # umap embedding

n_classification_features = X_classification.shape[1]

print('X_classification shape : ',X_classification.shape)

# %% tags=[]
##########################
# Scalers
##########
# scaler , preprocessing_type = preprocessing.StandardScaler().fit(X_classification)    , 'StandardScaler'
# scaler , preprocessing_type = preprocessing.MinMaxScaler().fit(X_classification) , 'MinMaxScaler'
scaler , preprocessing_type = preprocessing.RobustScaler().fit(X_classification) , 'RobustScaler'
# scaler, preprocessing_type = preprocessing.FunctionTransformer(lambda x:x), 'None'

# ##########################
# classifier = 'K-Means'
# n_clusters = 2
# # Kmeans estimator instance and Classify scaled data
# k_means = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(scaler.transform(X_classification))
# labels = k_means.labels_
# print('k_means.inertia_ : ',k_means.inertia_)

##########################
classifier = 'AgglomerativeClustering'
n_clusters = 3
aggclustering = cluster.AgglomerativeClustering(linkage='complete',
                                                affinity='l2',
                                                n_clusters=n_clusters).fit(scaler.transform(X_classification))
labels = aggclustering.labels_

##########################
# classifier = 'DBSCAN'
# dbscan = cluster.DBSCAN(eps=0.9, min_samples=5).fit(X_classification)
# labels = dbscan.labels_

##########################

# ##########
# classifier = 'HDBSCAN'
# import hdbscan
# hdbscan = hdbscan.HDBSCAN(
#             min_cluster_size=X_classification.shape[0]//2000,
#             min_samples=1,
#             cluster_selection_epsilon=0.75,
# #             allow_single_cluster=False,
#             )
# hdbscan.fit(scaler.transform(X_classification))
# labels = hdbscan.labels_

# %% tags=[]
##### Statistics
clust_stat_df = pd.DataFrame([{
         'size': labels[labels == val].size,
         'clAss_mean_fetures_delta': np.mean(np.max(X_classification[labels == val,:],axis=0)-np.min(X_classification[labels == val,:],axis=0)),
         'class_mean_features_std':np.max(X_classification[labels == val,:].std(axis=0)),
            }
            for val in np.unique(labels)],
          index=[val for val in np.unique(labels)]).sort_values('size',ascending=False)
print(clust_stat_df.shape[0])
display(clust_stat_df)
display(clust_stat_df[['size']].describe())

# %% code_folding=[] tags=[]
#####
# colors: relabelling the classes using the first centroids values

# calculate all the class centers in data space
y = X.groupby(labels).mean().values
# position of the data feature used to sort lables
feature_index = find_nearest(700)
# here the sorting index
centroids_sorting_index = np.argsort(y[:, feature_index])
# here the sorting labels, not the index!!
centroids_sorted_labels = np.argsort(centroids_sorting_index) 
# # use pd.Series.map(dict) di directly change values in place 
labels = pd.Series(labels).map(dict(zip(np.arange(n_clusters),centroids_sorted_labels))).values
print('index for label sort :',feature_index)
# print(' features y[:,index] :',y[:, feature_index])
# print(centroids_sorting_index)
# print(centroids_sorted_labels)
print(f'ind:y_feat  > new_index')
for i,yf,ni in zip(range(len(y[:, feature_index])),y[:, feature_index],centroids_sorted_labels):
    print(f'{i:3}:{yf:.5f} > {ni:>4}')


# %%
# holoview import breaks matplotlib inline 
# %matplotlib inline

# %% tags=[]
sns.set(style="ticks")

scatter_df = pd.DataFrame(scaler.transform(X_classification), columns = [f'feature_{x}' for x in range(n_classification_features)])
scatter_df['class'] = labels
scatter_df['class'] = scatter_df['class'].apply(lambda x: 'classs_{}'.format(x))

if X_classification.shape[1] > 2:
    g = sns.PairGrid(scatter_df,hue="class",height=4);
    # g = g.map_offdiag(sns.kdeplot,lw=1)
    g = g.map_offdiag(plt.scatter, s=0.1 , alpha=0.5);
    g;
else: 
    plt.figure(figsize=[8,8])
    # inverselly scale scatteplot size to clas size 
    classes , s = np.unique(scatter_df['class'],return_counts=True)
    min_size, max_size = 20, 100
    sizes = (((s-s.min())/(s.max()-s.min()))*(max_size-min_size))+min_size
    sns.scatterplot(x="feature_0", y="feature_1",hue="class", size='class', sizes=dict(zip(classes,sizes)), data=scatter_df, alpha=0.1)

# save_plot( f'Classification-scatter-features-n_clusters_{n_clusters}_classifier-{classifier}', out_format='png',save=save_plots_bool)

# %% tags=[]
hv.extension('bokeh')

print(np.unique(labels))
outdf_gdf.loc[spectral_df_nona_index,'R'] = labels
outdf_gdf.loc[outdf_gdf['R'] != 11,'R'] = np.nan

outdf_gdf[['x','y','R']].hvplot.scatter(x='x',y='y',c='R',
                    rasterize=True,aggregator='mean',dynamic=True,
                    x_sampling=2,y_sampling=1,cmap='rainbow',cnorm='eq_hist'
                    ).opts(height=600,width=1200,alpha=1)#*background.opts(cmap='Gray',alpha=.65)

# %% [markdown] heading_collapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ####  AgglomerativeClustering plot

# %% hidden=true
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_array = linkage(aggclustering.children_)

# %% hidden=true tags=[]
plt.figure(figsize=[10,8])
dendrogram(linkage_array,
    p=40,  # show only the last p merged clusters
    truncate_mode='lastp',  # show only the last p merged clusters
    orientation='top',
    no_labels=False,
    show_contracted=True,  # to get a distribution impression in truncated branches
    leaf_rotation=90.,
    leaf_font_size=16.,
);
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
# plt.ylim([3000,6200])
# plt.xlim([-0.5,10.5])
# plt.hlines(4500,0,100,color='red')
plt.show()
save_plot(f'{classifier}_dendrogram.png', out_format='png',save=save_plots_bool)


# %% [markdown]
# Load [scipy.cluster.hierarchy.inconsistent (SciPy v1.10.0)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.inconsistent.html).
#
# Calculate inconsistency statistics on a linkage matrix.
#
# This function behaves similarly to the MATLAB(TM) inconsistent function.
#
#     Y = inconsistent(Z) returns the inconsistency coefficient for each link of the hierarchical cluster tree Z generated by the linkage function. inconsistent calculates the inconsistency coefficient for each link by comparing its height with the average height of other links at the same level of the hierarchy. The larger the coefficient, the greater the difference between the objects connected by the link. For more information, see Algorithms.

# %% hidden=true
from scipy.cluster.hierarchy import inconsistent
depth = 3
incons = inconsistent(linkage_array, depth)
incons[-20:]


# %% hidden=true tags=[]
# see [SciPy Hierarchical Clustering and Dendrogram Tutorial | Jörn's Blog](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

plt.figure(figsize=[10,8])
fancy_dendrogram(
    linkage_array,
    truncate_mode='lastp',
    p=20,
    leaf_rotation=90.,
    leaf_font_size=30.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.ylim([2000,6500])
plt.show()

# %% [markdown] tags=[]
# ####  Classification Vis

# %% tags=[]
hv.extension('bokeh')

vdims = ['component']
kdims=['wavelenght']
width=1000
height=500

cm = plt.cm.Spectral_r
colors = cm(np.linspace(0,1,len(np.unique(labels))))
matplotlib.colors.LinearSegmentedColormap.from_list('Spectral',colors , N=len(np.unique(labels)))
cm_cycle = hv.Cycle([cm(c) for c in np.linspace(0,1,len(np.unique(labels)))])
hv.Cycle.default_cycles['default_colors'] = cm_cycle

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in X.groupby(labels).mean().iterrows()}

overlay_dict['Mean'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.mean()
                    ),vdims=vdims,kdims=kdims).opts(line_width=0.5,line_dash='dashed',color='black',alpha=1).relabel('mean')

hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4,color=cm_cycle),
                hv.opts.Curve('mean',color='black'),
                hv.opts.NdOverlay(width=width,height=height)
                )

# %% tags=[]
hv.extension('matplotlib')

hv.Cycle.default_cycles['default_colors'] = hv.Cycle([cm(c) for c in np.linspace(0,255,len(np.unique(labels)))])

vdims = ['component']
kdims=['wavelenght']

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in X.groupby(labels).mean().iterrows()}

overlay_dict['Mean'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.mean()
                    ),vdims=vdims,kdims=kdims).relabel('mean')

overlay_dict['Median'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.median()
                    ),vdims=vdims,kdims=kdims).relabel('median')

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.Curve('mean',linewidth=6,linestyle=':',color='black',alpha =0.5),
                hv.opts.Curve('median',linewidth=3,color='black',alpha =0.5),
                hv.opts.NdOverlay(fig_size=600, aspect=2)
                )

out

# # hv.save(out,out_figure_path / f'Spectral-centroids_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)

# %% tags=[]
## normalised to global mean

hv.extension('matplotlib')

hv.Cycle.default_cycles['default_colors'] = hv.Cycle([cm(c) for c in np.linspace(0,255,len(np.unique(labels)))])

vdims = ['component']
kdims=['wavelenght']

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in (X.groupby(labels).mean()/spectral_df.median()).iterrows()}

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.NdOverlay(fig_size=600, aspect=2)
                )

out

# # hv.save(out,out_figure_path / f'Spectral-centroids_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)

# %% tags=[]
hv.extension('bokeh')


# %% tags=[]
outdf_gdf['labels'] = np.nan
outdf_gdf.loc[spectral_df_nona_index,'labels'] = labels

# %% tags=[]
# data layer  on top, transparent 
# basemap on the bottom, opaque.

hv.extension('bokeh')

out = background.opts(cmap='Gray')*df_shader(labels,cmap=cm).opts(height=600,width=1000,alpha=0.7)
out

# hv.save(out,out_figure_path / f'Classification-map_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)


# %% [markdown]
# The following uses [hvPlot 0.8.2](https://hvplot.holoviz.org/). 
#
# From the documentation : 
#
#
# A familiar and high-level API for data exploration and visualization hvPlot diagram
#
# <img src="https://hvplot.holoviz.org/assets/diagram.svg" width="600">
#
# `.hvplot()` is a powerful and interactive Pandas-like `.plot()` API
#
# By replacing `.plot()` with `.hvplot()` you get an interactive figure. Try it out below!

# %% tags=[]
# basemap on top , transparent 
# data layer on the bottom, opaque.

outdf_gdf[['x','y','labels']].hvplot.scatter(x='x',y='y',c='labels',
                    rasterize=True,aggregator='mean',dynamic=False,
                    x_sampling=1,y_sampling=1,cmap='Spectral_r',cnorm='eq_hist'
                    ).opts(height=500,width=1000,alpha=1)*\
background.opts(cmap='Gray',alpha=0.7)

# %% [markdown]
# Load [sklearn.metrics.silhouette_score (scikit-learn 1.2.0)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
#
# Compute the mean Silhouette Coefficient of all samples.
#
# The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. 
#
# The Silhouette Coefficient for a sample is $(b - a) / max(a, b)$. To clarify, $b$ is the distance between $a$ sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is $2 <= n_labels <= n_samples - 1$.
#
# This function returns the mean Silhouette Coefficient over all samples. To obtain the values for each sample, use silhouette_samples.
#
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

# %%
from sklearn import metrics
metrics.silhouette_score(X_classification, labels, metric='euclidean')

# %% [markdown]
# not that bad actually...

# %%
