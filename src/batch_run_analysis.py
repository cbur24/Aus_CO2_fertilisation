#!/usr/bin/env python
# coding: utf-8

# Attributing trends in annually summed NDVI and calculating
# the influence of CO2 on those trends

import os
import sys
import dask
import scipy
import warnings
import dask.array
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from dask import delayed
from odc.geo.xr import assign_crs

import sys
sys.path.append('/g/data/os22/chad_tmp/Aus_CO2_fertilisation/src')
from analysis import _preprocess, regression_attribution, robust_trends, calculate_beta

sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')

from _utils import start_local_dask
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import round_coords

## !!!!!----variables for script-------!!!!!!!!!!
n_workers=102
memory_limit='300GiB'
modelling_vars=['co2', 'srad', 'rain', 'tavg', 'vpd']
results_path = '/g/data/os22/chad_tmp/Aus_CO2_fertilisation/results/tiles/AusEFlux_GPP/'
template_path='/g/data/os22/chad_tmp/Aus_CO2_fertilisation/data/templates/'
model_var='GPP' #NDVI GPP
model_types = ['delta_slope', 'PLS', 'ML']
# ------------------------------------------------

n = os.getenv('TILENAME')

#define meta function
def attribution_etal(
    n,
    results_path,
    template_path,
    modelling_vars,
    model_var,
    model_types
):
    print('Working on tile', n)

    #open data
    d_path = f'/g/data/os22/chad_tmp/Aus_CO2_fertilisation/data/tiles/{model_var}_{n}.nc'
    dd_path = f'/g/data/os22/chad_tmp/Aus_CO2_fertilisation/data/tiles/COVARS_{n}.nc'
    ss_path = f'/g/data/os22/chad_tmp/Aus_CO2_fertilisation/data/tiles/SS_{n}.nc'

    d = assign_crs(xr.open_dataset(d_path)[model_var], crs='epsg:4326')
    dd = assign_crs(xr.open_dataset(dd_path), crs='epsg:4326')
    ss = assign_crs(xr.open_dataset(ss_path)['NDVI'], crs='epsg:4326')
    ss.name = 'NDVI'
    
    # transform the data and return all the objects we need. This code smooth and
    # interpolates the data, then stacks the pixels into a spatial index
    d, dd, ss, Y, idx_all_nan, nan_mask, shape = _preprocess(d, dd, ss)

    ## ----Find the trends in annually summed NDVI/GPP--------------
    if os.path.exists(f'{results_path}{model_var}_trends_perpixel_{n}.nc'):
        pass
    else:
        results=[]
        for i in range(shape[1]): #loop through all spatial indexes.
        
            #select pixel
            data = Y.isel(spatial=i)
            
            # First, check if spatial index has data. If its one of 
            # the all-NaN indexes then return empty template
            if i in idx_all_nan:
                xx = xr.DataArray(name='slope', data=np.nan,
                                  coords={'longitude':[1], 'latitude':[1]},
                                  dims=['longitude','latitude']).to_dataset()
                xx['p_value'] = np.nan
                # xx = trend_template.copy() #use our template    
                xx['latitude'] = [data.latitude.values.item()] #update coords
                xx['longitude'] = [data.longitude.values.item()]
            
            else:
                #run the trend function
                xx = robust_trends(data)
        
            #append results, either data or all-zeros
            results.append(xx)
        
        # bring into memory and combine
        trends = dask.compute(results)[0]
        
        trends = xr.combine_by_coords(trends, coords='minimal').astype('float32')
        # assign crs and export
        trends = assign_crs(trends, crs='EPSG:4326')
        trends.transpose().to_netcdf(f'{results_path}{model_var}_trends_perpixel_{n}.nc')

    # -----calculate CO2 beta coefficients iterate-------------------------------
    if os.path.exists(f'{results_path}beta_coefficient_perpixel_{n}.nc'):
        pass
    
    else:
        beta = []    
        for i in range(shape[1]): #loop through all spatial indexes.
            #select pixel
            data = Y.isel(spatial=i)
            
            lat = data.latitude.item()
            lon = data.longitude.item()
            
            fi = calculate_beta(data,
                               X=dd.sel(latitude=lat, longitude=lon),
                               model_var=model_var,
                               modelling_vars=modelling_vars,
                              )
            beta.append(fi)
        
        beta = dask.compute(beta)[0]
        beta = xr.combine_by_coords(beta).astype('float32')
        beta.to_netcdf(f'{results_path}beta_coefficient_perpixel_{n}.nc')

    # -----regression attribution iterate-------------------------------
    for model_type in model_types: 
        
        if os.path.exists(f'{results_path}attribution_{model_type}_perpixel_{n}.nc'):
            pass
        else:
            
            regress_template = xr.open_dataset(f'{template_path}template_{model_type}.nc').sel(feature=modelling_vars)
        
            p_attribution = []    
            for i in range(shape[1]): #loop through all spatial indexes.
                #select pixel
                data = Y.isel(spatial=i)
                
                lat = data.latitude.item()
                lon = data.longitude.item()
                
                fi = regression_attribution(data,
                                   X=dd.sel(latitude=lat, longitude=lon),
                                   template=regress_template,
                                   model_type=model_type,
                                   model_var=model_var,
                                   modelling_vars=modelling_vars,
                                  )
                p_attribution.append(fi)
            
            p_attribution = dask.compute(p_attribution)[0]
            p_attribution = xr.combine_by_coords(p_attribution).astype('float32')
            p_attribution.to_netcdf(f'{results_path}attribution_{model_type}_perpixel_{n}.nc')


#run function
if __name__ == '__main__':
    #start a dask client
    start_local_dask(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit
                    )

    # Run meta function
    attribution_etal(
        n=n,
        results_path=results_path,
        template_path=template_path,
        model_var=model_var,
        model_types=model_types,
        modelling_vars=modelling_vars,
    )

