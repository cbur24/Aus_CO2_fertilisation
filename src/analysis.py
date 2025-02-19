
'''
Atrribution of NDVI trends and quantifying 
the CO2 fertilisation effect
'''

import os
import sys
import math
import shap
import dask
import scipy
import numpy as np
import xarray as xr
import pandas as pd
import pingouin as pg
from scipy import stats
from datetime import datetime
from scipy.signal import detrend
from odc.geo.xr import assign_crs
from collections import namedtuple

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import theilslopes, kendalltau

import pymannkendall as mk
from pymannkendall.pymannkendall import __preprocessing, __missing_values_analysis, __mk_score, __variance_s, __z_score, __p_value, sens_slope

def _preprocess(d, dd, ss):
    """
    d = ndvi data
    dd = covariables for regression modelling.
    ss = soil signal xarray dataarray
    
    """
    ### --------Find NaNs---------
    ndvi_nan_mask = np.isnan(d).sum('time') >= len(d.time) / 10
    clim_nan_mask = dd[['rain','vpd','tavg','srad', 'cwd']].to_array().isnull().any('variable')
    clim_nan_mask = (clim_nan_mask.sum('time')>0)
    soil_nan_mask = np.isnan(ss)
    nan_mask = (clim_nan_mask | ndvi_nan_mask | soil_nan_mask)

    d = d.where(~nan_mask)
    dd = dd.where(~nan_mask)
    ss = ss.where(~nan_mask)
    
    ### --------Aggregate to annual sums--------
    if d.name=='NDVI':
        d = d - ss #remove soil signal
    d = d.groupby('time.year').sum()
    d = d.where(~nan_mask) #remask after summing
    
    #stack spatial indexes, this makes it easy to loop through data
    y_stack = d.stack(spatial=('latitude', 'longitude'))
    Y = y_stack.transpose('year', 'spatial')
    
    # We also need the shape of the stacked array
    shape = y_stack.values.shape
    
    # find spatial indexes where values are mostly NaN (mostly land-sea mask)
    # This is where the nan_mask we created earlier = True
    idx_all_nan = np.where(nan_mask.stack(spatial=('latitude', 'longitude'))==True)[0]

    return d, dd, ss, Y, idx_all_nan, nan_mask, shape


def mk_with_slopes(x_old, alpha = 0.05):
    """
    This function checks the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987).
    This was modified from the "pymannkendall" library to return fewer statistics which makes
    it a little more robust.
    
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (0.05 default)
    Output:
        p: p-value of the significance test
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
        
    """
    res = namedtuple('Mann_Kendall_Test', ['p','slope','intercept'])
    x, c = __preprocessing(x_old)
    x, n = __missing_values_analysis(x, method = 'skip')
    
    s = __mk_score(x, n)
    var_s = __variance_s(x, n)
    
    z = __z_score(s, var_s)
    p, h, trend = __p_value(z, alpha)
    slope, intercept = sens_slope(x_old)

    return res(p, slope, intercept)

@dask.delayed
def robust_trends(ds):
    """
    
    """
    slopes=[]
    p_values=[]
    out = xr.apply_ufunc(mk_with_slopes,
                  ds,
                  input_core_dims=[["year"]],
                  output_core_dims=[[],[],[],],
                  vectorize=True)
    
    #grab the slopes and p-value
    p = out[0].rename('p_value')
    s = out[1].rename('slope')
    # i = out[2].rename('intercept')

    slopes.append(s)
    p_values.append(p)
    # intercept.append(i)

    #merge all the variables
    slopes_xr = xr.merge(slopes)
    p_values_xr = xr.merge(p_values)
    # intercept_xr = xr.merge(intercept)

    #add back spatial data
    out_ds = xr.merge([slopes_xr,p_values_xr])
    
    try:
        out_ds = out_ds.drop_vars('spatial').astype('float32')
    
    except:
        out_ds = out_ds.astype('float32')
    
    lat = ds.latitude.item()
    lon = ds.longitude.item()
    # out_ds.assign_coords(longitude=lon, latitude=lat)
    out_ds = out_ds.expand_dims(longitude=[lon], latitude=[lat])
    return out_ds

@dask.delayed
def regression_attribution(
    target, 
    X, #covariables,
    template,
    model_var='NDVI',
    model_type='PLS',
    rolling=3,
    modelling_vars=['srad','co2','rain','tavg','vpd','cwd']
):
    """
    Attribute trends in NDVI/GPP using regression models
    
    returns:
    --------
    An xarray dataset with regression coefficients, along with 
    slope, p-values, and r2 values for actual and predicted.
    """
    
    #-------Get phenometrics and covariables in the same frame-------
    #check if this is a no-data pixel
    if np.isnan(target.isel(year=0).values.item()):
        fi = template.copy() #use our template    
        fi['latitude'] = [target.latitude.values.item()] #update coords
        fi['longitude'] = [target.longitude.values.item()]

    else:
        # df = target.squeeze().to_dataframe()
        lat = target.latitude.item()
        lon = target.longitude.item()

        # summarise climate data.
        co2 = X['co2'].groupby('time.year').mean()
        rain = X['rain'].groupby('time.year').sum()
        X = X.drop_vars(['rain', 'co2']).groupby('time.year').mean()
        X = xr.merge([co2.to_dataset(), rain.to_dataset(), X])

        #now add our target to the covars- so we have a neat object to work with
        X[model_var] = target
        
        #--------------- modelling---------------------------------------------------
        # fit rolling annuals to remove some of the IAV (controllable)
        # df = X.rolling(year=rolling, min_periods=rolling).mean().to_dataframe().dropna()
        X['year'] = [datetime.strptime(f'{int(y)} 1', '%Y %j') for y in X['year']]
        df = X.resample(year=f'{rolling}YE').mean().to_dataframe().dropna()

        #fit a model with all vars
        x = df[modelling_vars]
        y = df[model_var]        
        
        if model_type=='ML':
            #fit a RF model with all vars
            rf = RandomForestRegressor(n_estimators=100).fit(x, y)
            
            # use SHAP to extract importance
            explainer = shap.Explainer(rf)
            shap_values = explainer(x)
            
            #get SHAP values into a neat DF
            df_shap = pd.DataFrame(data=shap_values.values,columns=x.columns)
            df_fi = pd.DataFrame(columns=['feature','importance'])
            for col in df_shap.columns:
                importance = df_shap[col].abs().mean()
                df_fi.loc[len(df_fi)] = [col,importance]
    
            # Tidy up into dataset
            fi = df_fi.set_index('feature').to_xarray().expand_dims(latitude=[lat],longitude=[lon])

        if model_type=='PLS':
            lr = PLSRegression().fit(x, y)
            prediction = lr.predict(x)
            r2_all = r2_score(y, prediction)
            
            # Find the robust slope of actual
            result_actual = mk.original_test(y, alpha=0.05)
            p_actual = result_actual.p
            s_actual = result_actual.slope
            i_actual = result_actual.intercept
            
            #calculate slope of predicted variable with all params
            result_prediction = mk.original_test(prediction, alpha=0.05)
            p_prediction = result_prediction.p
            s_prediction = result_prediction.slope
            i_prediction = result_prediction.intercept
        
            #get the PLS coefficients
            fi = pd.Series(dict(zip(list(x.columns), list(lr.coef_.reshape(len(x.columns)))))).to_frame()
            fi = fi.rename({0:'PLS_coefficent'},axis=1)
            fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature')
            
            # create tidy df with all stats
            # fi['phenometric'] = 'vPOS'
            fi['slope_actual'] = s_actual
            fi['slope_modelled'] = s_prediction
            fi['p_actual'] = p_actual
            fi['p_modelled'] = p_prediction
            fi['i_actual'] = i_actual
            fi['i_modelled'] = i_prediction
            fi['r2'] = r2_all
    
            fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])

        if model_type=='delta_slope':
            lr = PLSRegression().fit(x, y)
            prediction = lr.predict(x)
            r2_all = r2_score(y, prediction)
    
            # Find the robust slope of actual
            result_actual = mk.original_test(y, alpha=0.05) #
            p_actual = result_actual.p
            s_actual = result_actual.slope
            i_actual = result_actual.intercept
            
            #calculate slope of predicted variable with all params
            result_prediction = mk.original_test(prediction, alpha=0.05) #
            p_prediction = result_prediction.p
            s_prediction = result_prediction.slope
            i_prediction = result_prediction.intercept
    
            # now fit a model without a given variable
            # and calculate the slope of the phenometric
            r_delta={}
            s_delta={}
            for v in modelling_vars:
                #set variable of interest as a constant value 
                constant = x[v].iloc[0:1].mean() #average of first 5 years
                xx = x.drop(v, axis=1)
                xx[v] = constant
            
                #model and determine slope
                lrr = PLSRegression().fit(xx, y)
                pred = lrr.predict(xx)
                r2 = r2_score(y, pred)
                
                result_p = mk.original_test(pred, alpha=0.1)
                s_p = result_p.slope
    
                #determine the eucliden distance between
                #modelled slope and actual slope (and r2)
                s_delta[v] = math.dist((s_prediction,), (s_p,))
                r_delta[v] = math.dist((r2_all,), (r2,))
    
            #determine most important feature
            s_delta = pd.Series(s_delta)
            r_delta = pd.Series(r_delta)
            fi = pd.concat([s_delta, r_delta], axis=1).rename({0:'delta_slope', 1:'delta_r2'}, axis=1)
            # fi = fi.loc[[fi['delta_slope'].idxmax()]]
            fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature')
    
            #create tidy df
            fi['slope_actual'] = s_actual
            fi['slope_modelled'] = s_prediction
            fi['p_actual'] = p_actual
            fi['p_modelled'] = p_prediction
            fi['i_actual'] = i_actual
            fi['i_modelled'] = i_prediction
            fi['r2'] = r2_all
            fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])
            
    return fi

@dask.delayed
def calculate_beta(
    target,
    X,
    model_var='NDVI',
    modelling_vars=['srad','co2','rain','tavg','vpd','cwd']
):
    """
    Following Zhan et al. (2024):
    https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JG007910

    Steps:
    1. Detrend NDVI, add back median value.
    2. Detrend climate variables (T, VPD, rain, SW, CWD), add back median value.
    3. Train model: NDVI-detrend = f(climate detrend)
    4. Predict NDVI with original climate data using model
    5. NDVI residual = Actual NDVI - predicted NDVI
    6. Beta = linear trend of NDVI residual vs CO2.
    
    """
    #check if this is a no-data pixel
    if np.isnan(target.isel(year=0).values.item()):
        #return xarray with NaNs
        df = pd.DataFrame({'beta':np.nan, 'beta_relative':np.nan, 'pvalue':np.nan}, index=[0])
        ds = df.to_xarray().squeeze().expand_dims(latitude=[-25],longitude=[25]).drop_vars('index')
        # update coords
        ds['latitude'] = [target.latitude.values.item()] 
        ds['longitude'] = [target.longitude.values.item()]

    else:
        lat = target.latitude.item()
        lon = target.longitude.item()
    
        # summarise climate data.
        co2 = X['co2'].groupby('time.year').mean()
        rain = X['rain'].groupby('time.year').sum()
        X = X.drop_vars(['rain', 'co2']).groupby('time.year').mean()
        X = xr.merge([co2.to_dataset(), rain.to_dataset(), X])
    
        #now add our target to the covars
        X[model_var] = target
        
        #--------------- modelling---------------------------------------------------
        # convert to df
        df = X.to_dataframe().dropna()
    
        #step 1 & 2
        for v in [model_var]+modelling_vars:
            df[f'{v}_detrend'] = detrend(df[v])
            df[f'{v}_detrend'] = df[f'{v}_detrend'] + df[v].median()
    
        #step 3
        x = df[[x+'_detrend' for x in modelling_vars]]
        xx = df[modelling_vars]
        y = df[model_var+'_detrend']
        rf = RandomForestRegressor(n_estimators=100).fit(x, y)
        
        # step 4
        ## predict using original climate data (add suffix to trick scikit learn)
        df[model_var+'_predict'] = rf.predict(xx.add_suffix('_detrend'))
    
        #step 5
        df[model_var+'_residual'] = df[model_var] - df[model_var+'_predict']
        df[model_var+'_residual_percent'] = df[model_var+'_residual']/df[model_var][0:3].mean()
    
        #step 6
        #find robust regression slope
        beta  = theilslopes(y=df[model_var+'_residual'], x=df['co2']).slope
        beta_relative = theilslopes(y=df[model_var+'_residual_percent'], x=df['co2']).slope * 100 * 100
        pvalue = kendalltau(y=df[model_var+'_residual'], x=df['co2']).pvalue

        #export values as xarray
        dff = pd.DataFrame({'beta':beta, 'beta_relative':beta_relative, 'pvalue':pvalue}, index=[0])
        ds = dff.to_xarray().squeeze().drop_vars('index')
        ds = ds.expand_dims(latitude=[lat],longitude=[lon])
          
    return ds.astype('float32')













