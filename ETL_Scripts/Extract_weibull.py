import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np


def city_map(lat, lng, raster):
    '''
    Isolates city data on country map

    Input:
    Latitude, Longitude, Raster Data

    Output:
    City Data in array format
    '''
    left = lng - 0.4
    bottom = lat - 0.4
    right = lng + 0.4
    top = lat + 0.4
    rst = raster.read(1, window=from_bounds(left, bottom , right, top, raster.transform))
    return rst
    


##Reading World City Library
world_city = pd.read_csv("C:/Users/ghane/Projects/Wind_Turbine_Dashboard/Data/world_wind.csv")

##Initialize empty arrays for weibull data 
avg_wind_speed = []
weibull_a_list = []
weibull_k_list = []

for index, row in world_city.iterrows():
    lat = row["lat"]
    lng = row["lng"]
    city = row["city_ascii"]
    country_code = row["iso3"]
    #path to map files
    path = '.\wind_100m_map\{0}_wind_speed_100.tif'.format(country_code)
    path_weibull_a = '.\weibull_a_100m_map\{0}_weibull_a_100.tif'.format(country_code)
    path_weibull_k = '.\weibull_k_100m_map\{0}_weibull_k_100.tif'.format(country_code)
    
    with rasterio.open(path) as raster:
        rst = city_map(lat, lng, raster)
    
    ws_lower = np.quantile(rst, 0.60)
    ws_upper = np.quantile(rst, 0.80)
    mask_pre = np.ma.masked_where(rst < ws_lower, rst)
    mask = np.ma.masked_where(mask_pre > ws_upper, mask_pre)
    avg_wind = np.mean(mask)
    
    with rasterio.open(path_weibull_a) as raster_wa:
        rst_wa = city_map(lat, lng, raster_wa)

    masked_wa = np.ma.masked_where(np.ma.getmask(mask), rst_wa)
    weib_a = np.mean(masked_wa)

    with rasterio.open(path_weibull_k) as raster_wk:
        rst_wk = city_map(lat, lng, raster_wk)

    masked_wk = np.ma.masked_where(np.ma.getmask(mask), rst_wk)
    weib_k = np.mean(masked_wk)
    
    avg_wind_speed.append(avg_wind)
    weibull_a_list.append(weib_a)
    weibull_k_list.append(weib_k)

weibull_df = pd.DataFrame(data ={"avg_wind" : avg_wind_speed, 
                                 "weibull_a" : weibull_a_list, 
                                 "weibull_k" : weibull_k_list
                                }
                         )

print(weibull_df.head())

world_city["avg_wind"] = weibull_df["avg_wind"]
world_city.to_csv("world_wind.csv",index=False)

