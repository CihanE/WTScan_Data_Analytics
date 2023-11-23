##Importing Libraries
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creating functions

def downscale(raster,scale=0.1):
    """Downscales the raster data
        
        Parameters:
        raster object 
        scale factor (float)
        
        Returns:
        map array (np array)
        transform function (rasterio.transform)
        
    """
    
    #Create map array
    data = raster.read(
        out_shape=(
            raster.count,
            int(raster.height * scale),
            int(raster.width * scale)
            ),
        resampling=Resampling.bilinear
        )
    
    # scale image transform
    transform = raster.transform * raster.transform.scale(
        (raster.width / data.shape[-1]),
        (raster.height / data.shape[-2])
    )
    
    return data, transform


def write_map(raster, data, transform, path):
    """Writes current raster to a geotiff file 
        
        Parameters:
        raster object
        data (np array)
        transform (rasterio.transform)
        path (directory + filename)
        
    """
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=1,
        dtype=data.dtype,
        crs=raster.crs,
        transform=transform,
    ) as dst:
        dst.write(data)    
    
    return print("{0} is saved".format(path)) 

def map_plot_country(data, title, size = (10, 5)):
    """Plot maps with color legend

        Parameters:
        data (np array)
        size (int, int) tuple
        title (str)
        
        """
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(
        title,
        fontsize = 16)

    map = ax.imshow(data[0], cmap='terrain')

    # add legend
    colorbar = fig.colorbar(map, ax=ax)
    plt.axis('off')
    #plt.show()
    return plt

##Reading World City Library
world_city = pd.read_csv("worldcities.csv")

not_found = ["XGZ", "GIB", "XKS", "MCO", "FLK", "VAT", "XSV", "SGS", "XWB", "GUM", "BLM"]
world_city = world_city[~world_city.iso3.isin(not_found)]

#Creating an array for country codes
country_codes = world_city.iso3.unique()

#Extracting 100 m weibull k parameter from globalwindatlas.info

for code in country_codes:
    
    #url for geotiff file from global wind atlas
    url = "https://globalwindatlas.info/api/gis/country/{0}/combined-Weibull-k/100".format(code)
    
    #path to save file
    path = '.\weibull_k_100m_map\{0}_weibull_k_100.tif'.format(code)
    
    with rasterio.open(url) as raster:
            #downscaling the map
            rst, trans= downscale(raster)
                      
            #writing downscaled map
            write_map(raster, rst, trans, path) 

#Extracting 100 m weibull a parameter from globalwindatlas.info

for code in country_codes:
    
    #url for geotiff file from global wind atlas
    url = "https://globalwindatlas.info/api/gis/country/{0}/combined-Weibull-A/100".format(code)
    
    #path to save file
    path = '.\weibull_a_100m_map\{0}_weibull_a_100.tif'.format(code)
    
    with rasterio.open(url) as raster:
            #downscaling the map
            rst, trans= downscale(raster)
                      
            #writing downscaled map
            write_map(raster, rst, trans, path) 

#Extracting 100 m wind country maps from globalwindatlas.info

for code in country_codes:
    
    #url for geotiff file from global wind atlas
    url = "https://globalwindatlas.info/api/gis/country/{0}/wind-speed/100".format(code)
    
    #path to save file
    path = '.\wind_100m_map\{0}_wind_speed_100.tif'.format(code)
    
    with rasterio.open(url) as raster:
            #downscaling the map
            rst, trans= downscale(raster)
                      
            #writing downscaled map
            write_map(raster, rst, trans, path)  

#Saving 100 m wind country maps as png

for code in country_codes:
    
    #path for geotiff file
    map_path = '.\wind_100m_map\{0}_wind_speed_100.tif'.format(code)
    #path to save file
    path = '.\Wind_Speed_Map_Images\{0}_wind_speed_100.png'.format(code)
    
    title = "{0} Wind Speed at 100 M elevation (m/s)".format(code)
    
    with rasterio.open(map_path) as raster:
        data = raster.read()
        plt = map_plot_country(data,title)
        plt.savefig(path, bbox_inches='tight')
