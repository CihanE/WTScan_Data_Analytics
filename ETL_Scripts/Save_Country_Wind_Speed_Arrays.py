
##Importing Libraries
import rasterio
import pickle
import pandas as pd
import numpy as np

df_weib = pd.read_csv("weibull_df.csv")
country_codes = df_weib.country_code.unique()

all_country_ws_dict = {}
for code in country_codes:
    #path for geotiff file
    map_path = '.\wind_100m_map\{0}_wind_speed_100.tif'.format(code)
    with rasterio.open(map_path) as raster:
        data = raster.read()
    all_country_ws_dict[code] = data[0]

with open('all_country_ws.pkl', 'wb') as file:
    pickle.dump(all_country_ws_dict, file)
    print('dictionary saved successfully to file')