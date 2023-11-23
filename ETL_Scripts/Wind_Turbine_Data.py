import pandas as pd
import numpy as np
import pickle

#Read Wind Turbine Dataset
Wt = pd.read_csv('C:/Users/ghane/Projects/WindTurbineProject/supply__wind_turbine_library.csv')
Wt = Wt[~Wt['manufacturer'].isin(['aerodyn', 'IEA'])]
Wt = Wt[Wt['has_power_curve'] == True].reset_index()

#Generating power categories for the dataset
group_names =['0-2 MW', '2-3 MW', '3-4 MW', '4-6 MW','6-8 MW', '8-10 MW']
ranges = [0,2000,3000,4000,6000,8000,np.inf]
Wt['power_cat'] = pd.cut(Wt['nominal_power'], bins=ranges, labels=group_names)

#Generating Wind Speed Array of Power Curves
curves_wind_speed = Wt['power_curve_wind_speeds'].str.strip('[]')
curves_wind_speed = curves_wind_speed.str.replace(' ', '')
curves_wind_speed = curves_wind_speed.str.split(',')
speed_arr = [[float(i) for i in val] for val in curves_wind_speed]

#Generating Power Output Array of Power Curves
curve_power = Wt['power_curve_values'].str.strip('[]')
curve_power = curve_power.str.replace(' ', '')
curve_power = curve_power.str.split(',')
pwr_arr = [[float(i) for i in val] for val in curve_power]

#Save power curve arrays in pickle format
power_curve_dict = {}
power_curve_dict["wind_speed"] = speed_arr
power_curve_dict["power"] = pwr_arr

with open('power_curve.pkl', 'wb') as file:
    pickle.dump(power_curve_dict, file)
    print('dictionary saved successfully to file')

#Save Windturbine data frame
Wt.to_csv("Wind_Turbine_Database.csv")
