import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

Wt = pd.read_csv('C:/Users/ghane/Projects/WindTurbineProject/supply__wind_turbine_library.csv')
Wt = Wt[~Wt['manufacturer'].isin(['aerodyn', 'IEA'])]
Wt = Wt[Wt['has_power_curve'] == True].reset_index()
print(Wt.shape)

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

def reg_model_builder(arr_w, arr_p):
    """Receives:power curve speed(list), 
                power curve power output(list)
        Result: DecisionTreeRegressionModel for PowerCurve
                poly_transformer
                wind_speed_arr
                power_output_arr
    """
    X = arr_w + [arr_w[-1]+0.1] + [arr_w[-1]+5]
    y = arr_p + [0] + [0]
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(np.array(X).reshape(-1,1))
    reg_model = DecisionTreeRegressor()
    reg_model.fit(poly_features, y)
    preds= reg_model.predict(poly_features)
    return reg_model, poly, X, preds

def AEP_calc(reg_model, poly, ref_w):
    """Receives:power curve speed(list), 
                power curve power output(list)
                reference annual wind distribution with hourly spans(list) 
        Result: Annual Energy production with %95 production availability rate(int)"""
    X_pred = poly.fit_transform(np.array(ref_w).reshape(-1,1))
    preds= reg_model.predict(X_pred)
    AEP = np.sum(preds)*0.95

    return int(AEP/1000)

def CF_calc(AEP, nominal_power):
    """Calculates Capacity Factor
        
        Input:
        Annual energy production
        Nominal Power
        
        Result:
        Capacity Factor
    """
    return AEP * 1000 / 8760 / nominal_power

def annual_wind_dist_arr(weibull_a, weibull_k):
    w_dist_arr = weibull_min.rvs(weibull_k, scale=weibull_a, size=8760, random_state=42)
    return w_dist_arr

def wind_profile(weibull_a, weibull_k):
    wind_x = np.linspace(weibull_min.ppf(0.01, weibull_k, scale=weibull_a),
                weibull_min.ppf(0.99, weibull_k, scale=weibull_a), 100)
    wind_y = weibull_min.pdf(wind_x, weibull_k, scale=weibull_a)
    return wind_x, wind_y

def Pwr_Curve_plotter(wt_name, wind_speed, power_out):
    fig,ax = plt.subplots(figsize=(6,4))
    sns.lineplot(x=wind_speed, y=power_out, color='red').set(title= wt_name + 'Power Curve', 
                                            xlabel='Wind Speed (m/s)', 
                                            ylabel='Power Output (kW)')
    plt.show()
    return
