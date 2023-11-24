import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures


page_title = "WTScan Wind Turbine Analytics"
head = ":cyclone: WTScan"
main_title = ":blue[Global Wind Turbine Analytics App]"
author = "**Cihan Erturk , Data Scientist / Senior R&D Engineer**"
intro = "WTScan is a comprehensive data application that analyzes global wind speeds and forecasts wind turbine energy production levels. The app uses empirical Weibull wind models to forecast annual energy generation and capacity factors. It also provides information on the wind turbines with the highest energy yields for various regions of the world. With WTScan, you can easily access and analyze data on wind energy production and make informed decisions."

def display_map(df):
    '''
    Creates folium world map with wind turbine locations presented as markers. 
    Markers provides average wind speed and city name for each location

    Input:
    Dataframe with city_name, wind speed and coordinate information
     
    Output:
    Display map
    Returns city_name to use as a filter
    '''
    map_st = folium.Map(location=[20,20], zoom_start=2, tiles="CartoDB positron")

    icon_url = "WData/wind-mill.png"

    #Adding city markers
    for index, row in df.iterrows():
        lat = row["lat"]
        lng = row["lng"]
        city_name = row["city_ascii"]
        icon1 = folium.CustomIcon(icon_url, icon_size=(20,20))
        folium.Marker([lat, lng], tooltip=city_name, icon=icon1).add_to(map_st)
    
    st_map = st_folium(map_st, width = 700, height = 500)

    #Assign city name from map
    if st_map["last_object_clicked_tooltip"]:
        city_name = st_map["last_object_clicked_tooltip"]
    else:
        city_name = ""
    
    return city_name

def wind_metrics(df, city_name):
    '''
    Creates wind parameters in metric format

    Input:
    Wind Dataframe & city_name
     
    Output:

    Display metrics, return avg_wind, weibull_a, weibull_k
    '''
    df = df.set_index("city_ascii")
    avg_wind = df.loc[city_name, "avg_wind"]
    weibull_a = df.loc[city_name, "weibull_a"]
    weibull_k = df.loc[city_name, "weibull_k"]

    st.metric("Average Wind Speed", "{:.2f} m/s".format(avg_wind))
    st.metric("Weibull Scale Parameter", "{:.2f} m/s".format(weibull_a))
    st.metric("Weibull Shape Parameter", "{:.2f}".format(weibull_k))

    return avg_wind, weibull_a, weibull_k

def annual_wind_dist_arr(weibull_a, weibull_k):
    '''
    Creates Annual Wind distribution Array
    
    Input:
    Weibull Scale Int Parameter
    Weibull Shape Int Parameter
    
    Output:
    Plots histogram plot of wind distribution
    Returns Annual Wind Distribution Array
    '''
    w_dist_arr = weibull_min.rvs(weibull_k, scale=weibull_a, size=8760, random_state=42)
    fig,ax = plt.subplots()
    sns.histplot(x=w_dist_arr, color = "#0068C9", ax=ax).set(title='Annual Wind Speed Distributions',
                                             xlabel= 'Wind Speed m/s', ylabel='Hours')
    st.pyplot(fig)
    return w_dist_arr

def reg_model_builder(arr_w, arr_p):
    """
    Builds power curve regression model.

    Input:
    power curve speed(list), 
    power curve power output(list)
        
    Output: 
    DecisionTreeRegressionModel for PowerCurve
    poly_transformer
    wind speed array
    power output array
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
    """
    Calculates Annual energy production.
    Input:
    Regresion model 
    Poly transformer
    Reference annual wind distribution with hourly spans(list) 

    Output: 
    Annual Energy production with %95 production availability rate(int)
    """
    X_pred = poly.fit_transform(np.array(ref_w).reshape(-1,1))
    preds= reg_model.predict(X_pred)
    AEP = np.sum(preds)*0.95

    return int(AEP/1000)

def AEP_CF_main(pw_curve, wind_arr,wt_data):
    """
    Calculates Annual energy productions, capacity factors and create city wind turbine data frame.
    Input:
    Power Curve Dictionary
    Wind Distribution Array
    Wind Turbine Dataframe

    Output: 
    City Wind Turbine Dataframe
    """
    AEP_arr = []
    for wind, power in zip(pw_curve["wind_speed"], pw_curve["power"]):
        reg_model, poly, X, preds = reg_model_builder(wind, power)
        AEP = AEP_calc(reg_model, poly, wind_arr)
        AEP_arr.append(AEP)

    city_wt_df = wt_data[["manufacturer", "turbine_type", "power_cat", "nominal_power"]]
    city_wt_df["AEP"] = AEP_arr
    city_wt_df["CF"] = city_wt_df["AEP"] * 1000 / 8760 / city_wt_df["nominal_power"]
    city_wt_df.columns = ["Manufacturer", "Turbine Name", "Power Category", "Nominal Power", "Annual Energy Production (MWh)", "Capacity Factor"]

    return city_wt_df

def display_wt_parameters(city_wt_data,city_name,w_dist_arr, power_curve_dict, pwr_cat_filter, man_filter, top_filter):
    '''
    Displays Best Wind Turbine Table and Power Curve Plot
    
    Input:
    City Wind Turbine Dataframe
    City Name
    Wind Distribution Array
    Power Curve Dictionary
    Power Category Filter (str)
    Manufacturer Filter (list)
    Top N filter (int)
    
    Output:
    None
    '''
    # Apply Filters
    if pwr_cat_filter != "All Categories":
        city_wt_data_f = city_wt_data[city_wt_data["Power Category"] == pwr_cat_filter]
    else:
        city_wt_data_f = city_wt_data
    
    if " All Companies" not in man_filter:
        city_wt_data_f = city_wt_data_f[city_wt_data_f["Manufacturer"].isin(man_filter)]

    #Sort Values and store indexes to use on power curve plots   
    city_wt_data_f = city_wt_data_f.sort_values(by="Annual Energy Production (MWh)", ascending=False)       
    wt_indexes = city_wt_data_f.head(top_filter).index

    #Modify dataframe to show as table
    city_wt_data_f.reset_index(inplace=True,drop=True)
    city_wt_data_f.index += 1
    st.subheader("Top {1} Turbines with Highest Energy Yield in {0}".format(city_name, top_filter), divider="blue")
    st.table(city_wt_data_f.head(top_filter))

    #Plot Power Curves with Wind distribution  
    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    sns.histplot(x=w_dist_arr, color = "#0068C9",fill=False, ax=ax).set(
                                           title='Top {0} Wind Turbine Power Curves'.format(top_filter),
                                           xlabel= 'Wind Speed m/s', ylabel='Hours')
        
    for index, i in enumerate(wt_indexes):
        lab = "{0} {1}".format(city_wt_data_f.loc[index+1, "Manufacturer"], city_wt_data_f.loc[index+1, "Turbine Name"])
        sns.lineplot(x=power_curve_dict["wind_speed"][i], y= power_curve_dict["power"][i], ax=ax1, label = lab)
        
    sns.set_palette("pastel")
    ax1.set(ylabel= "Power Output (kW)")
    ax1.legend()
    st.pyplot(fig)

def display_city_filter(world_wind,city_name):
    '''
    Creates city selection slider on sidebar
    
    Input:
    City Dataframe
    City_name
    
    Output:
    City_name
    '''
    city_list = [""] + list(world_wind["city_ascii"].unique())
    city_list.sort()
    city_index = city_list.index(city_name)

    return st.sidebar.selectbox("City", city_list, city_index)

def display_power_cat_filter(wt_data, power_cat = "All Categories"):
    '''
    Creates power category selector on sidebar
    
    Input:
    Wind Turbine Dataframe
    
    Output:
    Power Category
    '''
    cat_list = ["All Categories"] + list(wt_data["power_cat"].unique())
    cat_list.sort()
    cat_index = cat_list.index(power_cat)

    return st.sidebar.radio("Power Category", cat_list, cat_index)

def display_manufacturer_filter(wt_data):
    '''
    Creates company selector on sidebar
    
    Input:
    Wind Turbine Dataframe
    
    Output:
    Manufacturers
    '''
    man_list = [" All Companies"] + list(wt_data["manufacturer"].unique())
    man_list.sort()

    return st.sidebar.multiselect("Companies", man_list, " All Companies")

def display_top_filter():
    '''
    Creates number of turbine selector on sidebar
    
    Input:
    None
    
    Output:
    Number
    '''
    return st.sidebar.number_input("Top", 1, 10, 3)

#MAIN FUNCTION
def main():
    st.set_page_config(page_title, page_icon=':cyclone:')
    st.title(head)
    st.header(main_title)
    st.markdown(author)
    st.markdown(intro)

    #####Load Data#####

    #DataFrames
    wt_database = pd.read_csv("WData/Wind_Turbine_Database.csv")  
    world_wind = pd.read_csv("WData/world_wind.csv")  
    #Dictionaries
    with open('WData/power_curve.pkl', 'rb') as file:
        power_curve_dict = pickle.load(file)

    st.subheader("Wind Parameters", divider= "blue")
    #####Display Map#####
    st.write("**:orange[Please select a location on the map or use the selection panel on the left side]**")
    city_name = display_map(world_wind)

    #####Display Filters#####
    city_name = display_city_filter(world_wind, city_name)
    pwr_cat_filter = display_power_cat_filter(wt_database)
    man_filter = display_manufacturer_filter(wt_database)
    top_filter = display_top_filter()

    col1, col2 = st.columns([0.66,0.33])
    #####Display Wind Metrics#####
    with col2:
        if city_name:
            st.write("**{} Wind Parameters**".format(city_name))
            avg_wind, weibull_a, weibull_k = wind_metrics(world_wind, city_name)
    
    if city_name:
        st.markdown("Note: Wind parameters are derived from the area where the wind speed is between %60 - %80 percentile of the zone at 100 M elevation.")

    #####Display Annual Wind Distributions#####
    with col1:
        if city_name:
            w_dist_arr = annual_wind_dist_arr(weibull_a, weibull_k)

    #Calculate AEP for the turbines
    if city_name:
        city_wt_data = AEP_CF_main(power_curve_dict, w_dist_arr, wt_database)

    #####Display Top Turbines#####
    if city_name:
        display_wt_parameters(city_wt_data,city_name,w_dist_arr, power_curve_dict, pwr_cat_filter, man_filter, top_filter)

    #####Limitations#####
    st.subheader("Limitations")
    st.caption("**1.** The dataset contains the data from 66 wind turbine configurations by different commercial manufacturers until 2019.")
    st.caption("**2.** There is no guarantee that the methods used to collect technical specifications are unified across all manufacturers.")
    st.caption("**3.** There are parameters such as air density, wake effects, wind direction etc. that are not took into account. Annual energy production and capacity factors evaluated can be considered optimal values with generated wind distributions at 100 meter elevation.")

    #####References#####
    st.subheader("References")
    st.caption("**1.** Global Wind Atlas, https://globalwindatlas.info")
    st.caption("**2.** Open Energy Platform Wind Turbine Library, https://openenergy-platform.org/dataedit/view/supply/wind_turbine_library")
    st.caption("**3.** 'Wind Turbine Benchmarking by using Weibull Wind Distributions and Power Curve Regression Models' , Cihan Erturk, 03.01.2023, https://medium.com/@ghanertrk/wind-turbine-benchmark-analysis-with-annual-energy-productions-and-capacity-factors-a5d3e5eb2ed7")

    #####Disclaimer#####
    st.caption("*DISCLAIMER: The information in this application is intended solely for the personal non-commercial use of the user who accepts full responsibility for its use. The information contained in this application is provided on an 'as is' basis with no guarantees of completeness, accuracy, usefulness or timeliness.*")
    st.caption("*DISCLAIMER 2: This application is accomplished by Cihan Erturk in his personal capacity. The analysis shared in this application is the author's own and does not reflect the view of any corporate entity.*")

if __name__ == "__main__":
    main()
