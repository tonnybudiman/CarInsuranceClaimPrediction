import streamlit as st
import streamlit.components.v1 as html
from streamlit_option_menu import option_menu
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
# from st_aggrid import AgGrid
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

with st.sidebar:
    choose = option_menu("DS Application", ["About", "Preparing", "EDA", "Modeling & Evaluation"],
                         icons=['house', 'kanban', 'camera fill', 'book'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

st.markdown(""" <style> .font {font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
logo = Image.open(r'C:\Users\tonny\StreamLit\Tugas\images\claim.jpeg')

# Buat fungsi untuk split dan replace data
def split_replace_to_float(value, character_value, replace_value):
  parts = value.split(character_value)
  return float(parts[0].replace(replace_value,''))

# Buat fungsi konversi to bit number
def convert_to_int(value):
  if value == "False":
    return 0
  return 1

def run_process(df):
    st.write('**The data has {} rows and {} columns**'.format(df.shape[0], df.shape[1]))
    st.dataframe(df.head())

    st.write('---')

    st.markdown(":white_check_mark: Check Null")
    st.write(df.isnull().sum())

    st.write('---')

    st.markdown(":white_check_mark: Apply Function")

    max_torque = df["max_torque"].astype(str).apply(lambda x:split_replace_to_float(x,"@","Nm"))
    df["max_torque"] = max_torque

    max_power = df["max_power"].astype(str).apply(lambda x:split_replace_to_float(x,"@","bhp"))
    df["max_power"] = max_power

    is_esc = df["is_esc"].apply(lambda x:convert_to_int(x))
    df["is_esc"] = is_esc

    is_adjustable_steering = df["is_adjustable_steering"].apply(lambda x:convert_to_int(x))
    df["is_adjustable_steering"] = is_adjustable_steering

    is_tpms = df["is_tpms"].apply(lambda x:convert_to_int(x))
    df["is_tpms"] = is_tpms

    is_parking_sensors = df["is_parking_sensors"].apply(lambda x:convert_to_int(x))
    df["is_parking_sensors"] = is_parking_sensors

    is_parking_camera = df["is_parking_camera"].apply(lambda x:convert_to_int(x))
    df["is_parking_camera"] = is_parking_camera

    is_front_fog_lights = df["is_front_fog_lights"].apply(lambda x:convert_to_int(x))
    df["is_front_fog_lights"] = is_front_fog_lights

    is_rear_window_wiper = df["is_rear_window_wiper"].apply(lambda x:convert_to_int(x))
    df["is_rear_window_wiper"] = is_rear_window_wiper

    is_rear_window_washer = df["is_rear_window_washer"].apply(lambda x:convert_to_int(x))
    df["is_rear_window_washer"] = is_rear_window_washer

    is_rear_window_defogger = df["is_rear_window_defogger"].apply(lambda x:convert_to_int(x))
    df["is_rear_window_defogger"] = is_rear_window_defogger

    is_brake_assist = df["is_brake_assist"].apply(lambda x:convert_to_int(x))
    df["is_brake_assist"] = is_brake_assist

    is_power_door_locks = df["is_power_door_locks"].apply(lambda x:convert_to_int(x))
    df["is_power_door_locks"] = is_power_door_locks

    is_central_locking = df["is_central_locking"].apply(lambda x:convert_to_int(x))
    df["is_central_locking"] = is_central_locking

    is_power_steering = df["is_power_steering"].apply(lambda x:convert_to_int(x))
    df["is_power_steering"] = is_power_steering

    is_driver_seat_height_adjustable = df["is_driver_seat_height_adjustable"].apply(lambda x:convert_to_int(x))
    df["is_driver_seat_height_adjustable"] = is_driver_seat_height_adjustable

    is_day_night_rear_view_mirror = df["is_day_night_rear_view_mirror"].apply(lambda x:convert_to_int(x))
    df["is_day_night_rear_view_mirror"] = is_day_night_rear_view_mirror

    is_ecw = df["is_ecw"].apply(lambda x:convert_to_int(x))
    df["is_ecw"] = is_ecw

    is_speed_alert = df["is_speed_alert"].apply(lambda x:convert_to_int(x))
    df["is_speed_alert"] = is_speed_alert

    st.dataframe(df.head())
    st.session_state.my_df = df

if choose == "About":    
    st.markdown('<p class="font">Car Insurance Claim Prediction</p>', unsafe_allow_html=True)
    st.image(logo, width=550)
elif choose == "Preparing":
    st.markdown('<p class="font">Preparing</p>', unsafe_allow_html=True)

    if "my_df" in st.session_state:
        df = st.session_state.my_df
        run_process(df)
    else:
        uploaded_file = st.file_uploader(":cloud: Upload Data")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            run_process(df)

elif choose == "EDA":

    st.markdown('<p class="font">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if "my_df" in st.session_state:
        st.markdown(":white_check_mark: Find Correlation")
        df = st.session_state.my_df
        st.write(df["is_claim"].value_counts())
        st.dataframe(df.corr())
        
        st.write('---')

        st.markdown(":white_check_mark: Describe")
        st.dataframe(df.describe())

        st.write('---')

        st.markdown(":white_check_mark: Detecting Outliers")

        selected_column = ["policy_tenure", "age_of_car", "age_of_policyholder","population_density","max_torque","max_power","turning_radius","length","width","height","is_claim"]
        df_boxplot = df[selected_column]
        df_boxplot.head(3)

        st.markdown("1. Is Claim VS Policy Tenure")
        fig = px.box(df_boxplot, x="is_claim", y="policy_tenure", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("2. Is Claim VS Age of Car")
        fig = px.box(df_boxplot, x="is_claim", y="age_of_car", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        st.markdown("3. Is Claim VS Policy Holder")
        fig = px.box(df_boxplot, x="is_claim", y="age_of_policyholder", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        st.markdown("4. Is Claim VS Population Density")
        fig = px.box(df_boxplot, x="is_claim", y="population_density", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        st.markdown("5. Is Claim VS Max Torque")
        fig = px.box(df_boxplot, x="is_claim", y="max_torque", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("6. Is Claim VS Max Power")
        fig = px.box(df_boxplot, x="is_claim", y="max_power", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("7. Is Claim VS Turning Radius")
        fig = px.box(df_boxplot, x="is_claim", y="turning_radius", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("8. Is Claim VS Lenght")
        fig = px.box(df_boxplot, x="is_claim", y="length", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("9. Is Claim VS Width")
        fig = px.box(df_boxplot, x="is_claim", y="width", color="is_claim")
        st.plotly_chart(fig)

        st.markdown("10. Is Claim VS Height")
        fig = px.box(df_boxplot, x="is_claim", y="height", color="is_claim")
        st.plotly_chart(fig)

        st.write('---')

        st.markdown(":white_check_mark: Exclude Outliers With Quantile 94%")
        quantile_ageofcar = df["age_of_car"].quantile(0.94)
        quantile_ageofpolicyholder = df["age_of_policyholder"].quantile(0.94)
        quantile_populationdensity = df["population_density"].quantile(0.94)
        st.write("Quantile Age of Car = {}".format(quantile_ageofcar))
        st.write("Quantile Age of Policy Holder = {}".format(quantile_ageofpolicyholder))
        st.write("Quantile Population Density = {}".format(quantile_populationdensity))
        df_after_quantile = df[(df["age_of_car"] < quantile_ageofcar) & (df["age_of_policyholder"] < quantile_ageofpolicyholder) & (df["population_density"] < quantile_populationdensity)]

        st.write(' ')
        st.write('- **Data before quantile has {} rows and {} columns**'.format(df.shape[0], df.shape[1]))
        st.write('- **Data after quantile has {} rows and {} columns**'.format(df_after_quantile.shape[0], df_after_quantile.shape[1]))
        st.write(' ')

        st.markdown("2. Is Claim VS Age of Car")
        fig = px.box(df_after_quantile, x="is_claim", y="age_of_car", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        st.markdown("3. Is Claim VS Policy Holder")
        fig = px.box(df_after_quantile, x="is_claim", y="age_of_policyholder", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        st.markdown("4. Is Claim VS Population Density")
        fig = px.box(df_after_quantile, x="is_claim", y="population_density", color="is_claim")
        st.plotly_chart(fig) #terdapat outlier

        df = df_after_quantile
        # df = df.drop(['policy_id','area_cluster','make','segment','model','fuel_type','engine_type','airbags','rear_brakes_type','transmission_type','steering_type','ncap_rating'], axis=1)

elif choose == "Modeling & Evaluation":

    st.markdown('<p class="font">Modeling & Evaluation</p>', unsafe_allow_html=True)

    if "my_df" in st.session_state:
        df = st.session_state.my_df
        # Hapus kolom yang tidak terkorelasi
        df = df.drop(['policy_id','area_cluster','make','segment','model','fuel_type','engine_type','airbags','rear_brakes_type','transmission_type','steering_type','ncap_rating'], axis=1)

        # Data Target
        y = df["is_claim"]

        # Data Features
        X = df.iloc[:, :-1]

        # Split into training and Testing
        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test = tts(X,y,test_size=0.20,random_state=2023)

        # Scaling
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        from sklearn.metrics import classification_report
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_percentage_error

        def logistic_regression():
            # Logistik Regression
            model_lg = LogisticRegression()
            model_lg.fit(X_train, y_train)
            # prediksi dari model Logistic Regression
            y_pred_lr = model_lg.predict(X_test)

            # st.write(model_lg.score(y_test, y_pred_lr))
            st.write(classification_report(y_test, y_pred_lr))
            st.write(evaluasi(y_test, y_pred_lr))
        
        def decission_tree():
            # Decision Tree
            model_dt = DecisionTreeClassifier()
            model_dt.fit(X_train, y_train)

            # prediksi dari model Decision Tree
            y_pred_dt = model_dt.predict(X_test)

            # st.write(model_dt.score(y_test, y_pred_dt))
            st.write(classification_report(y_test, y_pred_dt))
            st.write(evaluasi(y_test, y_pred_dt))

        def svm():
            # SVM
            model_svm = SVC()
            model_svm.fit(X_train, y_train)

            # prediksi dari model SVM
            y_pred_svm = model_svm.predict(X_test)

            # st.write(model_svm.score(y_test, y_pred_svm))
            st.write(classification_report(y_test, y_pred_svm))
            st.write(evaluasi(y_test, y_pred_svm))

        def random_forest():
            # Random Forest
            model_rf = RandomForestClassifier()
            model_rf.fit(X_train, y_train)

            # prediksi dari model Random Forest
            y_pred_rf = model_rf.predict(X_test)

            # st.write(model_rf.score(y_test, y_pred_rf))
            st.write(classification_report(y_test, y_pred_rf))
            st.write(evaluasi(y_test, y_pred_rf))

        def xgboost():
            # XGBOOST
            xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=3)
            xgb_classifier.fit(X_train, y_train)

            # prediksi dari model XGB
            y_pred_xgboost = xgb_classifier.predict(X_test)

            # st.write(xgb_classifier.score(y_test, y_pred_xgboost))
            st.write(classification_report(y_test, y_pred_xgboost))
            st.write(evaluasi(y_test, y_pred_xgboost))
        
        def evaluasi(asli, prediksi):
            mse = mean_squared_error(asli, prediksi)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(asli, prediksi)
            return ["MSE : " + str(mse), "RMSE : " + str(rmse), "MAPE : " + str(mape)]
        
        st.button("Logistik Regression Model", on_click=logistic_regression)
        st.button("Decission Tree", on_click=decission_tree)
        st.button("SVM", on_click=svm)
        st.button("Random Forest", on_click=random_forest)
        st.button("XG Boost", on_click=xgboost)