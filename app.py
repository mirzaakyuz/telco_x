import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

img = Image.open("exit_unsplash_ws.jpg")
caption = """
Photo by [Possessed Photography] 
(https://unsplash.com/@possessedphotography) 
on [Unsplash] 
(https://unsplash.com/)"""
st.image(img, width=600)
st.write(caption)

st.header('TELCO Customer Churn Prediction App')

st.write("""
            This app predicts whether the customer has a probability of churn or not according to the given features.
            Data obtained from [Kaggle] (https://www.kaggle.com/blastchar/telco-customer-churn) 
            """)
st.sidebar.header("Please input features of customer:")


def user_input_fetures():
    contract = st.sidebar.selectbox('Which type of contract has the customer?',
                                    ('Month-to-month', 'One year', 'Two year'))
    payment = st.sidebar.selectbox('Which payment method prefer the customer?',
                                   ('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                    'Credit card (automatic)'))
    gender = st.sidebar.selectbox('What is the gender of customer ?', ('Female', 'Male'))
    tenure = st.sidebar.slider('How long the customer takes service?', 0, 72, 32)
    month_charge = st.sidebar.slider("How much charged to the customer monthly?", 18, 119, 64)
    total_charge = st.sidebar.slider("How much charged to the customer total?", 18, 8685, 2200)
    internet_service = st.sidebar.selectbox('Do the customer get Internet Service?',
                                            ('DSL', 'Fiber optic', 'No'))
    if internet_service != 'No':
        services = st.sidebar.multiselect("Which services do the customer have?",
                                          ('OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                           'TechSupport', 'StreamingTV', 'StreamingMovies'))
    else:
        services = []
    phone_service = st.sidebar.selectbox('Do the customer get Phone Service?',
                                         ('Yes', 'No'))
    if phone_service != 'No':
        multiple_lines = st.sidebar.selectbox('Is Multiple lines is inclusive in Phone Service?', ('Yes', 'No'))
    else:
        multiple_lines = 'No'
    senior_citizen = st.sidebar.selectbox('Is the customer senior?', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Has the customer a partner?', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Has the customer dependents?', ('Yes', 'No'))
    paperless_bill = st.sidebar.selectbox('Get the customer paperless bill?', ('Yes', 'No'))

    data = {'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'PaperlessBilling': paperless_bill,
            'gender': gender,
            'InternetService': internet_service,
            'Contract': contract,
            'PaymentMethod': payment,
            'TotalCharges': total_charge,
            'MonthlyCharges': month_charge
            }

    features = pd.DataFrame(data, index=[0])
    # st.write(features)
    service_list = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']

    dum_dict = {}
    for i in service_list:
        if i in services:
            dum_dict[i] = [1]
        else:
            dum_dict[i] = [0]

    # st.write(type(data["InternetServices"]))
    service_df = pd.DataFrame.from_dict(dum_dict)
    input_df = pd.concat([features, service_df], axis=1)
    # st.write(list(input_df.columns))

    input_df = input_df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                         "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                         "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                         "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]]
    st.write("Below is your selection:")
    st.table(input_df.T)

    map_dict = {'No': 0, 'Yes': 1}

    input_df["SeniorCitizen"] = input_df["SeniorCitizen"].map(map_dict)
    input_df["Partner"] = input_df["Partner"].map(map_dict)
    input_df["Dependents"] = input_df["Dependents"].map(map_dict)
    input_df["PhoneService"] = input_df["PhoneService"].map(map_dict)
    input_df["MultipleLines"] = input_df["MultipleLines"].map(map_dict)
    input_df["PaperlessBilling"] = input_df["PaperlessBilling"].map(map_dict)

    # st.write(input_df)

    return input_df


df = user_input_fetures()


@st.cache
def feature_load():
    with open('to_streamlit.pkl', 'rb') as m:
        f = pickle.load(m)
        return f


@st.cache(allow_output_mutation=True)
def model_load():
    with open('model_ada_load.pkl', 'rb') as m:
        f = pickle.load(m)
        return f


unpickled_df = feature_load()

df_total = pd.concat([unpickled_df, df])

df_dum = pd.get_dummies(data=df_total, drop_first=True)
df_dum['#_of_Services'] = df_dum[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                  'TechSupport', 'StreamingTV', 'StreamingMovies']].sum(axis=1)

df_dum['TotalCharges_logged'] = np.log(df_dum['TotalCharges'])
df_dum['MonthlyCharges_logged'] = np.log(df_dum['MonthlyCharges'])
df_dum = df_dum.drop(['TotalCharges', 'MonthlyCharges'], axis=1)

df_for_predict = df_dum.iloc[-1].values.reshape(1, 24)

st.write("If your selection is done, please Click on Predict button.")

load_clf = model_load()

if st.button("Predict"):
    prediction = load_clf.predict(df_for_predict)
    prediction_proba = load_clf.predict_proba(df_for_predict)

    if prediction == 0:
        st.success(f"The customer with a probability of % {prediction_proba.flat[0]:.3f} continue to use our service.")
    else:
        st.error(f"The customer with a probability of % {prediction_proba.flat[1]:.3f} may churn.")

if st.checkbox("Top customers can churn"):

    num_of_cust_churn = st.number_input("Please type a number", 1, 7043, 5, key='1')

    if st.button("Show", key='0'):
        prediction = load_clf.predict(df_dum[:-1].values)
        prediction_proba = load_clf.predict_proba(df_dum[:-1].values)

        pred_dict = {'Churn': list(prediction), 'Probability': list(prediction_proba[:, 1])}

        pred_df = pd.DataFrame.from_dict(pred_dict)

        to_show = pd.concat([df_total[:-1], pred_df], axis=1)

        to_show = to_show[to_show['Churn'] == 1].sort_values("Probability", ascending=False).head(num_of_cust_churn)

        st.write(to_show)

if st.checkbox("Top loyal customers"):

    num_of_cust_stay = st.number_input("Please type a number", 1, 7043, 5, key='2')

    if st.button("Show", key='1'):
        prediction = load_clf.predict(df_dum[:-1].values)
        prediction_proba = load_clf.predict_proba(df_dum[:-1].values)

        pred_dict = {'Churn': list(prediction), 'Probability': list(prediction_proba[:, 0])}

        pred_df = pd.DataFrame.from_dict(pred_dict)

        to_show = pd.concat([df_total[:-1], pred_df], axis=1)

        to_show = to_show[to_show['Churn'] == 0].sort_values("Probability", ascending=False).head(num_of_cust_stay)

        st.write(to_show)
