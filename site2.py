from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt 
import pickle
import shap


data=pd.read_csv('X_resampled3.csv')

valid_x=pd.read_csv('valid_x2.csv')
listid=valid_x['SK_ID_CURR']
model = pickle.load(open('model.pkl','rb'))

expected_values=np.load('expected-val.npy').item()

explainer=shap.TreeExplainer(model, data, feature_names=data.columns.tolist())

def predict(id):
    x=valid_x.loc[valid_x['SK_ID_CURR'] == id]
    y=model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1]
    return y

def indix(id):
    i=valid_x.loc[valid_x['SK_ID_CURR'] == id].index.values
    return i

def shap_plot(j):
    explainerModel = shap.TreeExplainer(model)
    
    shap_values = explainerModel.shap_values(data)
    p = shap.decision_plot(explainerModel.expected_value[0],shap_values[0][j],valid_x, ignore_warnings=True)
    return(p)


explainerModel = shap.TreeExplainer(model)


#def run():

exp = shap.TreeExplainer(model).expected_value[0]

st.title("CREDIT PREDICTION")
st.header('Influence des criteres sur le choix')
st.image('shap_summary.png')
id_input = st.selectbox("Choisissez l'identifiant d'un client", listid)



if st.button('Prediction'):
    #PREDICTION
    output=predict(id_input)
    st.metric(label= 'probabilite de remboursement', value=1-output[0])

    #GRAPHE1
    st.header('Graphe de decision')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    implt=shap_plot(indix(id_input))
    st.pyplot(implt)

    st.header("Positionement du client dans la base de donees")
    enumeration=['NAME_INCOME_TYPE_Working','CODE_GENDER_M', 'NAME_FAMILY_STATUS_Married', 'REGION_RATING_CLIENT_W_CITY', 'AMT_CREDIT' ]
    fig, ax = plt.subplots()
    sns.boxplot(data=data[enumeration], ax = ax, flierprops={"marker": "x"}, color='skyblue', showcaps=True)
    plt.setp(ax.get_xticklabels(), rotation=90)
    client_data=valid_x.query(f'SK_ID_CURR == {id_input}')
  
    for k in  enumeration:
        ax.scatter(k, client_data[k].values, marker='X', s=100, color = 'black', label = 'Client selectionné')

    #GRAPHE2
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    st.pyplot(fig)
