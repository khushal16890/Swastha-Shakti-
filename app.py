import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load(open("decision_tree_model.pkl", 'rb'))

# Set page configuration
st.set_page_config(page_title="Heating and Cooling Load Prediction", page_icon="🏠", layout='wide')

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard"])

# Define user input function
def user_input():
    with st.form("input_form"):
        relative_compactness = st.slider('Relative Compactness', 0.62, 0.98, 0.75, 1.0, key='rc')
        wall_area = st.slider('Wall Area', 245.0, 416.5, 330.0, 1.0, key='wa')
        roof_area = st.slider('Roof Area', 110.25, 220.5, 192.5, 1.0, key='ra')
        overall_height = st.slider('Overall Height', 3.5, 7.0, 5.25, 0.25, key='oh')+
        glazing_area = st.slider('Glazing Area', 0.0, 0.4, 0.2, 0.1, key='ga')
        submitted = st.form_submit_button("Submit")
        if submitted:
            data = {
                'Relative Compactness': relative_compactness,
                'Wall Area': wall_area,
                'Roof Area': roof_area,
                'Overall Height': overall_height,
                'Glazing Area': glazing_area,
            }
            st.session_state.user_data = pd.DataFrame(data, index=[0])
        return None

if page == 'Home':
    st.title("Home")
    st.header("Input Data and Prediction")
    user_input()
    if 'user_data' in st.session_state:
        # Predictions
        heating_load = model.predict(st.session_state.user_data)
        cooling_load = model.predict(st.session_state.user_data)
        st.subheader('Predictions')
        st.write('The Predicted Heating Load is:', heating_load[0])
        st.write('The Predicted Cooling Load is:', cooling_load[0])

elif page == 'Dashboard':
    st.title("Dashboard")
    st.header("Data Visualization")
    if 'user_data' in st.session_state:
        # Histograms for each feature
        fig, ax = plt.subplots()
        sns.pairplot(st.session_state.user_data, diag_kind='hist')
        st.pyplot(fig)

        # Scatter matrix plot
        st.write('## Scatter Matrix Plot')
        fig, ax = plt.subplots(figsize=(10, 8))
        pd.plotting.scatter_matrix(st.session_state.user_data, alpha=1, figsize=(10, 10), ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("No data available. Please input data in the 'Home' sectsion.")
