#Dashbord segmentation.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Segmentation Client",
    page_icon="📊",
    layout="wide"
)

##Load the data
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, index_col='CustomerID')
        return df
    except FileNotFoundError:
        st.error(f"File not find : {filepath}. Firstly execute the script 'main_segmentation.py'")
        return None

df = load_data('customer_segments.csv')

#If loading failed, stop the script execution
if df is None:
    st.stop()


# --- Dashboard title  ---
st.title("Dashbord of client's segmentation")
st.markdown("This interactive dashboard allows you to explore customer segments identified by RFM analysis and K-Means clustering.")


# --- Segment Overview (KPIs and Charts)---
st.header("Segment Overview")

# Define a colour palette for visual consistency
color_map = {
    'Champions': 'gold',
    'Clients Fidèles': 'royalblue',
    'Clients Occasionnels': 'lightgray',
    'Clients à Risque': 'darkred'
}

# Organise the overview
col1, col2 = st.columns(2)

with col1:
    # Graph: Breakdown of customers by segment
    st.subheader("customers by segment")
    fig_pie = px.pie(
        df, 
        names='Persona', 
        title='Percentage of customers by segment',
        color='Persona',
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Graph: Revenue by segment
    st.subheader("Contribution to turnover")
    revenue_by_segment = df.groupby('Persona')['MonetaryValue'].sum().reset_index()
    fig_bar = px.bar(
        revenue_by_segment, 
        x='Persona', 
        y='MonetaryValue', 
        title='Total Revenue by Segmentt',
        color='Persona',
        color_discrete_map=color_map,
        labels={'MonetaryValue': 'Chiffre d\'Affaires Total (€)', 'Persona': 'Segment'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Detailed Segment Analysis ---
st.header("Detailed Segment Analysis")

# Display descriptive statistics for each segment
st.subheader("Average Characteristics of Segments")
st.dataframe(df.groupby('Persona')[['Recency', 'Frequency', 'MonetaryValue']].agg(['mean', 'median']).round(1))


# --- Customer Data Exploration ---
st.header("Customer Data Exploration")

# Create a selector so that the user can choose a segment
all_personas = ['All'] + df['Persona'].unique().tolist()
selected_persona = st.selectbox("Select a segment to display :", all_personas)

# Filter the DataFrame based on the selection
if selected_persona == 'All':
    filtered_df = df
else:
    filtered_df = df[df['Persona'] == selected_persona]

# Display the number of customers in the selection and data
st.write(f"Display **{len(filtered_df)} total** on **{len(df)}** clients.")
st.dataframe(filtered_df.sort_values(by='MonetaryValue', ascending=False).head(20))


# --- Footer ---
st.markdown("---")
st.write("Customer Segmentation Project carried out by [NGAH Yves-Bernard-Simplice]")