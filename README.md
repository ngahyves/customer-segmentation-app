# Customer segment analysis (RFM) usink K-Means
This project presents the development of an end-to-end machine learning pipeline to segment a customer database based on their purchasing behaviour. Using more than 400,000 transactions, we transform raw data into actionable customer segments, enabling targeted and personalised marketing campaigns.
The data set is coming for kaggle: https://www.kaggle.com/datasets/vijayuv/onlineretail
# Dashbord overview
![image](https://github.com/user-attachments/assets/2a1aa795-d589-42c9-bc35-97016f92a5ce)

# Key Features
End-to-End Pipeline: A fully automatable script (main_segmentation.py) that handles data cleaning, feature engineering (RFM), and clustering.
Unsupervised Learning: Application of the K-Means algorithm to identify distinct customer groups.
Business-Oriented Personas: Segments are translated into understandable personas (e.g., "Champions", "At-Risk") with actionable marketing recommendations.
Interactive Dashboard: A user-friendly web application (dashboard.py) for visualizing segment characteristics and exploring the data.
# 1-Project Objective
The main objective is to design an automatable system capable of:
Cleaning and preparing transactional data.
Calculate relevant behavioural metrics (RFM).
Apply an unsupervised clustering algorithm (K-Means) to identify homogeneous customer groups.
Provide strategic recommendations for each identified segment.

# 2-Methodology
## a-Data Preparation & Feature Engineering
Cleaning: Handling of missing values, removal of duplicates, and filtering of irrelevant entries (e.g., returns).
Feature Creation: Calculation of RFM (Recency, Frequency, Monetary Value) metrics for each customer, which form the basis of our behavioural analysis.
## b-Unsupervised Modelling (Clustering)
Data Transformation: Application of a logarithmic transformation (log1p) to reduce the skewness of RFM distributions.
Standardization: Use of StandardScaler to ensure that each feature has an equivalent weight in the model.
Clustering: Application of the K-Means algorithm to group customers. The optimal number of clusters (K=4) was determined using the Elbow Method.
## c-Analysis & Interpretation of Segments
Characterization: Calculation of the average RFM profile for each identified cluster.
Persona Assignment: Creation of meaningful personas for each cluster (e.g., '🏆 Champions', '👻 At-Risk Customers') to make the results understandable for business stakeholders.
Recommendations: Formulation of specific marketing strategies to effectively engage each segment.
## d-Operationalization & Visualization
Automation: Development of a fully automatable Python script (main_segmentation.py) that executes the entire pipeline.
Integration: Generation of a JSON export file ready for integration with external tools like CRMs or marketing platforms.
Visualization: Creation of an interactive dashboard with Streamlit to enable visual exploration of segments and support data-driven decisions.
The project pipeline is structured in four main stages:

# 3-Technologies and Skills
Data Analysis & Manipulation: Python, Pandas, NumPy
Machine Learning: Scikit-learn (KMeans, StandardScaler)
Data Visualisation: Matplotlib, Seaborn
Dashboarding & Web Application: Streamlit
Cross-functional Skills: Feature Engineering, Exploratory Data Analysis (EDA), Clustering, 

# 4-How to Run the Project Locally
Prerequisites
Python 3.8 or higher
Git
## 1. Clone the Repository
Open your terminal and clone this repository:
git clone https://github.com/ngahyves/Machine_learning_projects.git
cd Machine_learning_projects
## 2. Create and Activate a Virtual Environment (Recommended)
It's a best practice to create a virtual environment to manage project-specific dependencies.
On Windows:
Generated bash
python -m venv venv
.\venv\Scripts\activate
## 3-Run the Data Pipeline
Execute the main script to process the data and generate the segmentation files. This will create customer_segments.csv and export_for_integration.json.
Generated bash
python Customer_Segmentation.py
## 4-Launch the Interactive Dashboard
Now, run the Streamlit app to visualize the results:
Generated bash
streamlit run Dashboard_Segmentation.py
