# main_segmentation.py


import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

#Loading and cleaning dataset

def load_and_clean_data(filepath):
    """Load and clean the data."""
    print("Step 1 1: Loading and cleaning...")
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print("Cleaning finished.")
    return df

#Calculate RFM

def calculate_rfm(df):
    """Calculate RFM score for each customer."""
    print("Step 2: RFM metrics...")
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)
    print("RFM calculation finished.")
    return rfm_df

#Perform Clustering

def perform_clustering(rfm_df, n_clusters=4):
    """Process data and perform K-Means clustering."""
    print(f"Step 3: Clustering in {n_clusters} segments...")
    rfm_log = np.log1p(rfm_df)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(rfm_scaled)
    
    rfm_df['Cluster'] = kmeans.labels_
    print("Clustering finished.")
    return rfm_df

#Personas classification

def assign_personas(rfm_df):
    """Assign persona to clusters."""
    print("Step 4: Assign personas...")
    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
    
    # Cluster identification by their characteristics
    champions_cluster = cluster_summary['MonetaryValue'].idxmax()
    at_risk_cluster = cluster_summary['Recency'].idxmax()
    
    # Mapping the clusters
    persona_map = {}
    for cluster_id in cluster_summary.index:
        if cluster_id == champions_cluster:
            persona_map[cluster_id] = 'Champions'
        elif cluster_id == at_risk_cluster:
            persona_map[cluster_id] = 'Client at Risk'
        elif cluster_summary.loc[cluster_id, 'Frequency'] > 4: 
            persona_map[cluster_id] = 'loyal clients'
        else:
            persona_map[cluster_id] = 'Occasionnal client'

    rfm_df['Persona'] = rfm_df['Cluster'].map(persona_map)
    print("Personas assigned.")
    return rfm_df

#Integration

def prepare_for_integration(df, output_filename='export_for_integration.json'):
    """
  This function selects essential columns (customer ID and segment)and exports them in a standard format (JSON) that is easy to use by an API or other service.
    """
    print("\nfinal step: Preparing data for integration...")
    
    export_df = df[['Persona']].reset_index() 
    
    # Export in JSON. 
    export_df.to_json(output_filename, orient='records', lines=True)
    
    print(f"-> File '{output_filename}' success.")
    print("   This file contains data ready to be sent to a CRM or marketing platform.")
    
    # To illustrate, we can display a preview of what is generated.
    print("\n3 first rows :")
    print(export_df.head(3).to_json(orient='records', lines=True, indent=4))

# Pipeline execution

def main():
    """Main function for executing the segmentation pipeline."""
    RAW_DATA_PATH = 'OnlineRetail.csv'
    OUTPUT_PATH = 'customer_segments.csv'
    
    df = load_and_clean_data(RAW_DATA_PATH)
    rfm_df = calculate_rfm(df)
    rfm_with_clusters = perform_clustering(rfm_df)
    final_segments = assign_personas(rfm_with_clusters)
    
    # Save the final result
    final_segments.to_csv(OUTPUT_PATH)
    print(f"\nProcess finished '{OUTPUT_PATH}'.")
    print(final_segments.head())

if __name__ == '__main__':
    main() 