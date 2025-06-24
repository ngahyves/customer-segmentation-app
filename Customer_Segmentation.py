#!/usr/bin/env python
# coding: utf-8

# # Customer segmentation using unsupervised machine learning

# ## I-Importing the data set and data exploration

# ### 1-Getting data

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

df=pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
df.head()


# ### 2-Data cleaning

# In[5]:


#Shape of our data set
shape=df.shape
print(f'Our data set has {shape[0]} rows and {shape[1]} columns')


# #### a-Data types

# In[16]:


# Data types
df.dtypes


# In[14]:


print(df['InvoiceNo'].unique()[:10]) 


# In[81]:


# Changing the data type of 'invoice number' and 'invoice date'

df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['InvoiceNo']=df['InvoiceNo'].astype(str)


# In[82]:


#Summary statistics
df.describe(include='all')


# In[83]:


# Checking general information about our data set

df.info()


# #### b-Missing values

# In[23]:


# Checking for missing values
missing=df.isnull().sum()
missing=missing.sort_values(ascending=False)

missing_prop=df.isnull().sum()/len(df)*100
missing_prop=missing_prop.sort_values(ascending=False)

missing_table=pd.DataFrame({
    'count':missing,
    'percentage':missing_prop
})
missing


# As we want to group the customers, we will delete the rows where we don't have custmerID and replace rows with missing values in description by the mode

# In[29]:


# Removing missing values in customerID column
df.dropna(subset=['CustomerID'], inplace=True)

#Replacing missing values by the mode in 'description column'
mode_value=df['Description'].mode()[0]
df['Description'].fillna(mode_value, inplace=True)


# #### c-Duplicates

# In[32]:


# Checking for duplicates

duplicates=df[df.duplicated()]
print(duplicates.count())


# In[36]:


# Removing duplicateed rows
df.drop_duplicates(inplace=True)


# In[39]:


print(df[df.duplicated()])
print(df.shape)

Now we do not have any duplicated rows.
# #### d-Removing inconsistencies

# When we were looking at the summary statistics, we noticed that there
# are some rows where the "Qauantity" column had negative values which is not normal.
# So we will just select rows with positive value to improve the accuracy and the consistency of our analysis

# In[44]:


df=df[df["Quantity"]>0]
df["Quantity"].min() # we just verify that we do not have missing values


# In[45]:


df.shape

Now our dataset has 392732 rows and 8 columns
# ### 3- Exploratory Data analysis

# In[48]:


# We will create a column "total price" to have the total amount of each invoice
df['total_price']=df['Quantity'] * df['UnitPrice']
df.head()


# #### a-Top 10 countries xith the most customers

# In[69]:


# Number of unique customers 
n_clients=df['CustomerID'].nunique()
print(f'Number of unique customers: {n_clients}\n')

#Number of countries where customers coming from
n_countries=df['Country'].nunique()
print(f'Number of countries: {n_countries}\n')

#Countries with the most invoices
top10_countries=df['Country'].value_counts().head(10)
print(f'Top 10 countries: {top10_countries}\n')


# In[70]:


# Top 10 countries with the most customers
top10_country = (
    df.groupby('Country')['CustomerID']
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)
top10_country=pd.DataFrame(top10_country)
top10_country


# We can see that the most representated country by customers is United kingdom

# In[75]:


top10_country.plot(kind='bar', color='teal', figsize=(10, 6))

plt.title("Top 10 countries by unique customers")
plt.xlabel("countries")
plt.ylabel("Number of customers")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# #### b-Analysis of sales during the time

# In[87]:


#Create a month column
df['month']=df['InvoiceDate'].dt.to_period('M')

# Calculate the Monthly sales
monthly_sales=df.groupby('month')['total_price'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].astype(str)


#Plot the line chart
plt.figure(figsize=(12,6))
plt.plot(monthly_sales['month'], monthly_sales['total_price'],marker='o', color='royalblue')
plt.title('Mothly sales')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
           

#There was strong growth during 2011, with a massive peak in November. This is typical of e-commerce with end-of-year shopping (Black Friday, Christmas). December 2011 appears to be incomplete in the dataset, which is important to note for our RFM analysis ‘snapshot date’.
# In[ ]:





# #### c-Top sales products

# ##### i-Top 10 best-selling products by quantity

# In[92]:


#Table of top 10 best selling products (quantity)

top_products_quantity=(df.groupby('Description')['Quantity']
                                .nunique()
                                .sort_values(ascending=False)
                                .head(10))
print(top_products_quantity)

#Bar plot of top 10 best selling products
plt.figure(figsize=(12,6))
sns.barplot(x=top_products_quantity.values, y=top_products_quantity.index)
plt.title('top 10 best selling products (Quantity)')
plt.xlabel('Total sales')
plt.ylabel('Products')
plt.show()


# ##### i-Top 10 products generating the most revenue

# In[95]:


#Table of top 10 best selling products (sales)
top_products_sales=(df.groupby('Description')['total_price']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10))
print(top_products_sales)

#Barplot of top 10 selling products
plt.figure(figsize=(12,8))
sns.barplot(x=top_products_sales.values, y=top_products_sales.index)
plt.title('top 10 best selling products')
plt.xlabel('Total sales')
plt.ylabel('Products')
plt.show()


# In[ ]:





# ## II- Feature Engineering with RFM analysis
Now we will try to calculate 3 measures
1. Recency: How many days have passed since the customer's last purchase? (A customer who purchased yesterday is more valuable than a customer who purchased a year ago).
2.    Frequency: How many times did the customer purchase in total during the period analysed? (A customer who purchases often is more committed).
3.    Monetary: How much money did the customer spend in total? (A customer who spends a lot is more valuable).
# In[96]:


# Creating a ‘today's date’ the day after the last transaction in our dataset).
last_transaction_date=df['InvoiceDate'].max()
print(last_transaction_date)

#today' s date
snapshot_date = last_transaction_date + dt.timedelta(days=1)
print(snapshot_date)


# In[99]:


# Creating a new dataframe where each row represent unique customer
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'total_price': 'sum'
})

# Renaming columns
rfm_df.rename(columns={'InvoiceDate': 'Recency',
                       'InvoiceNo': 'Frequency',
                       'total_price': 'MonetaryValue'}, inplace=True)
rfm_df.head()


# In[100]:


# Statistic description of DataFrame RFM
print("\nStatistic description of DataFrame RFM")
print(rfm_df.describe())


# The Max of Recency 374  means that there are customers who have not purchased anything for over a year.The mean of Recency (93) is much higher than the median (50th percentile: 51), indicating an asymmetrical distribution: a majority of customers have purchased fairly recently, but a long tail of customers has been inactive for a long time.
# 
# The mean of Frequency (4.2) is higher than the median (50th percentile: 2), which means that most customers buy infrequently, but a few very frequent customers push the mean upwards.
# 
# For the Monetary value  The mean (2048) is much higher than the median (50th percentile: 668). This confirms the EDA hypothesis: there are a few ‘big fish’ who spend a lot.

# In[ ]:





# ## III- Clustering modelling

# ### 1-Data preprocessing

# #### a- logarithmic transformation of data

# In[103]:


# Data visualization before transformation

fig, axes=plt.subplots(1,3, figsize=(18,5))
sns.histplot(rfm_df['Recency'],ax=axes[0], kde=True).set_title('Initial distribution - Recency')
sns.histplot(rfm_df['Frequency'],ax=axes[1], kde=True).set_title('Initial distribution - Frequency')
sns.histplot(rfm_df['MonetaryValue'],ax=axes[2], kde=True).set_title('Initial distribution - MonetaryValue')
plt.show()


# In[106]:


# logarithmic transformation of data
rfm_log=np.log1p(rfm_df)rfm_log.head()


# In[107]:


#Visualization of transformed data
fig, axes= plt.subplots(1,3, figsize=(18,5))
sns.histplot(rfm_log['Recency'], ax=axes[0], kde=True).set_title('log distribution - Recency')
sns.histplot(rfm_log['Frequency'],ax=axes[1], kde=True).set_title('log distribution - Frequency')
sns.histplot(rfm_log['MonetaryValue'],ax=axes[2], kde=True).set_title('log distribution - MonetaryValue')
plt.show()


# #### b- Scaling Data

# In[109]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
rfm_scaled=scaler.fit_transform(rfm_log)

#Transorm it into a DataFrame
rfm_scaled_df=pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=rfm_df.columns)
rfm_scaled_df.head()


# ### 2- Clustering with K-means algorithm

# #### a-Selecting the right number of K

# In[115]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Selecting the right number of k "clusters" with the elbow method
inertias=[]
silhouettes=[]
K=range(2,10)
for k in K:
    kmeanmodel=KMeans(n_clusters=k, random_state=42)
    kmeanmodel.fit(rfm_scaled_df)
    inertias.append(kmeanmodel.inertia_)
    silhouettes.append(silhouette_score (rfm_scaled_df, kmeanmodel.labels_))


# In[116]:


#Inertia curve
plt.plot(range(2,10), inertias)
plt.xlabel('Cluster number')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.show()


# In[118]:


# Silhouette score graph
plt.figure(figsize=(10,4))
plt.plot(K, silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette score showing optimal k')
plt.show()


# We will choose K=4

# #### b-K-Means with the optimal K

# In[119]:


#Final kmeans model
kmodel=KMeans(n_clusters=4, random_state=42)
kmodel.fit(rfm_scaled_df)


# In[121]:


#Gather the labels
cluster_labels=kmodel.labels_
rfm_df['cluster']=cluster_labels

print("\nDataFrame RFM final with clusters :")
print(rfm_df.head())
print("\nSize of each cluster :")
print(rfm_df['cluster'].value_counts())


# In[122]:


# Saving the data set
rfm_df.to_csv('rfm_with_clusters.csv')


# ## IV-Segment Analysis and Interpretation

# #### 1-Analysis of each cluster

# In[123]:


# Cluster summary
cluster_summary=rfm_df.groupby('cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'cluster': 'count'  
}).rename(columns={'cluster': 'Nb_Clients'}).round(2)


# In[124]:


# Percentage of each cluster
cluster_summary['%_Clients'] = (cluster_summary['Nb_Clients'] / cluster_summary['Nb_Clients'].sum() * 100).round(2)

print("\nSummary of clusters 's characteristics :")
print(cluster_summary)

Cluster 0: Average recency, low frequency, low/average amount. These are occasional customers or customers who are losing interest.

Cluster 1: Very low recency (they have purchased recently), very high frequency, very high amount. These are clearly our best customers.

Cluster 2: Average/high recency, average frequency and average/good amount. These are good, loyal customers, but not yet at the level of the best.

Cluster 3: Very high recency (they haven't purchased in a long time), very low frequency, very low amount. These are our lost or very high-risk customers.

# #### 2- Visualization of each cluster 

# In[126]:


# Ploting box plots for each category
fig, axes=plt.subplots(1, 3, figsize=(20,7))
plt.suptitle("Customers's segment comparison by RFM")

#Recency boxplot
sns.boxplot(ax=axes[0], x='cluster', y='Recency', data=rfm_df)
axes[0].set_title('Recency per Cluster\n(Lower = Better)')

#Frequency boxplot
sns.boxplot(ax=axes[1], x='cluster', y='Frequency', data=rfm_df)
axes[1].set_title('Frequency per Cluster\n(Higher = Better)')
axes[1].set_yscale('log')

#Monetary value boxplot
sns.boxplot(ax=axes[2], x='cluster', y='MonetaryValue', data=rfm_df)
axes[2].set_title('Amount per Cluster\n(Higher = Better)')
axes[2].set_yscale('log') 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# These graphs visually confirm what the averages were telling us.

# ### 3-Creation of Personas and Recommendations

# In[127]:


# Persona creation
persona_map = {
    0: 'Occasional Customers',
    1: 'Champions',
    2: 'Loyal Customers',
    3: 'Dormant/At-Risk Customers'
}
#Final data frame
rfm_df['persona']=rfm_df['cluster'].map(persona_map)
rfm_df.head()

Segment 1:  Champions 
Our best customers. They buy very often, very recently, and spend the most. They represent 27% of our customer base but probably a much larger share of our revenue.
Marketing objective: Build loyalty, reward them, and turn them into ambassadors.
Recommended Actions:
Create a VIP programme with exclusive benefits (free delivery, early access to new products).
Send personalised gifts or thank you notes.
Do NOT send them aggressive promotions.

Segment 3: Dormant/At-Risk Customers
 Customers who haven't bought anything in a long time (more than 6 months on average). They are about to be lost permanently.
Marketing objective: Reactivate them with an attractive offer.
Recommended actions:
Launch a ‘We miss you’ campaign with a significant discount offer (e.g. 20% off).
Send a survey to understand why they are no longer buying.
Exclude them from general newsletters so as not to overwhelm them.

Segment 0:  Occasional Customers
They rarely buy, for small amounts, and are not very recent. They have low value.
Marketing objective: Maintain contact without investing too much, try to increase frequency.
Recommended actions:
Include them in content newsletters (blog, advice).
Send them mass promotions.

Segment 2: Loyal Customers 
The foundation of our customer base. They buy regularly, are recent customers and have good monetary value.
Marketing objective: Increase their engagement to turn them into ‘Champions’.
Recommended actions:
Offer them deals based on their previous purchases (cross-selling, upselling).
Encourage them to leave product reviews in exchange for loyalty points.
# In[ ]:




