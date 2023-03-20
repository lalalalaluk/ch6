#Customer Segmentation: A Technical Guide With Python Examples(https://www.mktr.ai/applications-and-methods-in-data-science-customer-segmentation/)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import VisualizeHelper.VisualizeCluster as VisualizeCluster

# read-in the excel spreadsheet using pandas 
df = pd.read_excel(r'./Datasets/Online Retail.xlsx', sheet_name='Online Retail')
print(df.head())

# Data Cleanup
# Drop cancelled orders
df = df.loc[df['Quantity'] > 0]

# Drop records without CustomerID
df = df[pd.notnull(df['CustomerID'])]

# Drop incomplete month
df = df.loc[df['InvoiceDate'] < '2011-12-01']

# Calculate total sales from the Quantity and UnitPrice
df['Sales'] = df['Quantity'] * df['UnitPrice']

# use groupby to aggregate sales by CustomerID
customer_df = df.groupby('CustomerID').agg({'Sales': sum, 'InvoiceNo': lambda x: x.nunique()})

# Select the columns we want to use
customer_df.columns = ['TotalSales', 'OrderCount'] 

# create a new column 'AvgOrderValu'
customer_df['AvgOrderValue'] = customer_df['TotalSales'] / customer_df['OrderCount']
print(customer_df.head())

# Normalize data
rank_df = customer_df.rank(method='first')
normalized_df = (rank_df - rank_df.mean()) / rank_df.std()
print(normalized_df.head())

#=======
VisualizeCluster.PlotElbow(normalized_df)
#=======

#create model
model = KMeans(n_clusters=4).fit(normalized_df)
#four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
normalized_df['Cluster'] = model.labels_
print(normalized_df.sample(10))

#=======
VisualizeCluster.PlotCluster(normalized_df)
#=======

#Find the best-selling item by cluster
for n in range(4):
    cluster = normalized_df.loc[normalized_df['Cluster'] == n]
    BestSellingItemByCluster=pd.DataFrame(df.loc[df['CustomerID'].isin(cluster.index)].groupby(
        'Description').count()['StockCode'].sort_values(ascending=False))
    print(BestSellingItemByCluster.head())