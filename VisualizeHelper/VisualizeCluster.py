from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

def PlotElbow(normalized_df):
    sse = []
    krange = list(range(2,11))
    X = normalized_df[['TotalSales','OrderCount','AvgOrderValue']].values
    for n in krange:
        model = cluster.KMeans(n_clusters=n, random_state=3)
        model.fit_predict(X)
        cluster_assignments = model.labels_
        centers = model.cluster_centers_
        sse.append(np.sum((X - centers[cluster_assignments]) ** 2))
        #sse.append(model.inertia_)

    plt.plot(krange, sse)
    plt.xlabel("$K$")
    plt.ylabel("Sum of Squares")
    plt.savefig('PlotElbow.png')

def PlotCluster(cluster_df):
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 0]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 0]['TotalSales'],
    c='blue')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 1]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 1]['TotalSales'],
    c='red')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 2]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 2]['TotalSales'],
    c='orange')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 3]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 3]['TotalSales'],
    c='green')
    plt.title('TotalSales vs. OrderCount Clusters')
    plt.xlabel('Order Count')
    plt.ylabel('Total Sales')
    plt.grid()
    plt.savefig('PlotCluster1.png')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 0]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='blue')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 1]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='red')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 2]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='orange')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 3]['OrderCount'], 
    cluster_df.loc[cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='green')
    plt.title('AvgOrderValue vs. OrderCount Clusters')
    plt.xlabel('Order Count')
    plt.ylabel('Avg Order Value')
    plt.grid()
    plt.savefig('PlotCluster2.png')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 0]['TotalSales'], 
    cluster_df.loc[cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='blue')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 1]['TotalSales'], 
    cluster_df.loc[cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='red')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 2]['TotalSales'], 
    cluster_df.loc[cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='orange')
    plt.scatter(
    cluster_df.loc[cluster_df['Cluster'] == 3]['TotalSales'], 
    cluster_df.loc[cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='green')
    plt.title('AvgOrderValue vs. TotalSalFs Clusters')
    plt.xlabel('Total Sales')
    plt.ylabel('Avg Order Value')
    plt.grid()
    plt.savefig('PlotCluster3.png')