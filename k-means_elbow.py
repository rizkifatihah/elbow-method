# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('name_file.csv')
data.head()

mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

Distortion = []
K = range(1,8)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Distortion.append(km.inertia_)

plt.plot(K, Distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeanModel = KMeans(n_clusters=3)
kmeanModel.fit(data)

data['k_means']=kmeanModel.predict(data)
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(data['x'], data['y'], c=data['k_means'])
axes[1].scatter(data['x'], data['y'], c=data['k_means'], cmap=plt.cm.Set1)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K_Means', fontsize=18)

