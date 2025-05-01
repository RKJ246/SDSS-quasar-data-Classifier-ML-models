import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# mention your model loadings here
pca_loadings = pd.DataFrame({
    'PCA1': [0.391393, 0.459222, 0.468072, 0.461977, 0.451013],
    'PCA2': [0.802036, 0.232939, -0.194673, -0.322953, -0.400352],
    'PCA3': [-0.428475, 0.686106, 0.311459, -0.180918, -0.464682],
    'PCA4': [-0.122102, 0.450463, -0.509533, -0.409582, 0.595645],
    'PCA5': [-0.071062, 0.247379, -0.621595, 0.694111, -0.256092],
}, index=['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag'])

# transpose pca components to x axis (optional)
#pca_loadings = pca_loadings.T

# creating heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pca_loadings, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.xlabel("PCA Components")
plt.ylabel("Photometric Bands")
plt.title("Heatmap of PCA Loadings")
plt.tight_layout()
plt.show()
