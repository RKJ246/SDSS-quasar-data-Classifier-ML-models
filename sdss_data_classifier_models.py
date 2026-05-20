import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


#file_path = r"C:\Users\Joshi Sir\Desktop\data science hw 1 codes\SDSS_quasar_data.txt"
file_path = '/mnt/c/Users/Joshi Sir/Desktop/data science hw 1 codes/SDSS_quasar_data.txt'

data = pd.read_csv(file_path, delim_whitespace=True)

# radio loudness criteria
data['Radio_Loud'] = data['Radio'] > 1

# plotting redshift distribution here
plt.figure(figsize=(8, 5))
sns.histplot(data['z'], bins=30, kde=True)
plt.xlabel('Redshift (z)')
plt.ylabel('Count')
plt.title('Distribution of Redshift')
plt.show()

# plotting HR diagram here
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['g_mag'] - data['r_mag'], y=data['i_mag'], hue=data['Radio_Loud'])
plt.gca().invert_yaxis()
plt.xlabel('g - r Color')
plt.ylabel('i Magnitude')
plt.title('Hertzsprung-Russell Diagram')
plt.show()

# plotting color color diagram here
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['u_mag'] - data['g_mag'], y=data['g_mag'] - data['r_mag'], hue=data['Radio_Loud'])
plt.xlabel('u - g')
plt.ylabel('g - r')
plt.title('Color-Color Diagram')
plt.show()

# plotting redshift vs i-magnitude here
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['z'], y=data['i_mag'], hue=data['Radio_Loud'])
plt.gca().invert_yaxis()
plt.xlabel('Redshift (z)')
plt.ylabel('i Magnitude')
plt.title('Redshift vs. i Magnitude')
plt.show()

# a correlation grid is plotted here
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# applying PCA to photometric magnitudes
features = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

pca = PCA()
pca_features = pca.fit_transform(scaled_features)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()

data['PCA1'], data['PCA2'] = pca_features[:, 0], pca_features[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=data['Radio_Loud'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Analysis')
plt.show()

pca_components = pd.DataFrame(pca.components_, columns=features, index=[f'PCA{i+1}' for i in range(len(features))])
print("PCA Component Loadings:")
print(pca_components)

# random forests

X = data[features]
y = data['Radio_Loud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42) #change these setting as per your will. We have as of now used some defaults and settings we wish
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# pca + logistic regression 
X_pca = data[['PCA1', 'PCA2']]
y = data['Radio_Loud']
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

pca_clf = LogisticRegression(class_weight='balanced', random_state=42) #class weight treats class imbalance
pca_clf.fit(X_train_pca, y_train_pca)

y_pca_pred = pca_clf.predict(X_test_pca)

#print("PCA-based Logistic Regression Classification Report (Balanced):")
print(classification_report(y_test_pca, y_pca_pred, zero_division=0))
print(f"PCA Classifier Accuracy: {accuracy_score(y_test_pca, y_pca_pred):.2f}")

# decision boundry (optional)
#h = .02
#gx, gy = np.meshgrid(
#    np.arange(X_pca['PCA1'].min() - 1, X_pca['PCA1'].max() + 1, h),
#    np.arange(X_pca['PCA2'].min() - 1, X_pca['PCA2'].max() + 1, h)
#)
#Z = pca_clf.predict(np.c_[gx.ravel(), gy.ravel()])
#Z = Z.reshape(gx.shape)

#plt.figure(figsize=(8, 5))
#plt.contourf(gx, gy, Z, alpha=0.3, cmap='coolwarm')
#sns.scatterplot(x='PCA1', y='PCA2', hue='Radio_Loud', data=data, edgecolor='k')
#plt.xlabel('PCA Component 1')
#plt.ylabel('PCA Component 2')
#plt.title('PCA Logistic Regression Decision Boundary')
#plt.show()

# t and ks test 
radio_loud_z = data[data['Radio_Loud']]['z']
radio_quiet_z = data[~data['Radio_Loud']]['z']
t_stat, t_pval = ttest_ind(radio_loud_z, radio_quiet_z, equal_var=False)
#ks_stat, ks_pval = ks_2samp(radio_loud_z, radio_quiet_z)

#print("Hypothesis Testing Results:")
print(f"T-test p-value: {t_pval:.5f}")
#print(f"KS-test p-value: {ks_pval:.5f}")
if t_pval < 0.05:
    print("Reject the null hypothesis: The redshift distributions are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in redshift distributions.")

#class cardinality for data (optional to check class imabalance)
#num_radio_loud = data['Radio_Loud'].sum()
#num_radio_quiet = len(data) - num_radio_loud

#print(f"no. of radio-Loud Quasars: {num_radio_loud}")
#print(f"no. of of radio-Quiet Quasars: {num_radio_quiet}")

# segregating only radio loud quasars from pca plot
radio_loud_data = data[data['Radio_Loud']]

plt.figure(figsize=(8, 5))
sns.scatterplot(x=radio_loud_data['PCA1'], y=radio_loud_data['PCA2'], color='red', label='Radio-Loud')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Projection (Radio-Loud Quasars Only)')
plt.legend()
plt.show()

# pca component wise explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("PCA Explained Variance by Component:")
for i, var in enumerate(explained_var):
    print(f"  PCA{i+1}: {var:.4f} (Cumulative: {cumulative_var[i]:.4f})")

#n_components_97 = np.argmax(cumulative_var >= 0.97) + 1
#print(f"\nNumber of components explaining ≥97% of the variance: {n_components_97}")

# class cardinality for pca set (optional to check class imbalance in pca treated data)

#num_radio_loud_pca_test = y_test_pca.sum()
#num_radio_quiet_pca_test = len(y_test_pca) - num_radio_loud_pca_test
#print(f"radio-loud quasars in pca dataset: {num_radio_loud_pca_test}")
#print(f"radio-quiet quasars in pca dataset: {num_radio_quiet_pca_test}")

# plotting pca eigenvectors

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=data['Radio_Loud'], edgecolor='k')

scaling_factor = 7  # adjust length of vector 
colors = ['red', 'blue']  
for i, eigenvector in enumerate(pca.components_[:2]):  # change no of components if required
    plt.arrow(0, 0, eigenvector[0]*scaling_factor, eigenvector[1]*scaling_factor,
              color=colors[i], width=0.05, head_width=0.3, head_length=0.3,
              length_includes_head=True, label=f'Eigenvector {i+1}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Components and Eigenvectors')
plt.legend()
plt.show()


# roc and prc curve

from sklearn.metrics import roc_curve, auc, precision_recall_curve
y_rf_proba = clf.predict_proba(X_test)[:, 1]

#random forest

# roc of random forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 5))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# prc of random forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_rf_proba)

plt.figure(figsize=(8, 5))
plt.plot(recall_rf, precision_rf, color='purple', label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()

#pca based linear regression (pbc) 

y_pca_proba = pca_clf.predict_proba(X_test_pca)[:, 1]

# roc of pbc
fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, y_pca_proba)
roc_auc_pca = auc(fpr_pca, tpr_pca)

plt.figure(figsize=(8, 5))
plt.plot(fpr_pca, tpr_pca, label=f'PCA-LogReg (AUC = {roc_auc_pca:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - PCA Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# prc of pbc
precision_pca, recall_pca, _ = precision_recall_curve(y_test_pca, y_pca_proba)

plt.figure(figsize=(8, 5))
plt.plot(recall_pca, precision_pca, color='blue', label='PCA Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - PCA Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()


# this code is for handling class imabalance by using SMOTE (optional)

#from imblearn.over_sampling import SMOTE

#smote = SMOTE(random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# random forest on SMOTE applied data

#clf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
#clf_smote.fit(X_resampled, y_resampled)
#y_pred_smote = clf_smote.predict(X_test)
#y_proba_smote = clf_smote.predict_proba(X_test)[:, 1]

#print("Random Forest (SMOTE)")
#print(classification_report(y_test, y_pred_smote))
#print(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.2f}")

# roc of random forest + smote
#fpr_rf_s, tpr_rf_s, _ = roc_curve(y_test, y_proba_smote)
#roc_auc_rf_s = auc(fpr_rf_s, tpr_rf_s)

#plt.figure(figsize=(8, 5))
#plt.plot(fpr_rf_s, tpr_rf_s, label=f'RF SMOTE (AUC = {roc_auc_rf_s:.2f})', color='darkred')
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve - Random Forest with SMOTE')
#plt.legend()
#plt.grid(True)
#plt.show()

# prc of random forest+smote
#precision_rf_s, recall_rf_s, _ = precision_recall_curve(y_test, y_proba_smote)

#plt.figure(figsize=(8, 5))
#plt.plot(recall_rf_s, precision_rf_s, color='maroon', label='RF with SMOTE')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve - Random Forest with SMOTE')
#plt.legend()
#plt.grid(True)
#plt.show()


# pca on smote applied data
#smote_pca = SMOTE(random_state=42)
#X_pca_resampled, y_pca_resampled = smote_pca.fit_resample(X_train_pca, y_train_pca)

# logistic regression on PCA+SMOTE
#pca_clf_smote = LogisticRegression(class_weight='balanced', random_state=42)
#pca_clf_smote.fit(X_pca_resampled, y_pca_resampled)
#y_pca_pred_smote = pca_clf_smote.predict(X_test_pca)
#y_pca_proba_smote = pca_clf_smote.predict_proba(X_test_pca)[:, 1]

#print("PCA-based Logistic Regression (SMOTE)")
#print(classification_report(y_test_pca, y_pca_pred_smote, zero_division=0))
#print(f"PCA Classifier Accuracy: {accuracy_score(y_test_pca, y_pca_pred_smote):.2f}")

# roc of pca+smote
#fpr_pca_s, tpr_pca_s, _ = roc_curve(y_test_pca, y_pca_proba_smote)
#roc_auc_pca_s = auc(fpr_pca_s, tpr_pca_s)

#plt.figure(figsize=(8, 5))
#plt.plot(fpr_pca_s, tpr_pca_s, label=f'PCA LogReg SMOTE (AUC = {roc_auc_pca_s:.2f})', color='darkgreen')
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve - PCA Logistic Regression with SMOTE')
#plt.legend()
#plt.grid(True)
#plt.show()

# prc of pca+smote
#precision_pca_s, recall_pca_s, _ = precision_recall_curve(y_test_pca, y_pca_proba_smote)

#plt.figure(figsize=(8, 5))
#plt.plot(recall_pca_s, precision_pca_s, color='teal', label='PCA LogReg with SMOTE')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve - PCA Logistic Regression with SMOTE')
#plt.legend()
#plt.grid(True)
#plt.show()

#this section is dedicated to handling class imbalance using xgboost (optional)

#import xgboost as xgb
#from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pca+xgboost

# applying pca
#pca = PCA(n_components=0.97)  # change the variance to be retained after pca as per requirement
#X_train_pca = pca.fit_transform(X_train)
#X_test_pca = pca.transform(X_test)

# xgboost+pca
#xgb_pca_clf = xgb.XGBClassifier(random_state=42)
#xgb_pca_clf.fit(X_train_pca, y_train)
#y_pred_pca = xgb_pca_clf.predict(X_test_pca)
#y_proba_pca = xgb_pca_clf.predict_proba(X_test_pca)[:, 1]

#print("XGBoost (PCA-based)")
#print(classification_report(y_test, y_pred_pca))
#print(f"Accuracy: {accuracy_score(y_test, y_pred_pca):.2f}")

# roc of pca+xgboost
#fpr_pca, tpr_pca, _ = roc_curve(y_test, y_proba_pca)
#roc_auc_pca = auc(fpr_pca, tpr_pca)

#plt.figure(figsize=(8, 5))
#plt.plot(fpr_pca, tpr_pca, label=f'XGBoost (PCA) (AUC = {roc_auc_pca:.2f})', color='purple')
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve - XGBoost (PCA-based)')
#plt.legend()
#plt.grid(True)
#plt.show()

# prc of pca+xgboost
#precision_pca, recall_pca, _ = precision_recall_curve(y_test, y_proba_pca)

#plt.figure(figsize=(8, 5))
#plt.plot(recall_pca, precision_pca, color='blueviolet', label='XGBoost (PCA)')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve - XGBoost (PCA-based)')
#plt.legend()
#plt.grid(True)
#plt.show()


# applying xgboost on random forests

# initialize random forest classifier
#rf_clf = RandomForestClassifier(random_state=42)
#rf_clf.fit(X_train, y_train)
#y_pred_rf = rf_clf.predict(X_test)
#y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

# random forest + xgboost
#xgb_rf_clf = xgb.XGBClassifier(random_state=42)
#xgb_rf_clf.fit(y_pred_rf.reshape(-1, 1), y_test)  # xgboost on random forest probs.
#y_pred_rf_xgb = xgb_rf_clf.predict(y_pred_rf.reshape(-1, 1))
#y_proba_rf_xgb = xgb_rf_clf.predict_proba(y_pred_rf.reshape(-1, 1))[:, 1]

#print(" XGBoost (Random Forest-based)")
#print(classification_report(y_test, y_pred_rf_xgb))
#print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_xgb):.2f}")

# roc of xgboost+random forest
#fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf_xgb)
#roc_auc_rf = auc(fpr_rf, tpr_rf)

#plt.figure(figsize=(8, 5))
#plt.plot(fpr_rf, tpr_rf, label=f'XGBoost (RF) (AUC = {roc_auc_rf:.2f})', color='green')
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve - XGBoost (Random Forest-based)')
#plt.legend()
#plt.grid(True)
#plt.show()

# prc of xgboost+random forest
#precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf_xgb)

#plt.figure(figsize=(8, 5))
#plt.plot(recall_rf, precision_rf, color='darkorange', label='XGBoost (Random Forest-based)')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve - XGBoost (Random Forest-based)')
#plt.legend()
#plt.grid(True)
#plt.show()
