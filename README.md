# SDSS-quasar-data-Classifier-ML-models
This repo. has codes for classification models trained on the SDSS quasar data.

In this work, we analyse SDSS quasar data based on differnet passbands offered by the SDSS DR16Q dataset (refer Lyke et al. 2020) and apply PCA to the high dimensional data. We train random forest, XGBOOST and pca based linear regression classifiers on the dataset to evaluate the efficiency of radio-loud and radio-quiet quasar classifications. Due to severe class imbalance in the original dataset, we are exploring the role of SMOTE and other resampling techniques on the minority class for improving the performace of the classifier. The latest code above just uses Balanced weights method at this stage for all models. Further work focuses on SMOTE and other techniques .Please cite this work wherever required.

The SDSS data set can be found at https://www.sdss4.org/dr17/algorithms/qso_catalog/

Thank you!
