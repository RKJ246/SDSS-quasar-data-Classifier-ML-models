# SDSS-quasar-data-Classifier-ML-models
This repo. has codes for classification models trained on the SDSS quasar data.

In this work, we analyse SDSS quasar data based on differnet passbands offered by the SDSS DR16Q dataset (refer Lyke et al. 2020) and apply PCA to the high dimensional data. We train random forest, XGBOOST and pca based logistic regression classifiers on the dataset to evaluate the efficiency of radio-loud and radio-quiet quasar classifications. Due to severe class imbalance in the original dataset, we are exploring the role of SMOTE and other resampling techniques on the minority class for improving the performace of the classifier. The latest code above just uses Balanced weights method at this stage for all models. Further work focuses on SMOTE and other techniques .Please cite this work wherever required.

The SDSS data set can be found at https://www.sdss4.org/dr17/algorithms/qso_catalog/

Please cite the paper as "Joshi, R., Shinde, V. A statistical investigation of radio-loudness in SDSS DR16Q quasars using ML-based classification and physical feature analysis. Eur. Phys. J. C 86, 766 (2026). https://doi.org/10.1140/epjc/s10052-026-15968-7" if you are using this work!

Note: Claude and ChatGPT have been used for finessing of the code, introducing some plotting styles, tidying up the variables and function calls, and neatly printing relevant tables. The idea and execution (employing classifiers, statistical analysis, separability metric and data selection, pre-processing etc.) is entirely authors'.

Thank you!
