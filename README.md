Alzheimer’s disease can be predicted with good accuracy by examining the thickness of cerebral cortex regions obtained through brain scans. In this report, I try to show different approaches and models that I used to predict Alzheimer’s disease.


Due to the redundant information and high dimensionality of the data, models, even simpler ones tend to overfit, thereby necessitating regularization and dimensionality reduction. I used simple models of Elastic net, Logistic regression, Linear discriminant analysis (LDA), and Support Vector Machines (SVM), as well as ensembled tree-based models including XGBoost and Random Forest models in this study to make a comparison of best-performing models. I also used PCA and autoencoders as dimensionality reduction techniques, which then fit within the models for all models except elastic net.

PCA and variational autoencoders have shown similar information (variance) preservation and similar scores indicating close to linear relationship among features. Hyperparameter tuning is implemented using grid-search-cv to select the best hyperparameters for model selection as well as dimensionality reduction.

Elastic-net model has shown one of the best performances. Similarly simpler models of logistic regression, LDA, and SVM showed the best performance, while RF and XGBoost tend to generalize less with a high risk of overfitting. Finally, LDA and logistic regression tend to have a high AUC score indicating a balanced prediction accuracy for both classes, in addition to the best overall accuracy.

 I used mlflow as well as manual experiments of tracking hyperparameter tuning. I have added the hyperparameter experiment documents and codes together.
