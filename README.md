# mlpipelines

This repository is to automate and facilitate your job with machine learning models!! 
It is made within notebooks to interact with the user while exploring and doing data cleansing and getting perfomrance  of pre-built ml models (So far it's built only for Binary Classificaiton and uses 2 models: lightGBM and DNN).

Although not with fancy visualization, it may help the users with low coding experience.


# data cleansing ane feature engineering

*import prepa as pr*

You can get your NaN values and transform then with the class *prepa*:

*pr = prepa(df_atp)*

call the *col_nan_errors(mode)* to transforme your NaN columns with the *m_max* or *m_min* value of each colum

*df_atp = pr.cols_nan_errors("m_max")*

<img src="assets/1prepa.png" width="900" height="500" >

Select columns to be categorized and generate new features:

<img src="assets/categ.png" width="1300" >

Here we are selecting columns 0 up to 9 plus the columns "Loser"

<img src="assets/categ3.png" width="900">

See the results right after:

<img src="assets/categ4.png" width="900">


# Running binary classification algorithm

Libraries to be used running your data analyses algorithms with a more user friendly interface.

User input for the model parametrization is allowed through an interative process.

Import the library:

*import binclass as bc*

call the BinaryClass(dataframe,target)

<img src="assets/bc1.png" width="900">

*res = bc.BinaryClass(df,"Labels")*

train your models and get the outputs:

*res.train_models(False,1)*

Get the confusion matrix and AUC & ROC curves at the same place:

<img src="assets/auc.png" width="900">
<img src="assets/roc_curve.png" width="900">

Make your data analyses, change them and rerun it:

*res.train_models(False,2)*

(here we are also running a DNN algorithm, that you can choose to run or not)
<img src="assets/dnn1.png" width="900">

<img src="assets/gbm&dnn.png" width="900">


<img src="assets/bc2.png" width="1300"  >



<img src="assets/categ2.png" width="900" >



# Visualize SHAP values to explain your models

SHAP values are an interesting way to explain why your model is doing the predictions in some way. For more documentation: [SHAP documentation](https://shap.readthedocs.io/en/latest/)

Get the most important features:

<img src="assets/shap_import.png" width="900">

Check SHAP values for single predictions (why your model chose to predict "1" instead of "0")

<img src="assets/shap_single.png" width="900">

Visualize the overall predictions of multiple samples:

<img src="assets/shap_ov.png" width="900">


