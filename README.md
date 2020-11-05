# mlpipelines

This repository is to automate and facilitate your job with machine learning models!! 
It is made within notebooks to interact with the user while exploring and doing data cleansing and getting perfomrance  of pre-built ml models (So far it's built only for Binary Classificaiton and uses 2 models: lightGBM and DNN).

Although not with a fancy visual interface for interaction, it may help the users with low coding experience.
User input for the model parametrization is allowed through an interative process.

Colab Notebook example: click [here](https://colab.research.google.com/drive/17lxCLGH25MlRo_imPJSvQTrZp02INm62?usp=sharing)

# data cleansing ane feature engineering

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

call the BinaryClass(dataframe,target) with your data --> if you don't specify the *outcome* column, you can input it right below:

<img src="assets/bc1.png" width="900">

If you didn't specify the columns/variables to train your model, don't worry!! It will let you do so selecting them!

<img src="assets/bc2.png" width="1300"  >

train your models and get the outputs:

*bc.train_models(False,1)*

Get the confusion matrix and AUC & ROC curves at the same place:

<img src="assets/auc.png" width="900">
<img src="assets/roc_curve.png" width="900">

Make your data analyses, change them and rerun it:

*bc.train_models(False,2)*

(here we are also running a DNN algorithm, that you can choose to run or not)
<img src="assets/dnn1.png" width="900">

<img src="assets/gbm&dnn.png" width="900">


# Visualize SHAP values to explain your models

SHAP values are an interesting way to explain why your model is doing the predictions in some way. For more documentation: [SHAP documentation](https://shap.readthedocs.io/en/latest/)

Get the most important features:

<img src="assets/shap_import.png" width="900">

Check SHAP values for single predictions (why your model chose to predict "1" instead of "0")

<img src="assets/shap_single.png" width="900">

Visualize the overall predictions of multiple samples:

<img src="assets/shap_ov.png" width="900">


