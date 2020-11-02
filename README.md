# mlpipelines

This repository is to automate and facilitate your job with machine learning models!! 
It is made within notebooks to interact with the user while exploring and doing data cleansing and getting perfomrance  of pre-built ml models (So far it's built only for Binary Classificaiton and uses 2 models: lightGBM and DNN).

Although not with fancy visualization, it may help the users with low coding experience.


# data cleansing ane feature engineering

You can get your NaN values and transform then with the class *prepa*:

<img src="assets/bc1.png" width="800" >

Libraries to be used running your data analyses algorithms with a more user friendly interface.

User input for the model parametrization is allowed through an interative process.

Import the library:

*import binclass as bc*

call the BinaryClass(dataframe,target)

<img src="assets/categ.png" width="1300" >

*res = bc.BinaryClass(df,"Labels")*

Run the algorithm:

*res.train_models(False,1)*

over = False, for overfitting or not
n = 1, the model number to get and save the results for further comparison

Make your data analyses, change them and rerun it:

*res.train_models(False,2)*

<img src="assets/bc2.png" width="1300"  >

<img src="assets/1prepa.png" width="900" height="500" >
<img src="assets/auc.png" width="900">
<img src="assets/gbm&dnn.png" width="900">
<img src="assets/categ2.png" width="900" >

<img src="assets/categ4.png" width="900">
<img src="assets/categ3.png" width="900">
<img src="assets/dnn1.png" width="900">

<img src="assets/no_categ.png" width="900">


<img src="assets/roc_curve.png" width="900">


<img src="assets/shap_import.png" width="900">

<img src="assets/shap_ov.png" width="900">

<img src="assets/shap_single.png" width="900">
