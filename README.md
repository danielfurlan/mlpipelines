# mlpipelines

This repository is to automate machine learning models!!

Libraries to be used running your data analyses algorithms with a more user friendly interface.

User input for the model parametrization is allowed through an interative process.

Import the library:

*import binclass as bc*

call the BinaryClass(dataframe,target)

*res = bc.BinaryClass(df,"Labels")*

Run the algorithm:

over = False, for overfitting or not
n = 1, the model number to get and save the results for further comparison

*res.train_models(False,1)*

Make your data analyses, change them and rerun it:

*res.train_models(False,2)*
