import lightgbm as lgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

conf = {}
acu = {}
roc = {}
bag = {}
y_predictions = {}
bag["gbm"] = {}
roc["gbm"] = {}
bag["dnn"] = {}
roc["dnn"] = {}

class BinaryClass:
    
    global conf
    global auc
    global roc
    global y_predictions
    global bag
    
    def __init__(self,dataframe,target=None,reset=None,feature_cols=None):
        
        """
        args:
            dataframe : pandas dataframe
            target : (str) the column target to perform the classification
            feature_cols : (list) the columns to train the model
        """
        bag["gbm"] = {}
        roc["gbm"] = {}
        bag["dnn"] = {}
        roc["dnn"] = {}
        self.dataframe = dataframe

        if target == None:
            print("What is your target columns?")
            display(self.dataframe.columns)
            
            ans = str(input())
            while ans not in self.dataframe.columns:
                print("Please, enter a valid column name! Pay attention with spaces!")
                ans = str(input())
            self.target = ans
#             self.feature_cols = dataframe.columns.drop(self.target)
        else:
            self.target = target
#             self.feature_cols = dataframe.columns.drop(self.target)
        
        if feature_cols == None:
            print("You didn't specified any columns for the features to train your model. I'll thus only drop the target column and get all the rest. If you would like to specify the feature columns, please enter below. (y/n)")
#             print("\ndata columns : ", pd.Series(self.dataframe.columns))
            
            ans = input()
            if ans == 'y':
                print("\n\nThese are the columns you have : \n\n")
                print(f'{[(idx,x) for idx,x in pd.Series(self.dataframe.columns).items()]}')
                self.feature_cols = get_cols(self.dataframe.columns)
            else:
                self.feature_cols = self.dataframe.columns.drop(self.target)
                print("All right. Then, your feature cols are set automatically : \n\n", self.feature_cols)
                
        else:
            self.feature_cols = feature_cols
            print("Nice! I see that you have already set up the feature columns!! Thank you to save my time =) ")
            
        
        self.train, self.valid, self.test = self.get_data_splits(self.dataframe)
        
    def get_data_splits(self,dataframe, valid_fraction=0.1):
        
        valid_fraction = 0.1
        valid_size = int(len(dataframe) * valid_fraction)

        train = dataframe[:-valid_size * 2]
        # valid size == test size, last two sections of the data
        valid = dataframe[-valid_size * 2:-valid_size]
        test = dataframe[-valid_size:]
        print(f"Train size : {len(train)}\nValidation size : {len(valid)}\nTest size : {len(test)}")
        return train, valid, test
    
    def train_models(self,over,n):
        """
        args:
            over: boolean,'True' or 'False', parameter to set condition if the model is overfitting or not
            n : int, number of the model
        """
        
        gbm = GBM(self.train,self.valid,self.test,self.target,self.feature_cols)

        gbm.train_GBM(over,n)
        print("Lenght of self.featcols = ", len(self.feature_cols))
        print("Do you want to run a DNN algorithm? (y/n)" )
        answer = input()
        if answer == 'y':
            dnn = DNN(self.train,self.valid,self.test,self.target,len(self.feature_cols),self.feature_cols)
            
            dnn.train_DNN(n)
        print("Do you want to get SHAP explanations for single predictions?")
        answer = input()
        if answer == 'y':
            sh = SHAP(bag)
            sh.single(n,1,self.target,self.feature_cols)
        print("\nIf you would also like to see overral explanations, please enter 'ov'. Otherwhise, tap any other key")
        ans = input()
        if ans == 'ov':
            sh = SHAP(bag)
            sh.overall(self.feature_cols,self.target)
            
    def reset(self):
        conf = {}
        acu = {}
        roc = {}
        y_predictions = {}
        bag = {}

            
import pandas
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DNN:
    
    def __init__(self,train,valid,test,target,inp_dim,feat_cols=None):
        
        """
        args:
            train,valid,test: (pandas df) data splitted
            target : (str) name of column to target the classification
            inp_dim : (int) dimension of input (usually the lenght of your feature columns used to train the model)
        """
        
        self.train = train
        self.valid = valid
        self.test = test
        self.target = target
        self.inp_dim = inp_dim
        
        if type(feat_cols)!= object or feat_cols == None:
            self.feat_cols = self.train.columns.drop(self.target)
        else:
            self.feat_cols = feat_cols
        
    def train_DNN(self,n):
        """
        args:
            n: (int) the model number
        """
        
        #global n
    
        print("\n\n You are now running a DNN algorithm!\n\n")
        #train,valid, test = get_data_splits(dt)
        
        #X = self.train.drop(self.target,axis=1) #[:,0:60].astype(float)
        X = pd.DataFrame(self.train[self.feat_cols])
        Y = pd.DataFrame(self.train[self.target])
    # define model
        model = Sequential()

        model.add(Dense(self.inp_dim, activation='relu'))
        model.add(Dense(60,activation='relu'))#input_dim=60
        model.add(Dense(1, activation='sigmoid'))    
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        history = model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
        
        # evaluate the model
        scores = model.evaluate(X, Y, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # save model and architecture to single file
        model.save("model.h5")
        print("Saved model to disk")
        bag["dnn"][n] = {"train":self.train,"valid":self.valid,"test":self.test,"model":model} 
        #print("bag dnn 1 : ", bag["dnn"][n])
        #### Get validation scores
        X_valid = np.array(self.valid[self.feat_cols])

        from sklearn.preprocessing import MinMaxScaler

        y_pred = np.array(model.predict_proba(X_valid))

    #     print("lenght ynew : ", len(y_pred))
    # print("X=%s, Predicted=%s" % (Xnew.iloc[0], ynew[0]))

        pl = plot(history.history['accuracy'],y_pred,model,self.target,"dnn")
        pl.graphs(n)

        
        
class plot:
    
    global conf
    global auc
    global roc
    global y_predictions
    global bag
    #global n
    
    def __init__(self,evals_result,valid_pred,model,target,m_type):
        self.evals_result = evals_result
        self.valid_pred = valid_pred
        self.model = model
        self.target = target
        self.m_type = m_type

    def graphs(self,n):
        ################################################### SMESSO QUI!!!!
        
        if self.evals_result != None and (type(self.evals_result) == dict ):
            acu[n] = self.evals_result
            fig1 = plt.figure(figsize=(45,10))
        #print('Plot metrics during training... Our metric : ', param["metric"])
        #print("evals_ results : ", evals_result)
            lgb.plot_metric(self.evals_result, metric='auc',figsize=(35,10))
            plt.xlabel('Iterations',fontsize=20)
            plt.ylabel('auc',fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title("AUC during training",fontsize=20)
            plt.legend(fontsize=20)
            plt.show()
        else:
            print("We are in 'else' for plot DNN")
            fig = plt.figure(figsize=(35,10))
    # history.history['accuracy']
            plt.plot(self.evals_result, color='blue', label='train')

            plt.xlabel('Epochs',fontsize=20)
            plt.ylabel('auc',fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title("AUC during training",fontsize=20)
            plt.legend(fontsize=20)

            plt.show()
            


            ##### CONFUSION MATRIX
        th = 0.5
        y_pred_class = self.valid_pred > th
        y_predictions[n] = y_pred_class
        cm = confusion_matrix(bag[self.m_type][n]["valid"][self.target], y_pred_class)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        tnr = tn / (tn + fp)
        tpr = tp / (tp + fn)
        numberModel = n
        conf[n] = {'fpr':f'{fpr:.3f}','fnr': f'{fnr:.3f}', 'tnr' : f'{tnr:.3f}', "tpr": f'{tpr:.3f}'}
        if n > 1 and fpr != 0 and fnr != 0 and tnr != 0 and tpr != 0:
            conf["ratio " + str(n) + "/" + str(n-1)] = {"fp":f'{float(conf[n]["fpr"])/float(conf[n-1]["fpr"]):.3f}', \
                                                        "fn":f'{float(conf[n]["fnr"])/float(conf[n-1]["fnr"]):.3f}', \
                                                        "tn":f'{float(conf[n]["tnr"])/float(conf[n-1]["tnr"]):.3f}', \
                                                        "tp":f'{float(conf[n]["tpr"])/float(conf[n-1]["tpr"]):.3f}'}
        
        fig2 = plt.figure(figsize=(35,10))
        fig2.add_subplot(1,2,1)
        sns.heatmap(cm, annot = True, fmt='d', cmap="Blues", vmin = 0.2,linewidths=.5,annot_kws={"fontsize": 20}); #cbar_kws={"fontsize": 20},annot_kws={"fontsize": 20}
        sns.set(font_scale=2)
        plt.title(f'Confusion Matrix Model {n} {self.m_type}',fontsize=20)
        plt.ylabel('True Class',fontsize=20)
        plt.xlabel('Predicted Class',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.text(0.1, 0.3, f' FPR: {fpr:.3f}\n FNR: {fnr:.3f}\n TNR: {tnr:.3f}\n TPR: {tpr:.3f}', style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}, fontsize=14)
        
        
        #Print Area Under Curve
        fig2.add_subplot(1,2,2)

        false_positive_rate, recall, thresholds = roc_curve(bag[self.m_type][n]["valid"][self.target], self.valid_pred)
        roc_auc = auc(false_positive_rate, recall)
        roc[self.m_type][n] = {'fpr':false_positive_rate,'recall':recall}
        
        plt.title('Receiver Operating Characteristic (ROC)')
        
        
        #for m in range(1,len(bag)+1):
        for key in bag.keys():
            for ma in range(1,len(bag[key])+1):
                print(f'This is the lenght of our {key} : {len(bag[key])}')
#                 for a in range(1,n+1):
                plt.plot(roc[key][ma]['fpr'], roc[key][ma]['recall'], 'b', label = f'Model {ma} {key}')
            
    #             plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0,1], [0,1], 'r--')
                plt.xlim([0.0,1.0])
                plt.ylim([0.0,1.0])
                plt.ylabel('Recall',fontsize=20)
                plt.xlabel('Fall-out (1-Specificity)',fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
        
        plt.show() 
        print("\n\nThese are the values of the confusion matrix: (if you're analysing more than 1 model - of the same type - , the 'ratio' parameter is the ratio btw the current & the previous model)\nconf : ")
        display(conf) 

import re
def get_cols(colu):
     
    sel = []
    
    print("\n\nPlease write the name - separeted only with a comma - or the first and last index of list without space or comma)\n\n"\
            "Example : '0:6' will grab columns from index 0 to 6\n"\
            "'Column1,Column4' will grab ONLY two columns, column 1 and column 4\n",
            "'0:6+Column8,Column10' will grab the first 6 columns plus the 8th and 10th\n"\
            "'0:6+8:10' will grab the first 6 columns plus the 8th,9th and 10th columns\n"
            )
    ans = input()
    ans_s = re.split(':|\+|,',ans)
    print("This is our splited str : ", ans_s)
    print("leng : ", len(ans_s))
    i=0
    while i < len(ans_s):
        print("Still true. i = ", i)
        if ans_s[i].isdigit() :
            idx1 = int(ans_s[i])
            idx2 = int(ans_s[i+1])
            sel= sel + [x for x in colu[idx1:idx2]]
            i = i + 2
        else:
            sel.append(ans_s[i])
            i = i + 1
    print("\n\nOk! This is the columns you selected : \n\n", sel)
    return sel
        
      

import shap
class SHAP:

    global bag
    
    def __init__(self,bag):
        self.bag = bag
        
        """
        Args:
            bag = dictionary with train,valid,test and model
        """    
    
    def most_import(self,shap_values,feature_cols,data):
               
        feature_cols = feature_cols
        display(shap.summary_plot(shap_values, data,feature_names = feature_cols))

    
    def overall(self,feature_cols,target):
        """
        m1 : number of model 1
        m2 : number of model 2
        """
        m1,m2 = 1,1
        print("\nI'll show up the overrall shap values for only the first model. If you would like to select (a) specific model(s), please enter 'y'")
        ans = input()
        if ans == 'y':
            print("\nOk. So put your models numbers - like '13' if you want to analyse the model 1 and 3. If you want only to see one type it twice - '22' - to see the model 2")
            ans = input()
            m1 = int(ans[0])
            m2 = int(ans[1])
        
        for i in range(m1,m2+1):
            feature_cols = bag['gbm'][i]["valid"].columns.drop(target)
            explainer = shap.TreeExplainer(bag['gbm'][i]["model"])
            data = bag['gbm'][i]['valid'][feature_cols]
            shap_values = explainer.shap_values(data)
            print(f'\n\nModel number {i}')
            display(shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], feature_cols)) #### we're taking 1
    
    
    def single(self,n,row,target,feature_cols):
        """
        args:
            n = number of models
            row = row to visualize
            target = the column target to perform the classification
        """
        if type(feature_cols)!= object or feature_cols == None:
            feature_cols = bag['gbm'][1]["valid"].columns.drop(target)
        else:
            feature_cols = feature_cols
            
        shap.initjs()
        for key in bag:
            for i in range(1,n+1):
                
                if key == 'dnn':
                    self.sh_dnn(i,feature_cols)

                else:
                    
                    row_to_show = row      ############# I'm looking here at the very first sample of our validation set! Change it to play with other examples! 
                    
                    
                    data_for_prediction = self.bag[key][i]["valid"][feature_cols].iloc[row_to_show]
                    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
                    idx = bag[key][i]["valid"].index[0]
                    print(f"Model number : {i}\nprediction of model : ",bag[key][i]["model"].predict(data_for_prediction_array), " ground truth : ", bag[key][i]["valid"][target][idx + row_to_show])
                    data_ser = data_for_prediction.values.reshape(1, -1)
    
                    explainer = shap.TreeExplainer(bag[key][i]["model"])

                    shap_values = explainer.shap_values(data_ser)

                    #display(explainer.expected_value,shap_values)    #### Uncomment to see what the shap_values are about!! It's an array with all the variables' shap values!

                    display(shap.force_plot(explainer.expected_value[1], shap_values[1], data_ser, feature_names = feature_cols))
                if key != 'dnn':
                    print("\nAnd this is the most important features and their importance on the model's predictions: (remember that blue is how much that feature contribbuted for predicting as '0' - or negative - whereas red is for '1' - positive prediction)")
                    self.most_import(shap_values,feature_cols,self.bag[key][i]["valid"][feature_cols])
        
    def sh_dnn(self,i,feature_cols):
        print("\nSorry, but we are still in progress to get SHAP values for DNN models!!")
        """TODO"""

        
        
import lightgbm as lgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt 

class GBM:
    

    def __init__(self,train,valid,test,target,feat_cols=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.target = target
        
        if type(feat_cols)!= object or feat_cols == None:
            self.feature_cols = self.train.columns.drop(self.target)
        else:
            self.feature_cols = feat_cols
        

            
    def train_GBM(self,over,n):
            
        dtrain = lgb.Dataset(self.train[self.feature_cols], label=self.train[self.target])
        dvalid = lgb.Dataset(self.valid[self.feature_cols], label=self.valid[self.target])
    
        if over:
            param = {'num_leaves': 31, 'objective': 'binary', "max_depth": 3,
             'metric': 'auc', 'seed': 7, 'reg_alpha':0.8, 'reg_lambda':0.8}
            print(f"Regularization : l1 = {param['reg_alpha']}, l2 = {param['reg_lambda']}")
        else:
            param = {'num_leaves': 64, 'objective': 'binary',
             'metric': 'auc', 'seed': 7}
            print(f"No regularization!")
        
        evals_result = {} 
        bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid,dtrain], 
                    early_stopping_rounds=10, verbose_eval=10, evals_result=evals_result)
        nameModel = "Model " + str(n) +".txt"
        import joblib
        # save model
        joblib.dump(bst, nameModel)
        
#         bag[n] = {"train":self.train,"valid":self.valid,"test":self.test,"model":bst}
        bag["gbm"][n] = {"train":self.train,"valid":self.valid,"test":self.test,"model":bst}
        #print("bag gbm 1 : ", bag["gbm"][n])
        
        self.evaluate_GBM(bst,evals_result,n)  
        
    def evaluate_GBM(self,bst,evals_result,n):
    
        valid_pred = bst.predict(self.valid[self.feature_cols])
        valid_score = metrics.roc_auc_score(self.valid[self.target], valid_pred)
    
        test_pred = bst.predict(self.test[self.feature_cols])
        test_score = metrics.roc_auc_score(self.test[self.target], test_pred)
    
        print(f"Validation AUC score: {valid_score:.4f}")
        print(f"Test AUC score: {test_score:.4f}")
        if valid_score > 0.95:
            print("\n\nYou're overfitting. Better rerun with parameter 'over' as 'True'. If already set as so\
            it is time to fine tune your model parameters!\n\n")
            
        pl = plot(evals_result,valid_pred,bag["gbm"][n]["model"],self.target,"gbm")
        pl.graphs(n)
        
        