from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder   
import pandas as pd
import numpy as np
import re

class prepa:
    
    def __init__(self,dataframe,col=None):
        
        self.df = dataframe
#         if col == None:
#             self.col = df.columns
#         else:
#             self.col = col
    
    def col_nan_errors(self,mode):
        obj_cols = [x for x in self.df.columns[self.df.dtypes.eq(object)]]
        if len(obj_cols)!= 0:
            print("You have these columns as object types (thus possibly with errors) : \n", obj_cols)
        else:
            print("\n\nYou don't have any columns as object type...")
        cols_str = [x for x in obj_cols if type(self.df[x][0]) == str]
        #type(coll)
        
        if len(cols_str)!=0:
            print("\n\nThe columns object that are string (will not be considered to 'coerce' and you be removed from the correction) are : \n", cols_str)
        
        cols_nan = []
        [cols_nan.append(name) for name,val in self.df.isnull().any().items() if val]

        ans = ""
        if len(cols_nan)!=0:
            print("\n\nThe columns that have nan values are : \n", cols_nan)
#             print("\n\nWould you like to preserve any of the other columns?\nIf yes, please enter 'y'. Otherwhise 'n'\n")
#             ans = input()
        else:
            print("\n\nSeems you have data in good shape!! No NaN values!!\n\n")
            
        
        
#         idx1,idx2 = "",""
        col_rem = []
        
#         if ans=='y':
#             print("Please enter the index of the first and last columns (without space!) you want to REMOVE from the correction and press ENTER\nAlternatively, you can\
#             write each one with a comma between")
#             ans2 = input()
#             if ans2.isdigit():
#                 idx1 = ans2[0]
#                 idx2 = ans2[1]
#             else:
#                 col_rem = [x for x in ans2.split(",")]
        

        all_cols = cols_nan
        
        all_cols = pd.Series(all_cols)

        all_cols.drop_duplicates(inplace=True)
        
#         select_cols = []
        if len(col_rem) != 0:
            print("Lenght col_rem != 0")
            select_cols = all_cols.drop([x for x in col_rem])
        else:
            select_cols = cols_nan
            if len(select_cols)!=0:
                print("\nOk! So I selected the NaN columns (", select_cols,") and they were already processed!\n")
              
        self.df.loc[:, select_cols] = self.df.loc[:, select_cols].apply(pd.to_numeric, errors='coerce')

        self.df = self.df.dropna(axis=1, how="all")
        
        if mode == "m_max":
            for each in select_cols: 
                self.df[each] = self.df[each].fillna(self.df[each].max()) 
        else:
            if mode == "m_min":
                for each in select_cols: 
                    self.df[each] = self.df[each].fillna(self.df[each].min())
                    
        print("\n\nNow just a final questions (I like questions): would you like to categorize any column or the object types columns? (y/n)\n")
        if len(obj_cols)!=0:
            print("\nObject columns : ",obj_cols)
        print("\nYour columns : ", self.df.columns)
        ans = input()
        if ans == 'y':
            print("\n\nOk! These are the columns you have : \n\n")
            print(f'{[(idx,x) for idx,x in pd.Series(self.df.columns).items()]}')
            print("\n\nCategorization : \n\n1) Label Encoding\n2) CatBoost\n3) Target Encoding\n")
            col_cat = []
            print("\nType of encoding (enter the number) : ")
            ans = input()
            while ans == '1' or ans == '2' or ans == '3':
                if ans == '1':
                    self.categorize("label", col_cat)
                    print("\nTake a look at what do you have now:")
                    display(self.df.head())
                    print("\nIf you would like to categorize again, please enter the type of encoding (1,2 or 3). Otherwhise, just press any other key!\n\n")
                    ans = input()
                if ans == '2':
                    self.categorize("cat", col_cat)            
                if ans == '3':
                    self.categorize("target", col_cat)
            print("\nMy job here is finished!! Now go full with your processed data!\n\nAhn... if you want to see me again, you will have to call me as you did!")

        else:
            print("\nAll right! No categorization!\n\nMy job here is finished!! Now go full with your processed data!\n\nAhn... if you want to see me again, you will have to call me as you did!")
            
        return self.df



    def categorize(self,cat,cols):
        if cat == "label":
            cols = self.get_cols(self.df.columns)
            print("\n\nLabel Encoding as its full power!!")
            for each in cols:
                le = preprocessing.LabelEncoder()
                LabelEncoder()

                le.fit(self.df[each])
                name = each + "_label"
                ser = le.transform(self.df[each])
                self.df[name] = ser 
            
        else:
            if cat == "cat":
                """
                get the catboost encoding
                """
            else:
                if cat == "target":
                    """
                    get the target encoding
                    """
    
    def get_cols(self,colu):
             
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