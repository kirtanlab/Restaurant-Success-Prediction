
# %%
#Importing Libraries
import numpy as np #NumPy is a general-purpose array-processing package.
import pandas as pd #It contains high-level data structures and manipulation tools designed to make data analysis fast and easy.
import matplotlib.pyplot as plt #It is a Plotting Library
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.
from sklearn.linear_model import LogisticRegression #Logistic Regression is a Machine Learning classification algorithm
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.model_selection import train_test_split #Splitting of Dataset
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import json
from pandas import json_normalize
from sklearn.preprocessing import StandardScaler
# from flask import Flask, jsonify, request,current_app,make_response
# %%
zomato=pd.read_csv("./zomato.csv")

# %%
zomato.head()
zomato.shape

# %%
#deleting Unnnecessary Columns
zomato=zomato.drop(['url','dish_liked','phone'],axis=1) 

# %%
#Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

# %%
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)
zomato.info()
# print(list(zomato['location'].unique()))
# print(list(zomato['rest_type'].unique()))
print(list(zomato['cuisines'].unique()))


# %%
#Changing the Columns Names
zomato.columns
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
zomato.columns

# %%
#Some Transformations
zomato['cost'] = zomato['cost'].astype(str)
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.'))
zomato['cost'] = zomato['cost'].astype(float)
zomato.info()

# %%
#Removing '/5' from Rates
zomato['rate'].unique()
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
zomato['rate'].head()

# %%
# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
zomato.cost.unique()
print(zomato)

# %%
#Encode the input Variables
my_dict={}
def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:
        var_name= f"encoded_{column}"
        zomato[column],my_dict[var_name] = pd.factorize(zomato[column])
    return zomato
zomato_en = Encode(zomato.copy())
# print(my_dict)
value_to_factorize = 'North Indian, Mughlai, Chinese'
idx = my_dict["encoded_cuisines"]
zomato_en_en = pd.concat([zomato, zomato_en], axis=1)
# idx_loc = idx.get_loc(value_to_factorize)
# encoded_value = idx[idx_loc]

# factorized_value = my_dict["encoded_address"].get(value_to_factorize, -1)
# print(factorized_value)



# encoded_zomato = zomato.copy()
    
# columns_to_encode = encoded_zomato.columns[~encoded_zomato.columns.isin(['rate', 'cost', 'votes'])]
    
# # Use pd.get_dummies() to one-hot encode the columns
# encoded_columns = pd.get_dummies(encoded_zomato[columns_to_encode], prefix_sep='_')

# # Replace the original columns with the encoded columns
# encoded_zomato = encoded_zomato.drop(columns=columns_to_encode)
# encoded_zomato = pd.concat([encoded_zomato, encoded_columns], axis=1)
# print(encoded_zomato.head())
# row_num# %%120210102120
#Get Correlation between different variables
corr = zomato_en.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
zomato_en.columns

# %%
weight_for_votes = 0.6
weight_for_rating = 0.4

# Calculate the combined metric using weighted averages
weighted_votes = zomato_en['votes'] * weight_for_votes
weighted_rating = zomato_en['rate'] * weight_for_rating
total_weight = weight_for_votes + weight_for_rating
success_metric = (weighted_votes + weighted_rating) / total_weight
zomato_en['success_metric'] = success_metric
# %%
print(zomato_en['success_metric'])

# %%
# print(zomato_en.head(10))
success_metric = zomato_en['success_metric']
scaler = StandardScaler()
scaler.fit(success_metric.values.reshape(-1, 1))
success_metric = scaler.transform(success_metric.values.reshape(-1, 1))
zomato_en['success_metric'] = success_metric
zomato_en['success_metric'].head()
# %%
#Defining the independent variables and dependent variables
x = zomato_en.iloc[:,[2,3,6,7,8,9]]
y = zomato_en['success_metric']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=353)
x_train.head()
y_train.head()
# print(y_train.iloc[11326])
# %%
# print("y_train",x_test.head())

# %%
print(x_train.head(),y_train.head())

# %%
#Prepare a Linear REgression Model
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred=lin_reg.predict(x_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
lin_r2 = r2_score(y_test,y_pred)
lin_mae = mean_absolute_error(y_test, y_pred)
lin_mse = mean_squared_error(y_test, y_pred)

# %%
#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
DTree_r2 =r2_score(y_test,y_predict)
DTree_mae =mean_absolute_error(y_test, y_predict)
DTree_mse= mean_squared_error(y_test, y_predict)

# %%
#Preparing Random Forest REgression
from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
RForest_r2 =r2_score(y_test,y_predict)
RForest_mae = mean_absolute_error(y_test, y_predict)
RForest_mse = mean_squared_error(y_test, y_predict)

# %%
#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ETree=ExtraTreesRegressor(n_estimators = 500)
ETree.fit(x_train,y_train)
y_predict=ETree.predict(x_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
ETree_r2 = r2_score(y_test,y_predict)
ETree_mae = mean_absolute_error(y_test, y_predict)
ETree_mse =mean_squared_error(y_test, y_predict)

# %%
#Gradient Boosting 
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_predict=gbr.predict(x_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
gbr_r2 = r2_score(y_test,y_predict)
gbr_mae = mean_absolute_error(y_test, y_predict)
gbr_mse = mean_squared_error(y_test, y_predict)

def General(x_test):
    # row_num = np.where(zomato_en_en.values == 'Food Court')[0][0]
    # print(row_num)
    # print(zomato_en.loc[row_num,'rest_type'])

    print("x_test",x_test)
    x_test =pd.DataFrame(x_test,index=[0])
    print("x_df",x_test)
    lin_y = lin_reg.predict(x_test)
    DTree_y = DTree.predict(x_test)
    RForest_y = RForest.predict(x_test)
    ETree_y=ETree.predict(x_test)
    Gbr_y=gbr.predict(x_test)
    out =[
        {
            "modal_name":"Linear Regression","prediction": lin_y[0],"R2Score": lin_r2,"MAE": lin_mae,"MSE": lin_mse

        },
        {
            "modal_name":"Decision Tree","prediction": DTree_y[0],"R2Score": DTree_r2,"MAE": DTree_mae,"MSE": DTree_mse

        },{
            "modal_name":"Random Forest","prediction": RForest_y[0],"R2Score": RForest_r2,"MAE": RForest_mae,"MSE": RForest_mse

        },{
            "modal_name":"Extra Tree","prediction": ETree_y[0],"R2Score": ETree_r2,"MAE": ETree_mae,"MSE": ETree_mse

        },{
            "modal_name":"Gradient Boosting","prediction": Gbr_y[0],"R2Score": gbr_r2,"MAE": gbr_mae,"MSE": gbr_mse

        },
    ]
    print("output",out)
    return jsonify(out)

# %%
