#!/usr/bin/env python
# coding: utf-8

# In[916]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[1338]:


data = pd.read_csv('data.csv')


# In[1334]:


data.info()


# In[7]:


sns.heatmap(data.isnull(), cbar=False)


# In[1339]:


remove = ['Transparent Window in Package','Prediction','Difference From Fresh']
data = data.drop(remove, axis=1)


# In[1340]:


missing_values = dict(data.isna().sum().sort_values(ascending=False)/data.shape[0])
missing = data.isna().sum().sort_values(ascending=False)/data.shape[0]


# In[1341]:


missing


# In[1342]:


data.isna().sum().mean()/len(data.index)


# In[1343]:


missing = pd.DataFrame(missing).reset_index()


# In[1344]:


no_nan = list(missing[missing[0]==0]['index'])


# # To fill NAs in Preservative added

# In[1345]:


input_data = data[no_nan + ['Preservative Added']]
input_data['Preservative Added'].replace(to_replace=['N', 'Y'], value=[0, 1],inplace=True)


# In[1346]:


train = input_data[input_data['Preservative Added'].notnull()]
test = input_data[input_data['Preservative Added'].isnull()]


# In[1347]:


from sklearn.model_selection import train_test_split

y = train['Preservative Added']
X = train.drop(['Preservative Added'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[1348]:


print(len(train[train['Preservative Added']==1])*100/len(train.index))
print(len(train[train['Preservative Added']==0])*100/len(train.index))


# In[1349]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')
    param_lr = {'logisticregression__C': [0.01,0.1,1,10,100] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,LogisticRegression(class_weight='balanced')),
                         param_grid=param_lr,scoring="accuracy",cv=KFold(shuffle=True))
    gs_lr.fit(X_train,y_train)
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1350]:


y_pred = gs_lr.best_estimator_.predict(X_test)


# In[1351]:


from sklearn.metrics import accuracy_score
def accuracy(y_true, y_pred):
    return(accuracy_score(y_true, y_pred))
    


# In[1352]:


accuracy(y_test, y_pred)


# In[1353]:


test_x = test.drop(['Preservative Added'],axis=1)
test_x['Preservative Added'] = gs_lr.best_estimator_.predict(test_x)


# In[1354]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# # To impute hexanal

# In[1355]:


hex_data = data['Hexanal (ppm)']
input_data = pd.concat([clean_data,hex_data],axis=1)


# In[1356]:


train = input_data[input_data['Hexanal (ppm)'].notnull()]
test = input_data[input_data['Hexanal (ppm)'].isnull()]


# In[1357]:


from sklearn.model_selection import train_test_split

y = train['Hexanal (ppm)']
X = train.drop(['Hexanal (ppm)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[1358]:


from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1359]:


from sklearn.linear_model import Ridge
est = make_pipeline(preprocess,Ridge(alpha=1.8))
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1360]:


est.score(X_train,y_train)


# In[1361]:


est.score(X_test,y_test)


# In[1362]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, pred)


# In[1363]:


mean_squared_error(y_train, est.predict(X_train))


# In[1364]:


plt.scatter(pred, y_test-pred)


# In[1365]:


test_pred = est.predict(test.drop(['Hexanal (ppm)'],axis=1))


# In[1366]:


test_pred.mean()


# In[1367]:


sns.boxplot(test_pred)


# In[1368]:


test_x = test.drop(['Hexanal (ppm)'],axis=1)
test_x['Hexanal (ppm)'] = test_pred


# In[1369]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# In[1370]:


clean_data.shape


# In[1371]:


data['Hexanal (ppm)'].median()


# In[1372]:


clean_data['Hexanal (ppm)'].median()


# In[1373]:


sns.boxplot(data['Hexanal (ppm)'])
sns.boxplot(clean_data['Hexanal (ppm)'])


# From the above, we see that after imputing with model, we haven't disturbed the original distribution drastically. So we are good.

# # To impute oxygen

# In[1374]:


oxy_data = data['Residual Oxygen (%)']
input_data = pd.concat([clean_data,oxy_data],axis=1)


# In[1375]:


train = input_data[input_data['Residual Oxygen (%)'].notnull()]
test = input_data[input_data['Residual Oxygen (%)'].isnull()]

y = train['Residual Oxygen (%)']
X = train.drop(['Residual Oxygen (%)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1376]:


from sklearn.linear_model import Ridge
est = make_pipeline(preprocess,Ridge(alpha=1))
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1377]:


est.score(X_train,y_train)


# In[1378]:


est.score(X_test,y_test)


# In[1379]:


mean_squared_error(y_test, pred)


# In[1380]:


mean_squared_error(y_train, est.predict(X_train))


# In[1381]:


plt.scatter(pred, y_test-pred)


# In[1382]:


test_pred = est.predict(test.drop(['Residual Oxygen (%)'],axis=1))


# In[1383]:


test_x = test.drop(['Residual Oxygen (%)'],axis=1)
test_x['Residual Oxygen (%)'] = test_pred


# In[1384]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# In[1385]:


clean_data.shape


# # to impute storage conditions

# From the subject, we know that storage conditions would depend on base ingredient, product type and process type. Hence use a decision tree based model to fill these missing values.

# In[1386]:


data['Storage Conditions'].unique()


# In[1387]:


plt.figure(figsize=(10, 6))
ax = sns.countplot(hue='Storage Conditions', x="Product Type", data=data)


# In[1388]:


plt.figure(figsize=(10, 6))
ax = sns.countplot(hue='Storage Conditions', x="Base Ingredient", data=data)


# In[1389]:


view = data[data['Base Ingredient'] == 'C'][['Base Ingredient','Storage Conditions']]
sns.heatmap(view.isnull(), cbar=False)


# In[1390]:


view = data[data['Product Type'] == 'E'][['Product Type','Storage Conditions']]
sns.heatmap(view.isnull(), cbar=False)


# In[1391]:


sc_data = data['Storage Conditions']
input_data = pd.concat([clean_data,sc_data],axis=1)


# In[1392]:


factor = pd.factorize(input_data['Storage Conditions'])


# In[1393]:


input_data['Storage Conditions'] = factor[0]


# In[1394]:


input_data['Storage Conditions'].unique()


# In[1395]:


train = input_data[input_data['Storage Conditions'] != -1]
test = input_data[input_data['Storage Conditions'] == -1]

y = train['Storage Conditions']
X = train.drop(['Storage Conditions'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1396]:


from sklearn.ensemble import RandomForestClassifier
est = make_pipeline(preprocess,RandomForestClassifier())
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1397]:


est.score(X_train,y_train)


# In[1398]:


est.score(X_test,y_test)


# In[1399]:


test_pred = est.predict(test.drop(['Storage Conditions'],axis=1))
test_x = test.drop(['Storage Conditions'],axis=1)
test_x['Storage Conditions'] = test_pred


# In[1400]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# In[1401]:


clean_data.shape


# In[1402]:


clean_data['Storage Conditions'] = clean_data['Storage Conditions'].astype(object)


# # To impute packaging stablizer

# In[1403]:


ps_data = data['Packaging Stabilizer Added']
input_data = pd.concat([clean_data,ps_data],axis=1)


# In[1404]:


train = input_data[input_data['Packaging Stabilizer Added'].notnull()]
test = input_data[input_data['Packaging Stabilizer Added'].isnull()]

y = train['Packaging Stabilizer Added']
X = train.drop(['Packaging Stabilizer Added'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1405]:


from sklearn.ensemble import RandomForestClassifier
est = make_pipeline(preprocess,RandomForestClassifier())
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1406]:


est.score(X_train,y_train)


# In[1407]:


est.score(X_test,y_test)


# In[1408]:


test_pred = est.predict(test.drop(['Packaging Stabilizer Added'],axis=1))
test_x = test.drop(['Packaging Stabilizer Added'],axis=1)
test_x['Packaging Stabilizer Added'] = test_pred

clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# In[1409]:


clean_data.shape


# # To impute moisture %

# In[1410]:


m_data = data['Moisture (%)']
input_data = pd.concat([clean_data,m_data],axis=1)


# In[1411]:


train = input_data[input_data['Moisture (%)'].notnull()]
test = input_data[input_data['Moisture (%)'].isnull()]

y = train['Moisture (%)']
X = train.drop(['Moisture (%)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1412]:


from sklearn.linear_model import Ridge
est = make_pipeline(preprocess,Ridge(alpha=0.01))
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1413]:


est.score(X_train,y_train)


# In[1414]:


est.score(X_test,y_test)


# In[1415]:


mean_squared_error(y_test, pred)


# In[1416]:


mean_squared_error(y_train, est.predict(X_train))


# In[1417]:


plt.scatter(pred, y_test-pred)


# In[1418]:


test_pred = est.predict(test.drop(['Moisture (%)'],axis=1))
test_x = test.drop(['Moisture (%)'],axis=1)
test_x['Moisture (%)'] = test_pred


# In[1419]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)
clean_data.shape


# # To impute base ingredient

# In[1420]:


BI_data = data['Base Ingredient']
input_data = pd.concat([clean_data,BI_data],axis=1)

train = input_data[input_data['Base Ingredient'].notnull()]
test = input_data[input_data['Base Ingredient'].isnull()]

y = train['Base Ingredient']
X = train.drop(['Base Ingredient'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1421]:


from sklearn.ensemble import RandomForestClassifier
est = make_pipeline(preprocess,RandomForestClassifier())
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1422]:


est.score(X_train,y_train)


# In[1423]:


est.score(X_test,y_test)


# In[1424]:


test_pred = est.predict(test.drop(['Base Ingredient'],axis=1))
test_x = test.drop(['Base Ingredient'],axis=1)
test_x['Base Ingredient'] = test_pred


# In[1425]:


clean_data = pd.concat([train,test_x])
clean_data = clean_data.sort_index(ascending=True)


# In[1426]:


clean_data.shape


# In[1427]:


data.shape


# In[1428]:


clean_data.info()


# # Building actual model with clean data

# In[1429]:


data = pd.read_csv('data.csv')

y = list(data['Difference From Fresh'])
y_c = []
fresh = 0
not_fresh = 0
for val in y:
    if val > 20:
        # not fresh
        y_c.append(0)
        not_fresh = not_fresh + 1
    if val <= 20:
        # fresh
        y_c.append(1)
        fresh = fresh + 1


# In[1430]:


print(not_fresh)
print(fresh)


# In[1431]:


X = clean_data


# In[1435]:


clean_data.info()


# # Baseline prediction model - simple imputation

# In[1159]:


from sklearn.impute import SimpleImputer

def prep_train_data(X_train):
    X_train_cont = X_train.select_dtypes(exclude=['object'])
    X_train_cat = X_train.select_dtypes(include=['object'])

    cont_cols = X_train_cont.columns
    cat_cols = X_train_cat.columns

    imp = SimpleImputer(strategy="median").fit(X_train_cont)
    X_train_cont = imp.transform(X_train_cont)
    X_train_cont = pd.DataFrame(X_train_cont,columns=cont_cols)

    imp1 = SimpleImputer(strategy="most_frequent").fit(X_train_cat)
    X_train_cat = imp1.transform(X_train_cat)
    X_train_cat = pd.DataFrame(X_train_cat,columns=cat_cols)

    for col in X_train_cat.columns:
        X_train_cont[col] = X_train_cat[col]

    X_train_final = X_train_cont
    X_train_dummies = pd.get_dummies(X_train_final)
    
    return X_train_dummies, X_train_cat, imp, imp1


# In[ ]:


def prep_test_data(X_test, X_train_cat, imp,imp1):
    X_test_cont = X_test.select_dtypes(exclude=['object'])
    X_test_cat = X_test.select_dtypes(include=['object'])

    cont_cols = X_test_cont.columns
    cat_cols = X_test_cat.columns

    X_test_cont = imp.transform(X_test_cont)
    X_test_cont = pd.DataFrame(X_test_cont,columns=cont_cols)

    X_test_cat = imp1.transform(X_test_cat)
    X_test_cat = pd.DataFrame(X_test_cat,columns=cat_cols)

    for col in X_test_cat.columns:
        X_test_cont[col] = X_test_cat[col]

    X_test_final = X_test_cont
    X_test_dummies = pd.get_dummies(X_test_final)
    Process_Type = X_train_cat['Process Type'].unique()
    Product_Type = X_train_cat['Product Type'].unique()
    Sample_ID = X_train_cat['Sample ID'].unique()
    Storage_Conditions = X_train_cat['Storage Conditions'].unique()
    Packaging_Stabilizer_Added = X_train_cat['Packaging Stabilizer Added'].unique()
    Base_Ingredient = X_train_cat['Base Ingredient'].unique()
    
    X_test_final['Process Type'] = pd.Categorical(X_test_final['Process Type'], categories = Process_Type)
    X_test_final['Product Type'] = pd.Categorical(X_test_final['Product Type'], categories = Product_Type)
    X_test_final['Sample ID'] = pd.Categorical(X_test_final['Sample ID'], categories = Sample_ID)
    X_test_final['Storage Conditions'] = pd.Categorical(X_test_final['Storage Conditions'], categories = Storage_Conditions)
    X_test_final['Packaging Stabilizer Added'] = pd.Categorical(X_test_final['Packaging Stabilizer Added'], categories = Packaging_Stabilizer_Added)
    X_test_final['Base Ingredient'] = pd.Categorical(X_test_final['Base Ingredient'], categories = Base_Ingredient)
    
    X_test_dummies = pd.get_dummies(X_test_final)
    
    return X_test_dummies


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Difference From Fresh','Prediction'],axis=1), 
                                                    y, test_size=0.33, random_state=42)


# In[1146]:


from sklearn.linear_model import LinearRegression
X_train_dummies,X_train_cat,imp, imp1 = prep_train_data(X_train)
X_test_dummies = prep_test_data(X_test,X_train_cat,imp,imp1)
est = LinearRegression().fit(X_train_dummies,y_train)


# In[1134]:


pred = est.predict(X_test_dummies)


# In[1135]:


mean_squared_error(y_test, pred)


# In[1136]:


plt.scatter(pred, y_test-pred)


# # Baseline classification model - Simple Imputation

# In[1164]:


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Difference From Fresh','Prediction'],axis=1), 
                                                    y_c, test_size=0.33, random_state=42)
X_train_dummies,X_train_cat,imp, imp1 = prep_train_data(X_train)
X_test_dummies = prep_test_data(X_test,X_train_cat,imp,imp1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    param_lr = {'logisticregression__C': [1, 5, 7, 10, 25, 50, 100] }
    
    gs_lr = GridSearchCV(estimator=make_pipeline(LogisticRegression()),
                         param_grid=param_lr,scoring="roc_auc",cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train_dummies,y_train)
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1165]:


from sklearn.metrics import f1_score
f1_score(y_test, gs_lr.best_estimator_.predict(X_test_dummies))  


# # Fit linear regression as baseline model for clean data

# In[1315]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[1316]:


X.info()


# In[1440]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1318]:


from sklearn.linear_model import LinearRegression
est = make_pipeline(preprocess,LinearRegression())
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[1319]:


est.score(X_train,y_train)


# In[1320]:


est.score(X_test,y_test)


# In[1321]:


mean_squared_error(y_test, pred)


# In[1322]:


plt.scatter(pred, y_test-pred)


# As can be seen, linear regression did not perform well. However, this is a baseline to start with.

# # Ridge Regression

# In[1025]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_lr = {'ridge__alpha': [0.01,0.1,0.5,1,2,3,10] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,Ridge()),
                         param_grid=param_lr,cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1026]:


est = gs_lr.best_estimator_


# In[1027]:


est.score(X_test,y_test)


# In[1028]:


mean_squared_error(y_test, gs_lr.best_estimator_.predict(X_test))


# Ridge is way better than linear regression

# # Lasso Regression

# In[1029]:


from sklearn.linear_model import Lasso
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_lr = {'lasso__alpha': [0.01,0.1,0.5,1,2,3] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,Lasso()),
                         param_grid=param_lr,cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1030]:


mean_squared_error(y_test, gs_lr.best_estimator_.predict(X_test))


# Lasso is just in par with ridge. Let us look at random forest.

# # Random forest regressor

# In[1329]:


from sklearn.ensemble import RandomForestRegressor
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_lr = {'randomforestregressor__n_estimators': [10,30,50,100,150,300] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,RandomForestRegressor()),
                         param_grid=param_lr,cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1330]:


mean_squared_error(y_test, gs_lr.best_estimator_.predict(X_test))


# In[1331]:


final_pred = gs_lr.best_estimator_.predict(X)
X['final_pred'] = final_pred


# In[1333]:


X.to_csv('res_pred.csv')


# In[1325]:


pred = gs_lr.best_estimator_.predict(X_test)
plt.scatter(pred, y_test-pred)


# In[1326]:


feat_imp = gs_lr.best_estimator_.named_steps['randomforestregressor'].feature_importances_


# In[1327]:


def get_feat_imp(best_est,feat_imp):
    cat_list = best_est.named_steps['columntransformer'].named_transformers_['onehotencoder'].categories_
    flat_list = [item for cat_list in cat_list for item in cat_list]
    cont = X_train.select_dtypes(exclude=['object'])
    tot_col = list(cont.columns)
    tot_col.extend(flat_list)
    df = pd.DataFrame({'cat':tot_col,'feat_imp':feat_imp})
    return df.reindex(df.feat_imp.abs().sort_values(ascending = False).index)

res = get_feat_imp(gs_lr.best_estimator_,feat_imp)
res


# In[1328]:


plt.xticks(rotation=90)
plt.bar(res.iloc[:10,0],res.iloc[:10,1])
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.title("features Vs Importances")


# # XGBoost regressor

# In[1037]:


from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV

params = {'n_estimators': [100,200,300], 'max_depth': [4,5,6,7]}
clf = ensemble.GradientBoostingRegressor()

random_search = RandomizedSearchCV(clf, param_distributions=params,n_iter=5,
                                   n_jobs=4,verbose=3, random_state=42 )

random_search.fit(preprocess.fit_transform(X_train), y_train)


# In[1039]:


preprocess.fit(X_train)
mean_squared_error(y_test, random_search.best_estimator_.predict(preprocess.transform(X_test)))


# In[1045]:


pred = random_search.best_estimator_.predict(preprocess.transform(X_test))
plt.scatter(pred, y_test-pred)


# # SVR

# In[1042]:


from sklearn.svm import SVR
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_lr = {'svr__C': [0.01,0.1,1,10,100,1000] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,SVR()),
                         param_grid=param_lr,cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1043]:


mean_squared_error(y_test, gs_lr.best_estimator_.predict(X_test))


# In[1044]:


pred = gs_lr.best_estimator_.predict(X_test)
plt.scatter(pred, y_test-pred)


# # Try polynomial features

# In[865]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

est = make_pipeline(preprocess,PolynomialFeatures(2),LinearRegression())
est.fit(X_train,y_train)
pred = est.predict(X_test)


# In[866]:


est.score(X_test,y_test)


# In[880]:


mean_squared_error(y_test, pred)


# # Classification approach

# In[1444]:


X_train, X_test, y_train, y_test = train_test_split(clean_data, y_c, test_size=0.33, random_state=42)


# In[1445]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    categorical = X_train.dtypes == object
    
    preprocess = make_column_transformer((StandardScaler(),~categorical),
                                         (OneHotEncoder(handle_unknown='ignore'),categorical),
                                         remainder='passthrough')


# In[1446]:


X_train.shape


# In[1447]:


X.info()


# In[1448]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    param_lr = {'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] }
    
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,LogisticRegression()),
                         param_grid=param_lr,scoring="roc_auc",cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1449]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test))


# This is indeed a good baseline.

# # Random forest classifier

# In[1441]:


from sklearn.ensemble import RandomForestClassifier
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_lr = {'randomforestclassifier__n_estimators': [10,30,50,100,150,300] }
    gs_lr = GridSearchCV(estimator=make_pipeline(preprocess,RandomForestClassifier(class_weight="balanced")),
                         scoring='roc_auc',
                         param_grid=param_lr,cv=KFold(shuffle=True))
    
    gs_lr.fit(X_train,y_train)
    
    print(gs_lr.best_params_)
    print(gs_lr.best_score_)


# In[1442]:


confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test))


# In[1443]:


tn, fp, fn, tp = confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test)).ravel()
print(tn, fp, fn, tp)


# In[1203]:


accuracy_score(y_test, gs_lr.best_estimator_.predict(X_test))


# In[1204]:


f1_score(y_test, gs_lr.best_estimator_.predict(X_test))


# In[1205]:


final_pred = gs_lr.best_estimator_.predict(X)


# In[1207]:


X['pred'] = final_pred
X.to_csv('res_class.csv')


# In[1195]:


feat_imp = gs_lr.best_estimator_.named_steps['randomforestclassifier'].feature_importances_

def get_feat_imp(best_est,feat_imp):
    cat_list = best_est.named_steps['columntransformer'].named_transformers_['onehotencoder'].categories_
    flat_list = [item for cat_list in cat_list for item in cat_list]
    cont = X_train.select_dtypes(exclude=['object'])
    tot_col = list(cont.columns)
    tot_col.extend(flat_list)
    df = pd.DataFrame({'cat':tot_col,'feat_imp':feat_imp})
    return df.reindex(df.feat_imp.abs().sort_values(ascending = False).index)

res = get_feat_imp(gs_lr.best_estimator_,feat_imp)
res


# In[1196]:


plt.xticks(rotation=90)
plt.bar(res.iloc[:5,0],res.iloc[:5,1])
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.title("features Vs Importances")


# # Ada Boost classifier

# In[1192]:


from sklearn.ensemble import AdaBoostClassifier
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    param_ada = {'adaboostclassifier__n_estimators': [3,5,10,30,50,100,150,300] }
    gs_ada = GridSearchCV(estimator=make_pipeline(preprocess,AdaBoostClassifier()),scoring='roc_auc',
                         param_grid=param_ada,cv=KFold(shuffle=True))
    
    gs_ada.fit(X_train,y_train)
    
    print(gs_ada.best_params_)
    print(gs_ada.best_score_)


# In conclusion, Random forest regressor and classifier work well on this data. The AUC with the best model is 0.86 and MSE is 57.8.

# In[ ]:




