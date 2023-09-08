#!/usr/bin/env python
# coding: utf-8

# # FACTOR ANALYZE

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as plot


# In[6]:


from factor_analyzer import FactorAnalyzer


# In[7]:


df = pd.read_excel("C:/Users/Aakash/Desktop/Main Hp.xlsx")


# In[8]:


df.tail(50)


# In[9]:


df.drop(['Timestamp','Email address'],axis=1,inplace=True)
df.columns


# In[10]:


df.dropna(inplace=True)


# In[11]:


df


# In[13]:


d = df.iloc[ :,:-1]
d


# In[11]:


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
Scaler.fit(df)
scaled = Scaler.transform(df)


# In[12]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)


# In[13]:


df.shape


# In[14]:


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value = calculate_bartlett_sphericity(d)
print("Chi square =",chi_square_value,"P value", p_value)


# In[15]:


from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(d)
print("KMO Value =",kmo_model)


# In[16]:


from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(rotation = None)
fa.fit(d)
ev,v = fa.get_eigenvalues()
ev


# In[ ]:





# In[19]:


from factor_analyzer import FactorAnalyzer
n = d.shape[1]
fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=n)
fa.fit(d)
ev,_ = fa.get_eigenvalues()
plt.scatter(range(1,n+1),ev)

plt.plot(range(1,n+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()


# In[20]:


fa = FactorAnalyzer(n_factors=4,rotation='equamax')
fa.fit(d)


# In[21]:


fa.loadings_


# In[22]:


floading = pd.DataFrame(fa.loadings_,index=d.columns)


# In[23]:


def highlight_max(s):
    # Get 5 largest values of the column
    is_large = s.nlargest(4).values
    # Apply style is the current value is among the 5 biggest values
    return ['background-color: orange' if v in is_large else '' for v in s]

floading.sort_values(by=0,ascending=False).style.apply(highlight_max)


# In[24]:


df.shape


# In[25]:


fscore = fa.transform(d)
fscore_data = pd.DataFrame(fscore)
fscore_data.head(20)


# In[27]:


fscore_data['dependent'] = y


# In[26]:


y = df.iloc[:,16]


# In[28]:


fscore_data


# In[76]:


fscore_data.to_csv(3.csv')


# In[29]:


fscore_data.rename(columns={3: 'Service', 'dependent': 'Cust.Satisf',
                            2: 'Sound & Display',1: 'Hardware',0: 'bulid or outook'}, inplace=True)


# In[30]:


fscore_data.head(20)


# In[31]:


y = fscore_data.iloc[:,4];y
X = fscore_data.iloc[:,0:4];X


# In[ ]:





# In[ ]:





# In[135]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=10)


# In[1]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)


# In[137]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[138]:


predictions = lm.predict(X_test)


# In[139]:


plt.scatter(y_test,predictions,color ='g')
plt.xlabel("prediction")
plt.ylabel("ytest")
plt.show()


# In[140]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)


# In[141]:


print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))


# In[116]:


residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[1]:


p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for linearity check')


# In[118]:


p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')


# In[142]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# the independent variables set
X = fscore_data.iloc[:,0:4];X

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
						for i in range(len(X.columns))]

print(vif_data)


# In[143]:


import warnings
warnings.filterwarnings("ignore")


# In[144]:


from sklearn import metrics


# In[145]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # EDA

# In[146]:


sns.pairplot(fscore_data)


# In[147]:


sns.heatmap(fscore_data.corr(),annot=True)


# In[56]:


sns.displot(fscore_data,kind='hist')


# In[57]:


sns.displot(fscore_data['Cust.Satisf'])


# In[ ]:





# In[58]:


g = pd.read_csv("C:/Users/Aakash/Desktop/All about STATISTICS/Drive/file3.csv")


# In[59]:


g


# In[60]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# the independent variables set


# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = g.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(g.values, i)
						for i in range(len(g.columns))]

print(vif_data)


# In[61]:


g.rename(columns={'A':'bulid_outlook_Hardware','B':'service_DisplayAudioPerformers'})


# In[62]:


g['CustSatis'] = y


# In[ ]:


y  = df.iloc[:,16]


# In[ ]:


g


# In[ ]:


y = g.iloc[:,2]
X = g.iloc[:,0:2]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[61]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)


# In[60]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[1]:


p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')


# In[ ]:





# In[ ]:




