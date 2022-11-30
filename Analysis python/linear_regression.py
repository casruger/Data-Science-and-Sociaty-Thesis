import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import seaborn as sns


total_df10 = pd.read_csv("18till2022_90min.csv")

print(total_df10)

#make interactions
total_df10["minut:home_goals"]=total_df10["minut"]*total_df10["home_goals"]
total_df10["minut:away_goals"]=total_df10["minut"]*total_df10["away_goals"]
total_df10["minut:home_red_cards"]=total_df10["minut"]*total_df10["home_red_cards"]
total_df10["minut:away_red_cards"]=total_df10["minut"]*total_df10["away_red_cards"]
total_df10["minut:home_yellow_cards"]=total_df10["minut"]*total_df10["home_yellow_cards"]
total_df10["minut:away_yellow_cards"]=total_df10["minut"]*total_df10["away_yellow_cards"]
total_df10["minut:home_substitutions"]=total_df10["minut"]*total_df10["home_substitutions"]
total_df10["minut:away_substitutions"]=total_df10["minut"]*total_df10["away_substitutions"]
total_df10["minut:home_strategy"]=total_df10["minut"]*total_df10["home_strategy"]
total_df10["minut:away_strategy"]=total_df10["minut"]*total_df10["away_strategy"]
total_df10["dif_elo"]=total_df10["home_elo"]-total_df10["away_elo"]
total_df10["league"]=total_df10["league"].astype("category")
total_df10["season"]=total_df10["season"].astype("category")
total_df10["log_home"]=np.log1p(total_df10["total_home_goals"])
total_df10["log_away"]=np.log1p(total_df10["total_away_goals"])
total_df10["log_home_goals"]=np.log1p(total_df10["home_goals"])
total_df10["log_away_goals"]=np.log1p(total_df10["away_goals"])

train=total_df10[0:367472]
test=total_df10[367472:493472]

y_train_home=train['total_home_goals']
y_train_away=train['total_away_goals']
print(y_train_home)
y_test_home=test['total_home_goals']
y_test_away=test['total_away_goals']
print(y_test_home)


x_train=train[['league','season', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_train)
x_test=test[['league','season', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_test)


#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#x_inter_train = interaction.fit_transform(x_train)
#x_inter_test = interaction.fit_transform(x_test)
print(x_train)
print(x_test)

#create the linear regression
reg_home=LinearRegression()
reg_home.fit(x_train,y_train_home)
predictions_home=reg_home.predict(x_test)
#predictions_home=np.expm1(predictions_home)

reg_away=LinearRegression()
reg_away.fit(x_train,y_train_away)
predictions_away=reg_away.predict(x_test)
#predictions_away=np.expm1(predictions_away)

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
#print results
regression_results(y_test_home,predictions_home)
regression_results(y_test_away,predictions_away)


#find significant variables
df3 = pd.get_dummies(total_df10,
                     columns = ['league', 'season'])
  
for col in df3.columns:
    print(col)

X = df3[['league_8', 'league_82', 'league_384', 'season_12962', 'season_13005', 'season_13133', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]

y = df3['total_home_goals']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

total_df10.corr()['total_home_goals']

#Assumption tests per minut

y_test_away_np=np.array(y_test_away)
y_test_home_np=np.array(y_test_home)
y_train_home_np=np.array(y_train_home)

print(len(train))
#Create predictions per minute
mintest=89
pred_min=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    pred_min.append(predictions_away[m])

actual_min=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    actual_min.append(y_test_away_np[m])

pred_min_home=[]
for i in range(4083):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    pred_min_home.append(predictions_home[m])

actual_min_home=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    actual_min_home.append(y_train_home_np[m])


residuals=[]
for i in range(len(actual_min_home)):
    residuals.append(actual_min_home[i]-pred_min_home[i])
print(residuals)

#Linearity 
#print(predictions_home)
#residuals = y_test_home.values - predictions_home
from statistics import mean
from statistics import stdev
meann = mean(residuals)
std = stdev(residuals)

stdresiduals = (residuals - meann)/std
print(stdresiduals)

print(len(residuals))
print(len(actual_min_home))
p = sns.scatterplot(x=actual_min_home,y=stdresiduals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Standerdized residuals')
plt.ylim(-6,10)
plt.xlim(-1,14)
p = plt.title('Standerdized residuals vs fitted values for 90 minute predictions')
plt.show()

#Normality of residuals
from scipy.stats import shapiro
p = shapiro (residuals)
print(p)

#https://www.kaggle.com/code/shrutimechlearn/step-by-step-assumptions-linear-regression


#multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
  
X = total_df10[['league','minut','season','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)


#homoscedasticity

y=total_df10['total_home_goals']
x=X
x=sm.add_constant(x)

model = sm.OLS(y, x).fit()
names = ['Lagrange multiplier statistic', 'p-value',
         'f-value', 'f p-value']

test_result = sms.het_breuschpagan(model.resid, model.model.exog)
 
lzip(names, test_result)




### Linearity 

#Normality of residuals
p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')
plt.show()

#auto correlation 
plt.figure()
p = sns.lineplot(x=actual_min_home,y=residuals,marker="o",  markersize='4',color='red',linewidth='2')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-1,5)
plt.xlim(0,10)
p = sns.lineplot([0,10],[0,0],color='blue',linewidth="5")
p = plt.title('Residuals vs fitted values plot for autocorrelation check')
plt.show()




#Correlation matrix
data = total_df10[['total_home_goals','league','season', 'minut', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]

df = pd.DataFrame(data)

corr_matrix = df.corr()
print(corr_matrix)

corr_matrix.to_csv("correlation_matrix.csv")


#autocorrelation of residuals
plt.figure()
p = sns.lineplot(x=predictions_home,y=residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')
plt.show()

#Check linear relationship
mean = residuals.mean()
std = residuals.std()

stdresiduals = (residuals - mean)/std
print(stdresiduals)

plt.figure()
p = sns.lineplot(x=predictions_home,y=stdresiduals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Standerdized residuals')
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Standardized residuals vs predicted values, to check linearity')
plt.show()







#Get predictions per minut

y_test_away_np=np.array(y_test_away)
y_test_home_np=np.array(y_test_home)

#Create predictions per minute
mintest=89
pred_min=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    pred_min.append(predictions_away[m])

actual_min=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    actual_min.append(y_test_away_np[m])

pred_min_home=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    pred_min_home.append(predictions_home[m])

actual_min_home=[]
for i in range(1400):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    actual_min_home.append(y_test_home_np[m])



mse_away=metrics.mean_squared_error(actual_min, pred_min)
r2_away=metrics.r2_score(actual_min, pred_min)

mse=metrics.mean_squared_error(actual_min_home, pred_min_home)
r2=metrics.r2_score(actual_min_home, pred_min_home)

print("away")
print("top",round(mse_away,4))
print("bot",round(r2_away,4))


print("home")
print("top", round(mse,4))
print("bot", round(r2,4))



