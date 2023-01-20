import pandas as pd
import json
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

#Import the data
total_df10 = pd.read_csv("18till2022_90min.csv")
print(total_df10)

#make dummies and other feature transformations
total_df10= pd.get_dummies(total_df10,
                     columns = ['league', 'season'])
total_df10['18-19']=total_df10['season_13158']+ total_df10['season_13133'] + total_df10['season_13005'] + total_df10['season_12962']
total_df10['19-20']=total_df10['season_16415']+ total_df10['season_16326'] + total_df10['season_16264'] + total_df10['season_16036']
total_df10['20-21']=total_df10['season_17488']+ total_df10['season_17480'] + total_df10['season_17420'] + total_df10['season_17361']
total_df10['21-22']=total_df10['season_18576']+ total_df10['season_18462'] + total_df10['season_18444'] + total_df10['season_18378']
total_df10["dif_elo"]=total_df10["home_elo"]-total_df10["away_elo"]
total_df10["league"]=total_df10["league"].astype("category")
total_df10["season"]=total_df10["season"].astype("category")
total_df10["log_home"]=np.log1p(total_df10["total_home_goals"])
total_df10["log_away"]=np.log1p(total_df10["total_away_goals"])
total_df10["log_home_goals"]=np.log1p(total_df10["home_goals"])
total_df10["log_away_goals"]=np.log1p(total_df10["away_goals"])

#Train test split
train=total_df10[0:367472]
test=total_df10[367472:493472]

y_train_home=train['total_home_goals']
y_train_away=train['total_away_goals']
print(y_train_home)
y_test_home=test['total_home_goals']
y_test_away=test['total_away_goals']
print(y_test_home)

#league 8 and season 18/19 removed as dummies
x_train=train[[ 'league_82','league_384', 'league_564',  '19-20', '20-21', '21-22', 'minut', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards','home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_train)
x_test=test[[ 'league_82','league_384', 'league_564',  '19-20', '20-21', '21-22', 'minut','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards','home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_test)


#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#x_inter_train = interaction.fit_transform(x_train)
#x_inter_test = interaction.fit_transform(x_test)
print(x_train)
print(x_test)

regr_home = RandomForestRegressor(max_depth= 6, max_features= 12, min_samples_leaf= 10, min_samples_split= 5, n_estimators= 100, random_state=0)
regr_home.fit(x_train, y_train_home)
predictions_home=regr_home.predict(x_test)
train_predictions_home=regr_home.predict(x_train)
#predictions_home=np.expm1(predictions_home)

regr_away = RandomForestRegressor(max_depth= 6, max_features= 12, min_samples_leaf= 10, min_samples_split= 5, n_estimators= 100, random_state=0)
regr_away.fit(x_train, y_train_away)
predictions_away=regr_away.predict(x_test)
train_predictions_away=regr_away.predict(x_train)
#predictions_away=np.expm1(predictions_away)



def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    smape= 1/len(y_true) * np.sum(2 * np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100)

    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print("SMAPE: ",round(smape,4))
#print results
regression_results(y_test_home,predictions_home)
regression_results(y_test_away,predictions_away)

regression_results(y_train_home,train_predictions_home)
regression_results(y_train_away,train_predictions_away)



#grid search
space=dict()
space['max_depth']=[4,6,8]
space['n_estimators']=[100]
space['min_samples_split']=[2,5,8,10]
space['min_samples_leaf']=[2,5,8,10]
space['max_features']=[5,8,10,12]

regr = RandomForestRegressor()
parameters = { 'max_depth': [4,6,8],
                'n_estimators': [100],
                'random_state': [1],
                'min_samples_split' : [2,5,8,10], 
                'min_samples_leaf' : [2,5,8,10],
                'max_features': [5,8,10,12]
}

tscv = TimeSeriesSplit(n_splits=3)

grid_regr = GridSearchCV(estimator=regr, scoring= 'neg_mean_squared_error', param_grid = parameters, cv = tscv, n_jobs=-1)
grid_regr.fit(x_train, y_train_home)   

grid_regr = RandomizedSearchCV(regr, space, scoring= 'neg_mean_squared_error', n_iter=30, cv = tscv, n_jobs=-1)
grid_regr.fit(x_train, y_train_home)   

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_regr.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_regr.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_regr.best_params_)



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



#Make graph chaning coach feature

dfnew = x_test.iloc[0:90]
print(type(dfnew))
print(dfnew)

meanline=regr_home.predict(dfnew)
meanline2=regr_away.predict(dfnew)
print(meanlijst_home_subs[89]/2)
print(meanlijst_away_subs[89]/2)
print(meanline)
print(away_strat2)
print(away_subs_5)
print(home_strat2)
print(home_strat_3)

for itemss in [44,59,69,79,89]:
    print(itemss,"---")
    print(round(float(meanline[itemss]),3))
    print(round(float(meanline2[itemss]),3))

#17 = away strat, 16= home strat, 15= away subs, 14=home subs, 2=minut
for i in range(len(dfnew)):
    dfnew.iat[i,1]=0
    dfnew.iat[i,2]=0
    dfnew.iat[i,3]=0 #elo_df
    dfnew.iat[i,4]=0
    dfnew.iat[i,5]=0
    dfnew.iat[i,7]=0
    dfnew.iat[i,8]=meanlijst_home[i]
    dfnew.iat[i,9]=meanlijst_away[i]
    dfnew.iat[i,10]=meanlijst_home_red[i]
    dfnew.iat[i,11]=meanlijst_away_red[i]
    dfnew.iat[i,12]=meanlijst_home_yellow[i]
    dfnew.iat[i,13]=meanlijst_away_yellow[i]
    dfnew.iat[i,14]=meanlijst_home_subs[i]*1.5
    dfnew.iat[i,15]=meanlijst_away_subs[i]*1.5
    dfnew.iat[i,16]=meanlijst_home_strat[i]
    dfnew.iat[i,17]=meanlijst_away_strat[i]

minuten=[]
for i in range(90):
    minuten.append(i+1)
print(len(minuten))

fig, ax = plt.subplots()

ax.plot(minuten,predict_home_0, label= "0", color="orange")
ax.plot(minuten,away_strat2, label="Away plus 3", color="blue")
ax.plot(minuten,away_strat_3, label="Away minus 3", color="green")
ax.plot(minuten,home_strat2, label="Home plus 3", color="darkblue")
ax.plot(minuten,home_strat_3, label="Home minus 3", color="darkgreen")
#ax.plot(minuten,home_goals_line, label="Home goals", color="gray")

plt.xlabel('Minute')
plt.ylabel('Predicted final number of home goals')
plt.title('Predicted final number of home goals for different strategy changes')
plt.legend()
plt.show()



win_home_pred=[]
for i in range(len(predictions_home)):
    score=(predictions_home[i]- predictions_away[i])
    if float(score) >= 0.4:
        win_home_pred.append('home')
    elif float(score) <=-0.3:
        win_home_pred.append('away')
    else:
        win_home_pred.append('tie')
print(len(win_home_pred))

win_home_test=[]
for i in range(len(y_test_home)):
    score=y_test_home.iloc[i] - y_test_away.iloc[i]
    if float(score) >= 1:
        win_home_test.append('home')
    elif float(score) <=-1:
        win_home_test.append('away')
    else:
        win_home_test.append('tie')
print(len(win_home_test))

from sklearn.metrics import accuracy_score
accuracy_score(win_home_test, win_home_pred)

#0.6356 Full model overall 10,
# second model nu subs 
#0.6289 basic model overall


mintest=45
pred_min=[]
test_min=[]
for i in range(1400):
    n=i*90
    m=n+mintest 
    pred_min.append(win_home_pred[m])
    test_min.append(win_home_test[m])
print(len(test_min))

accuracy_score(test_min, pred_min)

#0.6050 at 45 min Full model 
#0.7321 at 70 min Full model 
#0.9942 at 90 min Full model 

# at 45 min second model 
# at 70 min second model
# at 90 min second model

# at 45 min basic model
# at 70 min basic model
# at 90 min basic model


#Find if change is significantly big
predictions_home_nosubs=predictions_home
predictions_away_nosubs=predictions_away

predictions_home_full=predictions_home
predictions_away_full=predictions_away

import scipy.stats as stats

 
# conduct the Wilcoxon-Signed Rank Test
stats.wilcoxon(predictions_home_full, predictions_home_nosubs)





#Get scores per minute
y_test_away_np=np.array(y_test_away)
y_test_home_np=np.array(y_test_home)
y_train_home_np=np.array(y_train_home)
nimuntesns=[0,24,49,74,89]
for option in nimuntesns:
    mintest=option
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
    r2_noad_away=metrics.explained_variance_score(actual_min, pred_min)

    mse=metrics.mean_squared_error(actual_min_home, pred_min_home)
    r2=metrics.r2_score(actual_min_home, pred_min_home)
    r2_noad=metrics.explained_variance_score(actual_min_home, pred_min_home)

    print("MSE", option, "home - away")
    print(round(mse,4), round(mse_away,4))
    
    print("r2 adjusted", option, "home - away")
    print(round(r2,4),round(r2_away,4))

    print("r2", option, "home - away")
    print(round(r2_noad,4),round(r2_noad_away,4))

    pred_min=[]
    test_min=[]

    for i in range(1400):
        n=i*90
        m=n+mintest 
        pred_min.append(win_home_pred[m])
        test_min.append(win_home_test[m])
    
    print("accuracy", option)
    print(round(accuracy_score(test_min, pred_min),4))



fullmodelhome=predict1
fullmodelaway=predictions2
mod2home=predict1
mod2away=predictions2
mod1home=predict1
mod1away=predictions2

actual_list = y_test_home.tolist()
actual_list = y_test_away.tolist()
predict1=predictions_home.tolist()
predictions2=predictions_away.tolist()


for i in range(len(predictions2)):
   predictions2[i]= int(predictions2[i])

for i in range(len(predict1)):
   predict1[i]= int(predict1[i])


dm_test(actual_list, mod2home, fullmodelhome, h = 1, crit="MSE", power = 2)

fullrfhome=fullmodelhome
fullrfaway=fullmodelaway