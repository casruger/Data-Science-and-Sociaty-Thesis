import pandas as pd
import json
import numpy as np
import sklearn.metrics as metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


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



train=total_df10[0:367472]
test=total_df10[367472:493472]

y_train_home=train['total_home_goals']
y_train_away=train['total_away_goals']
#y_train_home=train['total_home_goals']
print(y_train_home)
y_test_home=test['total_home_goals']
y_test_away=test['total_away_goals']
#y_test_home=test['total_home_goals']
print(y_test_home)


x_train=train[['league','season', 'minut','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_train)
x_test=test[['league','season', 'minut','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_test)


#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#x_inter_train = interaction.fit_transform(x_train)
#x_inter_test = interaction.fit_transform(x_test)
print(x_train)
print(x_test)

regr_home = MLPRegressor(hidden_layer_sizes=(32,16),activation="relu" ,random_state=1, max_iter=200).fit(x_train, y_train_home)
predictions_home=regr_home.predict(x_test)
#predictions_home=np.expm1(predictions_home)

regr_away = MLPRegressor(hidden_layer_sizes=(32,16),activation="relu" ,random_state=1, max_iter=200).fit(x_train, y_train_away)
predictions_away=regr_away.predict(x_test)
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

#Gridsearch
regr = MLPRegressor()
parameters = { 'hidden_layer_sizes': [(16,6),(4,),(6,),(8,),(10,),(12,),(14,),(16,),(18,),(20,),(24,),(8,4),(10,5),(10,4),(6,6),(4,6),(5,10)],
                'max_iter': [200]
}

tscv = TimeSeriesSplit(n_splits=4)

grid_regr = GridSearchCV(estimator=regr, scoring= 'neg_mean_squared_error', param_grid = parameters, cv = tscv, n_jobs=-1)
grid_regr.fit(x_train, y_train_away)   

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





#Make graph of changes coach features
dfnew = x_test.iloc[0:90]
print(type(dfnew))
print(dfnew)

home_strat_3=regr_away.predict(dfnew)
print(predict_home_0)
print(away_strat2)
print(away_strat_3)
print(home_strat2)
print(home_strat_3)

#13 = away strat, 12= home strat, 11= away subs, 10=home subs, 2=minut
for i in range(len(dfnew)):
    print(dfnew.iloc[i][3])
    dfnew.iat[i,3]=0 #elo_df
    dfnew.iat[i,4]=0
    dfnew.iat[i,5]=0
    dfnew.iat[i,6]=0
    dfnew.iat[i,7]=0
    dfnew.iat[i,8]=0
    dfnew.iat[i,9]=0
    dfnew.iat[i,10]=0
    dfnew.iat[i,11]=0
    dfnew.iat[i,12]=-3
    dfnew.iat[i,13]=0

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
plt.ylabel('Predicted final number of away goals')
plt.title('Predicted final number of away goals for different strategy changes')
plt.legend()
plt.show()