import pandas as pd
import json
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
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

print(len(total_df10))


#Divide data in only 25minut rows
mintest=26
new_df_30=total_df10.loc[26:26]
for i in range(1,5483):
    n=i*90
    m=n+mintest #0 invullen is minut 1
    new_df_30=pd.concat([pd.DataFrame(new_df_30),total_df10.loc[m:m]])
print(new_df_30)

print(len(new_df_30))

train=new_df_30[0:4083]
test=new_df_30[4083:5483]

y_train_home=train['total_home_goals']
y_train_away=train['total_home_goals']
print(y_train_home)
y_test_home=test['total_home_goals']
y_test_away=test['total_away_goals']
print(y_test_home)


x_train=train[['league','season', 'minut','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_strategy', 'away_strategy','home_substitutions', 'away_substitutions']]
print(x_train)
x_test=test[['league','season', 'minut','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_strategy', 'away_strategy','home_substitutions', 'away_substitutions']]
print(x_test)


#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#x_inter_train = interaction.fit_transform(x_train)
#x_inter_test = interaction.fit_transform(x_test)
print(x_train)
print(x_test)

regr_home = RandomForestRegressor(max_depth= 10, max_features= 5, min_samples_leaf= 16, min_samples_split= 2, n_estimators= 200, random_state=0)
regr_home.fit(x_train, y_train_home)
predictions_home=regr_home.predict(x_test)
#predictions_home=np.expm1(predictions_home)

regr_away = RandomForestRegressor(max_depth= 10, max_features= 5, min_samples_leaf= 16, min_samples_split= 2, n_estimators= 200, random_state=0)
regr_away.fit(x_train, y_train_away)
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

regr.feature_importances_


#grid search
regr = RandomForestRegressor()
parameters = { 'max_depth': [1,2,10,12,15],
                'n_estimators': [100,200],
                'random_state': [1],
                'min_samples_split' : [1,2,5,10,16], 
                'min_samples_leaf' : [1,2,5,10,16],
                'max_features': [1,2,5,10,16]
}

tscv = TimeSeriesSplit(n_splits=3)

grid_regr = GridSearchCV(estimator=regr, scoring= 'neg_mean_squared_error', param_grid = parameters, cv = tscv, n_jobs=-1)
grid_regr.fit(x_train, y_train_home)   

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_regr.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_regr.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_regr.best_params_)



