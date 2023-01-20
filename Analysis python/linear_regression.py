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
import sklearn.metrics as metrics

#import the data
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
x_train=train[[ 'league_82','league_384', 'league_564',  '19-20', '20-21', '21-22', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards','home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_train)
x_test=test[[ 'league_82','league_384', 'league_564',  '19-20', '20-21', '21-22', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards','home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
print(x_test)


#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#x_inter_train = interaction.fit_transform(x_train)
#x_inter_test = interaction.fit_transform(x_test)


#create the linear regression for home 
reg_home=LinearRegression()
reg_home.fit(x_train,y_train_home)
predictions_home=reg_home.predict(x_test)
train_predictions_home=reg_home.predict(x_test)
#predictions_home=np.expm1(predictions_home) #Include when calculating the log of goals

#for away
reg_away=LinearRegression()
reg_away.fit(x_train,y_train_away)
predictions_away=reg_away.predict(x_test)
train_predictions_away=reg_away.predict(x_test)
#predictions_away=np.expm1(predictions_away) #Include when calculating the log of goals




# function to calculate the results 
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
X = total_df10[['league_82','league_384', 'league_564',  '19-20', '20-21', '21-22', 'dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
X_no_stat =  total_df10[['league_82','league_384', 'league_564',  '19-20', '20-21', '21-22','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions']] 
X_no_coach = total_df10[['league_82','league_384', 'league_564',  '19-20', '20-21', '21-22','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards']]

y = total_df10['total_home_goals']
y_away = total_df10['total_away_goals']

X2 = sm.add_constant(X_no_stat)
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

from scipy.stats import shapiro
p = shapiro (residuals)
print(p)

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
p = sns.scatterplot(x=actual_min,y=stdresiduals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Standerdized residuals')
plt.ylim(-6,10)
plt.xlim(-1,3)
p = plt.title('Standerdized residuals vs fitted values for 90 minute predictions')
plt.show()

#Normality of residuals
from scipy.stats import shapiro
p = shapiro (residuals)
print(p)

#https://www.kaggle.com/code/shrutimechlearn/step-by-step-assumptions-linear-regression


#multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
  
X = total_df10[['league_82','league_384', 'minut','league_564',  '19-20', '20-21', '21-22','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions', 'home_strategy', 'away_strategy']]
X_no_stat =  total_df10[['league_82','league_384', 'league_564',  '19-20', '20-21', '21-22','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards', 'home_substitutions', 'away_substitutions']] 
X_no_coach = total_df10[['league_82','league_384', 'league_564',  '19-20', '20-21', '21-22','dif_elo', 'home_goals', 'away_goals', 'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)


#homoscedasticity

y=total_df10['total_away_goals']
x=X_no_coach
x=sm.add_constant(x)

model = sm.OLS(y, x).fit()
names = ['Lagrange multiplier statistic', 'p-value',
         'f-value', 'f p-value']

test_result = sms.het_breuschpagan(model.resid, model.model.exog)
 
lzip(names, test_result)


#Get signicance of difference between models
model = sm.OLS(y, X).fit()
model_nostat = sm.OLS(y, X_no_stat).fit()
model_nocoach = sm.OLS(y, X_no_coach).fit()
full_ll = model.llf
nostat_ll=model_nostat.llf
nocoach_ll=model_nocoach.llf

model_a = sm.OLS(y_away, X).fit()
model_nostat_a = sm.OLS(y_away, X_no_stat).fit()
model_nocoach_a = sm.OLS(y_away, X_no_coach).fit()
full_ll_a = model_a.llf
nostat_ll_a=model_nostat_a.llf
nocoach_ll_a=model_nocoach_a.llf

LR_statistic = -2*(nocoach_ll-nostat_ll)

print(LR_statistic)


#calculate p-value of test statistic using 2 degrees of freedom
import scipy.stats 
p_val = scipy.stats.chi2.sf(LR_statistic, 17)

print(p_val)




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



#Get predictions per minut

y_test_away_np=np.array(y_test_away)
y_test_home_np=np.array(y_test_home)

#Create predictions per minute
mintest=0
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

win_home_pred=[]
for i in range(len(predictions_home)):
    score=(predictions_home[i]- predictions_away[i])
    if float(score) >= 0.4:
        win_home_pred.append('home')
    elif float(score) <=-0.4:
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







def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

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


dm_test(actual_list, mod2away, fullmodelaway, h = 1, crit="MSE", power = 2)

fullmodelhomerg=fullmodelhome
fullmodelawayrg=fullmodelaway

dm_test(actual_list, fullmodelhomerg, fullmlphome, h = 1, crit="MSE", power = 2)



#Add LR to hypothesis of MLP and RF
dfnew = x_test.iloc[0:90]
print(type(dfnew))
print(dfnew.iloc[4])

meanline=reg_home.predict(dfnew)
meanline2=reg_away.predict(dfnew)
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
    dfnew.iat[i,6]=0
    dfnew.iat[i,7]=meanlijst_home[i]
    dfnew.iat[i,8]=meanlijst_away[i]
    dfnew.iat[i,9]=meanlijst_home_red[i]
    dfnew.iat[i,10]=meanlijst_away_red[i]
    dfnew.iat[i,11]=meanlijst_home_yellow[i]
    dfnew.iat[i,12]=meanlijst_away_yellow[i]
    dfnew.iat[i,13]=meanlijst_home_subs[i]
    dfnew.iat[i,14]=meanlijst_away_subs[i]/2
    dfnew.iat[i,15]=meanlijst_home_strat[i]
    dfnew.iat[i,16]=meanlijst_away_strat[i]