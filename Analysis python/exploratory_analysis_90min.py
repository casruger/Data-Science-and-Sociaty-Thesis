import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load in the data
df = pd.read_csv("18till2022_90min.csv")

print(len(df))


train=df[0:394742]
test=df[394742:493472]

print(df['minut'][367472])


#make a list of minuts 0 till 100
minuten=[]
for i in range(90):
    minuten.append(i+1)
print(len(minuten))

#get the mean number of goals for the away team per minut
meanlijst_away=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        m=n+j+1          #the minut of the game
        lijstje.append(df['away_goals'][m])          #the column 
    meanlijst_away.append(np.average(lijstje))
del meanlijst_away[0]
print(meanlijst_away)

#get the mean number of goals for the home team per minut
meanlijst_home=[]
lijstje=[]
for j in range(1,91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        m=n+j          #the minut of the game
        lijstje.append(df['home_goals'][m])          #the column 
    meanlijst_home.append(np.average(lijstje))

print(meanlijst_home)


#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()


ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")

ax.axhline(y=1.1601313149735546, xmin=0.0, xmax=2.0, color='orange',linestyle='dashed')
ax.axhline(y=1.381542950939267, xmin=0.0, xmax=2.0, color='blue',linestyle='dashed')


plt.xlabel('Minute')
plt.ylabel('Average total number of goals')
plt.title('Average total number of goals per minute')
plt.legend()
plt.show()





#yellow cards
meanlijst_away=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(548):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['away_yellow_cards'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
meanlijst_away[0]=0
print(meanlijst_away)

#get the mean number of goals for the home team per minut
meanlijst_home=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['home_yellow_cards'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
meanlijst_home[0]=0
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")

ax.axhline(y=2.198905109489051, xmin=0.0, xmax=2.0, color='orange',linestyle='dashed')
ax.axhline(y=1.9901513769834032, xmin=0.0, xmax=2.0, color='blue',linestyle='dashed')


plt.xlabel('Minute')
plt.ylabel('Average total number of yellow cards')
plt.title('Average total number of yellow cards per minute')
plt.legend()
plt.show()

#red cards
meanlijst_away=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['away_red_cards'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
meanlijst_away[0]=0
print(meanlijst_away)

#get the mean number of goals for the home team per minut
meanlijst_home=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['home_red_cards'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
meanlijst_home[0]=0
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")

plt.xlabel('Minute')
plt.ylabel('Average total number of red cards')
plt.title('Average total number of red cards per minute')
plt.legend()
plt.show()

#substitutions
meanlijst_away=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(548):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['away_substitutions'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
meanlijst_away[0]=0
print(meanlijst_away)

#get the mean number of goals for the home team per minut
meanlijst_home=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['home_substitutions'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
meanlijst_home[0]=0
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")



plt.xlabel('Minute')
plt.ylabel('Average total number of substitutions used')
plt.title('Average total number of substitutions used per minute')
plt.legend()
plt.show()

#get the mean strategy score
meanlijst_away=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['away_strategy'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
meanlijst_away[0]=0
print(meanlijst_away)


#get the mean strategy score
meanlijst_home=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['home_strategy'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
meanlijst_home[0]=0
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")



plt.xlabel('Minute')
plt.ylabel('Average change in strategy')
plt.title('Average change in team strategy per minute')
plt.legend()
plt.show()

#Get mean and standard deviation of feature for given minute
def get_mean_std(name, minut,df ):
    meanlijst_away=[]
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+int(minut)+1          #the minut of the game
        lijstje.append(df[str(name)][n])          #the column 
    meanlijst=np.average(lijstje)
    meanlijst_std=np.std(lijstje)
    print(round(meanlijst,3))
    print(round(meanlijst_std,3))

get_mean_std('away_strategy',90,df)






#show predictions, fits, upper lower R
import pandas as pd

# Load the xlsx file
excel_data = pd.read_excel('Analysis R\excel_file.xlsx')
print(excel_data)



#get the mean number of goals for the away team per minut
upr=[]
lijstje=[]
for j in range(100):
    lijstje=[]
    for i in range(100):
        n=i*100
        n=n+j          #the minut of the game
        lijstje.append(excel_data['fit'][n])          #the column 
    upr.append(np.average(lijstje))
print(upr)

print(fit)

fig, ax = plt.subplots()

ax.plot(minuten,fit, label="fit", color="blue")
ax.plot(minuten,lwr, label= "Away team", color="orange")



plt.xlabel('Minute')
plt.ylabel('Average total number of yellow cards')
plt.title('Average total number of yellow cards per minute')
plt.legend()
plt.show()











predictions_mlp=predictions_home
predictions_linear=predictions_home
predictions_rf=predictions_home

predictions_mlp=predictions_away
predictions_linear=predictions_away
predictions_rf=predictions_away

print(len(predictions_mlp))
print(len(predictions_linear))
print(len(predictions_rf))


predictions_mlp_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j         #the minut of the game
        lijstje.append(predictions_mlp[m])          #the column 
    predictions_mlp_mean.append(np.average(lijstje))

print(predictions_mlp_mean)



predictions_linear_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j         #the minut of the game
        lijstje.append(predictions_linear[m])        #the column 
    predictions_linear_mean.append(np.average(lijstje))

print(len(predictions_linear_mean))
print(len(minuten))


predictions_rf_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j          #the minut of the game
        lijstje.append(predictions_rf[m])          #the column 
    predictions_rf_mean.append(np.average(lijstje))
print(len(predictions_rf_mean))

meanlijst_home=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(5483):
        n=i*90
        n=n+j+1          #the minut of the game
        lijstje.append(df['home_goals'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
meanlijst_home[0]=0
print(meanlijst_home)
home_goals_line=meanlijst_home[1:91]


fig, ax = plt.subplots()

ax.plot(minuten,predictions_mlp_mean, label= "Multilayer perceptron", color="orange")
ax.plot(minuten,predictions_linear_mean, label="Linear regression", color="blue")
ax.plot(minuten,predictions_rf_mean, label="Random forest", color="green")
#ax.plot(minuten,home_goals_line, label="Home goals", color="gray")
ax.axhline(y=1.1601313149735546, xmin=0.0, xmax=2.0, color='blue',linestyle='dashed')

plt.xlabel('Minute')
plt.ylabel('Predicted final number of away goals by all three models')
plt.title('Final number of away goals')
plt.legend()
plt.show()



#Residuals 
residuals_away=[]
for i in range(len(predictions_away)):
    verschil=(predictions_away[i]-y_test_away.iloc[i])
    residuals_away.append(verschil**2)
print(len(residuals_away))


predictions_mlp=residuals
predictions_linear=residuals
predictions_rf=residuals

predictions_mlp_away=residuals_away
predictions_linear_away=residuals_away
predictions_rf_away=residuals_away

print(len(predictions_mlp))
print(len(predictions_linear))
print(len(predictions_rf))


predictions_mlp_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j         #the minut of the game
        lijstje.append(predictions_mlp_away[m])          #the column 
    predictions_mlp_mean.append(np.average(lijstje))

print(len(predictions_mlp_mean))



predictions_linear_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j         #the minut of the game
        lijstje.append(predictions_linear_away[m])        #the column 
    predictions_linear_mean.append(np.average(lijstje))

print(len(predictions_linear_mean))
print(len(minuten))


predictions_rf_mean=[]
lijstje=[]
for j in range(91):
    lijstje=[]
    for i in range(1400):
        n=i*90
        m=n+j          #the minut of the game
        lijstje.append(predictions_rf_away[m])          #the column 
    predictions_rf_mean.append(np.average(lijstje))
print(len(predictions_rf_mean))


fig, ax = plt.subplots()

ax.plot(minuten,predictions_mlp_mean, label= "Multilayer perceptron", color="orange")
ax.plot(minuten,predictions_linear_mean, label="Linear regression", color="blue")
ax.plot(minuten,predictions_rf_mean, label="Random forest", color="green")
#ax.plot(minuten,home_goals_line, label="Home goals", color="gray")

plt.xlabel('Minute')
plt.ylabel('Mean squared error for predicted away goals')
plt.title('Mean squared error per minut for each model')
plt.legend()
plt.show()


#Errors per league
print(len(x_test.loc[367472:393031]))#Bundesliga
print(len(x_test.loc[393032:425971]))#La Liga
print(len(x_test.loc[425972:459361]))#Premier League
print(len(x_test.loc[59362:493474]))#Serie A

def regression_results(y_true, y_pred):
    mse=metrics.mean_squared_error(y_true, y_pred) 

    print('MSE: ', round(mse,4))
        
#print results
regression_results(y_test_home[22410:25560],predictions_home[22410:25560])#Bundes
regression_results(y_test_home[58500:91890],predictions_home[58500:91890])#La Liga
regression_results(y_test_home[113850:117000],predictions_home[113850:117000])#Premier League
regression_results(y_test_home[122850:126000],predictions_home[122850:126000])#Serie A

regression_results(y_test_away[22410:25560],predictions_away[22410:25560])#Bundes
regression_results(y_test_away[58500:91890],predictions_away[58500:91890])#La Liga
regression_results(y_test_away[113850:117000],predictions_away[113850:117000])#Premier League
regression_results(y_test_away[122850:126000],predictions_away[122850:126000])#Serie A
