import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load in the data
df = pd.read_csv("Clean the raw data\season_18_19_19_20_20_21_21_22.csv")

print(len(df))

#make a list of minuts 0 till 100
minuten=[]
for i in range(100):
    minuten.append(i)
print(minuten)

#get the mean number of goals for the away team per minut
meanlijst_away=[]
lijstje=[]
for j in range(100):
    lijstje=[]
    for i in range(5483):
        n=i*100
        n=n+j          #the minut of the game
        lijstje.append(df['away_goals'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
print(meanlijst_away)

#get the mean number of goals for the home team per minut
meanlijst_home=[]
lijstje=[]
for j in range(100):
    lijstje=[]
    for i in range(5483):
        n=i*100
        n=n+j          #the minut of the game
        lijstje.append(df['home_goals'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")

ax.axhline(y=1.1601313149735546, xmin=0.0, xmax=2.0, color='orange')
ax.axhline(y=1.381542950939267, xmin=0.0, xmax=2.0, color='blue')


plt.xlabel('Minute')
plt.ylabel('Average number of home goals')
plt.title('Average total number of goals per minute')
plt.legend()
plt.show()











#yellow cards
meanlijst_away=[]
lijstje=[]
for j in range(100):
    lijstje=[]
    for i in range(5483):
        n=i*100
        n=n+j          #the minut of the game
        lijstje.append(df['away_yellow_cards'][n])          #the column 
    meanlijst_away.append(np.average(lijstje))
print(meanlijst_away)


meanlijst_home=[]
lijstje=[]
for j in range(100):
    lijstje=[]
    for i in range(5483):
        n=i*100
        n=n+j          #the minut of the game
        lijstje.append(df['home_yellow_cards'][n])          #the column 
    meanlijst_home.append(np.average(lijstje))
print(meanlijst_home)

#plot the average total number of goals for the home and away team
fig, ax = plt.subplots()

ax.plot(minuten,meanlijst_home, label="Home team", color="blue")
ax.plot(minuten,meanlijst_away, label= "Away team", color="orange")

ax.axhline(y=2.1701623198978663, xmin=0.0, xmax=2.0, color='orange')
ax.axhline(y=1.9901513769834032, xmin=0.0, xmax=2.0, color='blue')


plt.xlabel('Minute')
plt.ylabel('Average total number of yellow cards')
plt.title('Average total number of yellow cards per minute')
plt.legend()
plt.show()















#show predictions, fits, upper lower R
import pandas as pd

# Load the xlsx file
excel_data = pd.read_excel('Analysis R\excel_file.xlsx')
print(excel_data)


#make a list of minuts 0 till 100
minuten=[]
for i in range(100):
    minuten.append(i)
print(minuten)

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








