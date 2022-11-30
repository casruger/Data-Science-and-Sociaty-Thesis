import requests
import pandas as pd
import json
import soccerdata as sd
elo=sd.ClubElo()

def get_elo(datee, name):
	datee=str(datee)
	gdate=elo.read_by_date(datee)
	try:
		value=gdate.loc[name]['elo']
		return value
	except:
		return 0 


def get_pos_player(id_player_out,id_player_in):
	payload={}
	headers = {}
	player_id=str(id_player_out)
	url = "https://soccer.sportmonks.com/api/v2.0/players/"+player_id+"?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4&include=position"
	response = requests.request("GET", url, headers=headers, data=payload)
	if response.status_code == 200:
		resp_text = response.text
		dicto2=json.loads(resp_text)
		player_pos_out=dicto2['data']['position_id']
		player_id=str(id_player_in)
		url = "https://soccer.sportmonks.com/api/v2.0/players/"+player_id+"?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4&include=position"
		response = requests.request("GET", url, headers=headers, data=payload)
		if response.status_code == 200:
			resp_text = response.text
			dicto2=json.loads(resp_text)
			player_pos_in=dicto2['data']['position_id']
			if type(player_pos_in) == int:
				if type(player_pos_out) == int:
					dif=(player_pos_out-player_pos_in)
					return dif
				return 0
			return 0
		else: 
			return 0
	else: 
		return 0


def get_name_team(id_loc):
	payload={}
	headers = {}
	home=str(id_loc)
	url = "https://soccer.sportmonks.com/api/v2.0/teams/"+home+"?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4"
	response = requests.request("GET", url, headers=headers, data=payload)
	resp_text = response.text
	dicto2=json.loads(resp_text)
	name=dicto2['data']['name']
	return name


def newgametable(listt):
	datapergame={"league":listt[1],
				"season":listt[2],
				"minut":[0],
				"home_team": listt[3],
				"away_team": listt[4],
				"home_elo": [listt[8]],
				"away_elo": [listt[9]],
				"home_goals":[0],
				"away_goals":[0], 
				"home_red_cards":[0],
				"away_red_cards":[0],
				"home_yellow_cards":[0],
				"away_yellow_cards":[0],
				"total_home_goals":[0],
				"total_away_goals":[0],
				"home_substitutions":[0],
				"away_substitutions":[0],
				"home_strategy":[0],
				"away_strategy":[0],
				"home_team_name":[listt[5]],
				"away_team_name":[listt[6]],
				"date":[listt[7]],
				"fixture":listt[0]
	}
	dfpergame=pd.DataFrame(datapergame,index=[listt[0]])
	for i in range(0,101):
		dfpergame2=pd.concat([dfpergame]*100)
	j=0
	for index,row in dfpergame2.iterrows():
		dfpergame2.iloc[j,2]=j
		j+=1
	return(dfpergame2)


#Get game info
def gamedata(dicto):
	listt=[]
	gameid=dicto['data']['id']
	leagueid=dicto['data']['league_id']
	seasonid=dicto['data']['season_id']
	hometeamid=dicto['data']['localteam_id']
	awayteamid=dicto['data']['visitorteam_id']
	hometeamname=get_name_team(dicto['data']['localteam_id'])
	awayteamname=get_name_team(dicto['data']['visitorteam_id'])
	datee=dicto['data']['time']['starting_at']['date']
	home_elo=get_elo(str(datee),str(hometeamname))
	away_elo=get_elo(str(datee),str(awayteamname))
	listt=[gameid,leagueid,seasonid,hometeamid,awayteamid,hometeamname,awayteamname,datee,home_elo,away_elo]
	return listt
	

#get dicto from list urls
lijst_fixtures=[18117759,18117759]


def getdicto(fixture):
	payload={}
	headers = {}
	fixture=str(fixture)
	url = "https://soccer.sportmonks.com/api/v2.0/fixtures/"+fixture+"?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4&include=events.player"
	response = requests.request("GET", url, headers=headers, data=payload)
	resp_text = response.text
	dicto=json.loads(resp_text)
	return dicto

dicto=getdicto(18117759)


#this returns empty table per fixture
newdf=newgametable(gamedata(dicto))



#Get list of events per minut
def get_list_events(dicto):
	list_events=[]
	for i in dicto['data']['events']['data']:
		i['goals-home']=None
		i['goals-away']=None
		if i['result'] is not None:
			if type(i['result']) is not list: 
				if i['result'] is not str(''):
					i['result']=i['result'].split("-")
					i['goals-home']=i['result'][0]
					i['goals-away']=i['result'][1]
		list_events.append([i['minute'], i['team_id'],i['type'], i['player_id'],i['player_name'], i['related_player_id'], i['related_player_name'],i['goals-home'],i['goals-away']])
	return list_events
list_events=get_list_events(dicto)

#run once to add nr of goals, yellow cards etc.
def add_count(list_events,newdf):
	for i in list_events:
		minut=i[0]
		if i[2] == "goal":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,7] += 1
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,8] += 1
		if i[2] == "own-goal":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,7] += 1
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,8] += 1
		if i[2] == "yellowcard":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,11] += 1
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,12] += 1
		if i[2] == "redcard":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,9] += 1
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,10] += 1
		if i[2] == "yellowred":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,9] += 1
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,10] += 1
		if i[2] == "substitution":
			if i[1]==str(newdf.iloc[2,3]): #home
				newdf.iloc[minut:100,15] += 1
				newdf.iloc[minut:100,17] += int(get_pos_player(i[3],i[5]))
			if i[1]==str(newdf.iloc[2,4]): #away
				newdf.iloc[minut:100,16] += 1
				newdf.iloc[minut:100,18] += int(get_pos_player(i[3],i[5]))
	newdf.iloc[:100,13]=newdf.iloc[99,7]
	newdf.iloc[:100,14]=newdf.iloc[99,8]
	return newdf

#Run per fixture
def run_per_fixture(fixture):
	dicto=getdicto(fixture)
	newdf=newgametable(gamedata(dicto))
	list_events=get_list_events(dicto)
	newdf=add_count(list_events,newdf)

	return(newdf)


def get_list_fixture(list_fixtures,total_df):
	j=0
	for i in list_fixtures:
		fixture=str(i)
		newdf=run_per_fixture(fixture)
		total_df=pd.concat([total_df,newdf],ignore_index=True)
		print(j)
		j+=1
	return total_df

list_fixtures=[]
def get_fixtures_date(start_date,end_date): #"2021-08-04"
	start_date=str(start_date)
	end_date=str(end_date)
	url = "https://soccer.sportmonks.com/api/v2.0/fixtures/between/"+start_date+"/"+end_date+"/?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4"

	payload={}
	headers = {}

	response = requests.request("GET", url, headers=headers, data=payload)
	resp_text = response.text
	dicto=json.loads(resp_text)
	list_fixtures=[]
	for i in dicto['data']:
		print(i['id'])
		list_fixtures.append(i['id'])
	return list_fixtures


with open("serie-a-20-21--5", "r") as fp:
     b = json.load(fp)
print(b)


lijst_fixtures=b
lijst_fixtures=lijst_fixtures[]
print(lijst_fixtures)

total_df = pd.read_csv('tot_file_20_21.csv')

total_df2=get_list_fixture(lijst_fixtures,total_df)

total_df2.to_csv('tot_file_20_21.csv', index=False)










