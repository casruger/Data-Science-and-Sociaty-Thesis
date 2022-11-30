import requests
import pandas as pd
import json

def get_fixtures_date(start_date,end_date): #"2021-08-04"
	start_date=str(start_date)
	end_date=str(end_date)
	url = "https://soccer.sportmonks.com/api/v2.0/fixtures/between/"+start_date+"/"+end_date+"/?api_token=IFEuXUPQM8LR3D1EEqIfKuSNs1eF1YgSy6Xqm9aCT1krBXQwvqB6UCsogyL4&leagues=564"

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



##############################################################################################################
##############################################################################################################
lijst_fixtures=get_fixtures_date("2020-07-12","2020-07-22")
print(len(lijst_fixtures))


#this works
with open("la-liga-19-20--5", "w") as fp:
     json.dump(lijst_fixtures, fp)

with open("bundesl-21-22", "r") as fp:
     b = json.load(fp)
print(b)
print(len(b))