
import utils 
import pandas as pd 
import csv
import tweepy as tw
from collections import Counter





def define_parameters():
	"""
	This functions initialises the parameters to retrieve tweets from the API. 
	Inputs: No input
	Outputs: a list of search terms to be used in the twitter search and
			 the authentication keys to connect to the Twitter API

	"""
	search_terms = ["\" #myschizophreniadiagnosis\"", "\" I am schizophrenic\"", "\" #schizophrenic\"", "\" #myschizophrenia\"", 
	"\" I've been diagnosed with schizophrenia\"", "\" I was diagnosed with schizophrenia\"", "\"I got schizophrenia\"", 
	" \" my psychosis\"", "\"#ihaveschizophrenia\"", "\" I got diagnosed with schizoaffective disorder\"", 
	"\" I got diagnosed as schizophrenic\"", "\" I got diagnosed with schizophrenia\"", "\"my schizophrenia\"", 
	"\"I am schizophrenic\"", "\"I have been diagnosed with schizophrenia\""]

	consumer_key = "gzu5YvfhuFUWFHqDU8E5VQjKW"
	consumer_secret = "UvN5QcPX5808t7IIpefQWwdMo6ubOiXCIWDRGD9yZz6MD32jq5"
	access_token = "938517631828623368-AjQq5UQ1gxbq7EVRTVCZo7iXKHdFEl4"
	access_token_secret = "2JiUdWfRaFEwwbaiHAlgeCmB683ulVHVlfx4Lx8FTqExl"

	return search_terms, consumer_key, consumer_secret, access_token, access_token_secret 




def get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret ):

	"""
	This functions connects the script to the Twitter API and retrieves tweets according to the search_terms.
	Inputs: the list of search_terms outputted by the function define_parameters and the authenitcation keys 
				to connect to the Twitter API.
	Outputs: it writes collection of tweets results in a csv file.
	"""
	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth, wait_on_rate_limit=True)


	with open('control_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		for key_term in tags:
			counter = 0
			for tweet in tw.Cursor(api.search,
				q = key_term,
				count = 200,
				tweet_mode='extended',
				lang = "en").items():				

				# here we are interested in authored content, so we exclude any retweeets from the search.
				if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
				counter += 1
				print("Fetching Tweet N%s " %counter )
				csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.full_text.encode('utf-8')])
					
	csvFile.close()


def get_user_tweets(username_list, consumer_key, consumer_secret, access_token, access_token_secret):

	"""
	This function retrieves the tweets for every username identified in the username_list. 
	The tweets are fetched from the user timelines.
	Input: the list of users and the authorisation keys
	Outputs: the retrieved tweets are saved in a csv file. 

	"""

	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth)



	all_usernames = len(username_list)
	with open('all_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		#make request through the search API for the most recent 200 tweets (200 is the max allowed)

		tweet_counter = 0 #establishes a counter to number tweets output
		user_counter = 0  #established a counter to number the user processed

		
		for username in username_list:
			user_counter += 1
			for tweet in tw.Cursor(api.user_timeline, screen_name= username, tweet_mode="extended",  wait_on_rate_limit=True).items():
				tweet_counter = tweet_counter + 1
				print("---------------------------------------------------------------")
				print("Fetching tweets from user number:", user_counter , "  out of:  ", all_usernames)


				if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.full_text.encode('utf-8')])




if __name__ == '__main__':


	search_terms, consumer_key, consumer_secret, access_token, access_token_secret = define_parameters()
	get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret )

	username_list = complete_tweets('all_tweets.csv', users)
	get_user_tweets(missing_usernames, consumer_key, consumer_secret, access_token, access_token_secret)
	
	
	
	



	