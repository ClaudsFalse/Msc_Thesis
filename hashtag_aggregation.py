

import pandas as pd 
import csv
import tweepy as tw
import twitter
from collections import Counter
from itertools import chain




def define_parameters():
	search_terms = ["\" #myschizophreniadiagnosis\"", "\" I am schizophrenic\"", "\" #schizophrenic\"", "\" #myschizophrenia\"", "\" I've been diagnosed with schizophrenia\"", "\" I was diagnosed with schizophrenia\"", "\"I got schizophrenia\"", " \" my psychosis\"", "\"#ihaveschizophrenia\"", "\" I got diagnosed with schizoaffective disorder\"", "\" I got diagnosed as schizophrenic\"", "\" I got diagnosed with schizophrenia\"", "\"my schizophrenia\"", "\"I am schizophrenic\"", "\"I have been diagnosed with schizophrenia\""]

	consumer_key = "gzu5YvfhuFUWFHqDU8E5VQjKW"
	consumer_secret = "UvN5QcPX5808t7IIpefQWwdMo6ubOiXCIWDRGD9yZz6MD32jq5"
	access_token = "938517631828623368-AjQq5UQ1gxbq7EVRTVCZo7iXKHdFEl4"
	access_token_secret = "2JiUdWfRaFEwwbaiHAlgeCmB683ulVHVlfx4Lx8FTqExl"

	return search_terms, consumer_key, consumer_secret, access_token, access_token_secret 




def get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret ):
	
	tags = ["\" #life\"", "\" #friday\"", "\" #summer\"" ]
	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth, wait_on_rate_limit=True)


	with open('control_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		for key_term in tags:
			for tweet in tw.Cursor(api.search,
				q = "key_term",
				count = 200,
				tweet_mode='extended',
				lang = "en").items():

				if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.full_text.encode('utf-8')])
					#print ('tweet:', tweet.text)

	csvFile.close()




def extract(filename):

	tweet_dataframe = pd.read_csv(filename)
	print(tweet_dataframe.shape)

	tweet_id = tweet_dataframe.iloc[:,0]
	tweet_username = tweet_dataframe.iloc[:,1]

	tweet_id_list = tweet_id.values.tolist()
	tweet_username_list = tweet_username.values.tolist() 

	username_list = [username for username in tweet_username_list if str(username) != 'nan']

	return set(username_list)




def get_user_tweets(username_list, consumer_key, consumer_secret, access_token, access_token_secret):

	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth)



	all_usernames = len(username_list)

	with open('all_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		#make request through the search API for the most recent 200 tweets (200 is the max allowed)

		tweet_counter = 0 #establishes a counter to number tweets output
		user_counter = 0

		
		for username in username_list:

			user_counter = user_counter + 1

			for tweet in tw.Cursor(api.user_timeline, screen_name= username, tweet_mode="extended",  wait_on_rate_limit=True).items():

				tweet_counter = tweet_counter + 1

				print("---------------------------------------------------------------")
				#print("Fetching tweets from:", username)
				#print("Tweet Number:", tweet_counter)
				#print("User Number",user_counter,"of", all_usernames)
				#print("---------------------------------------------------------------")

				print("Fetching tweets from user number:", user_counter , "  out of:  ", all_usernames)
				print("Tweet number", tweet_counter)

		



				if (not tweet.retweeted) and ('RT @' not in tweet.full_text):

					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.full_text.encode('utf-8')])
				#print ('all_tweets:', tweet.text)



def complete_tweets(filename, username_list):
	df = pd.read_csv(filename)
	done = df.iloc[:,1]

	usernames_done = set(done)
	print("done  ", len(usernames_done))


	missing_usernames = list(set(username_list) - set(usernames_done))
	print("missin  ", len(missing_usernames))
	

	tot_users = len(username_list)
	print(tot_users)

	tot_done = len(usernames_done)
	print(tot_done)

	expected = tot_users - tot_done
	outcome = len(missing_usernames)

#	if expected == outcome:
#		print("Expected amount of users to fetch:", expected)
#		print("Actual outcome of users to fetch: ", outcome)
#		print("SUCCESS")
#
#	else:
#		print("Expected amount of users to fetch:", expected)
#		print("Actual outcome of users to fetch: ", outcome)
#
#		raise ValueError('Something very sketchy is going on here')

	return missing_usernames



def anonymise_dataset(data_frame):

	data_frame['Username'] = 'user' + pd.Series(pd.factorize(data_frame['Username'])[0] + 1).astype(str)
	
	data_frame.to_csv("anonymised.csv", index=False)

	


def delete_duplicate_tweets(filename):

	data_frame = pd.read_csv(filename)
	data_frame.columns = ['TweetID', 'Username', 'Text']

	df = data_frame.sort_values(by='Username').drop_duplicates('Text')

	return df 



if __name__ == '__main__':


	search_terms, consumer_key, consumer_secret, access_token, access_token_secret = define_parameters()
	get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret )

	username_list = extract('tweets.csv')
	
	more_usernames = extract('more_tweets.csv')
	print(len(username_list))

	usernames = username_list.union(more_usernames)
	print(len(usernames))

	users = list(usernames)
	
	new_names= ['RayneAdrianax', 'Da3dalusStephen', 'Lucina68332065', 'hizspookygirl', 'kawaiilovesarah', 'RealSonchild196', 'irishnationali1'
	, 'tompostable', 'JasonPoolej2', 'ProvokingDrama', 'smackletweets', 'RaiWaddingham', 'IndigoDaya', 'iamtiaraye', 'uniqueblogme']
	
	for name in new_names:
		users.append(name)

	#username = [usernames[i] for i in range(1,198)]
	
	missing_usernames = complete_tweets('all_tweets.csv', users)

	
	get_user_tweets(missing_usernames, consumer_key, consumer_secret, access_token, access_token_secret)


	
	#data_frame = delete_duplicate_tweets("all_tweets.csv")
	#data_frame_anon = anonymise_dataset(data_frame)



	