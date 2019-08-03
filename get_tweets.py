

import pandas as pd 
import csv
import tweepy as tw
import twitter
from collections import Counter
from itertools import chain




def define_parameters():
	search_terms = ["\" #myschizophreniadiagnosis\"", "\" I am schizophrenic\"", "\" #schizophrenic\"", "\" #myschizophrenia\"", "\" I've been diagnosed with schizophrenia\"", "\" I was diagnosed with schizophrenia\"", "\"I got schizophrenia\"", " \" my psychosis\"", "\"#ihaveschizophrenia\"", "\" I got diagnosed with schizoaffective disorder\"", "\" I got diagnosed as schizophrenic\"", "\" I got diagnosed with schizophrenia\"", "\"my schizophrenia\"", "\"I am schizophrenic\"", "\"I have been diagnosed with schizophrenia\""]

	consumer_key = ""
	consumer_secret = ""
	access_token = ""
	access_token_secret = ""

	return search_terms, consumer_key, consumer_secret, access_token, access_token_secret 




def get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret ):
	
	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	#api = tw.API(auth, wait_on_rate_limit=True)


	with open('tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		for key_term in search_terms:
			for tweet in tw.Cursor(api.search,
				q = key_term,
				count = 200,
				tweet_mode='extended',
				lang = "en").items():

				if (not tweet.retweeted) and ('RT @' not in tweet.text):
					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.text.encode('utf-8')])
					print ('tweet:', tweet.text)

	csvFile.close()




def extract(filename):

	tweet_dataframe = pd.read_csv(filename)
	print(tweet_dataframe.shape)

	tweet_id = tweet_dataframe.iloc[:,0]
	tweet_username = tweet_dataframe.iloc[:,1]

	tweet_id_list = tweet_id.values.tolist()
	tweet_username_list = tweet_username.values.tolist() 

	return tweet_id_list, tweet_username_list





def get_user_tweets(tweet_username_list, consumer_key, consumer_secret, access_token, access_token_secret):

	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth)



	all_usernames = len(username_list)

	with open('all_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		#make request through the search API for the most recent 200 tweets (200 is the max allowed)

		tweet_counter = 0 #establishes a counter to number tweets output
		user_counter = 0

		for username in tweet_username_list:

			user_counter = user_counter + 1

			for tweet in tw.Cursor(api.user_timeline, screen_name= username, tweet_mode="extended",  wait_on_rate_limit=True).items():

				tweet_counter = tweet_counter + 1

				print("---------------------------------------------------------------")
				#print("Fetching tweets from:", username)
				#print("Tweet Number:", tweet_counter)
				#print("User Number",user_counter,"of", all_usernames)
				#print("---------------------------------------------------------------")

				print("Fetching tweets from user number:", user_counter )
				print("Tweet number", tweet_counter)





				if (not tweet.retweeted) and ('RT @' not in tweet.full_text):

					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.full_text.encode('utf-8')])
				#print ('all_tweets:', tweet.text)



'''
		try:
			new_tweets = api.user_timeline(screen_name = screen_name, count = 200, tweet__mode = 'extended')
		except tw.TweepError:
			print("Failed to run the command on that user, Skipping...")
	
	all_tweets.append(new_tweets)

	
	oldest = all_tweets[-1].id -1

	# keep grabbing tweets until there are no tweets left to grab

	while len(new_tweets) > 0:
		print ("getting tweets before %s" % (oldest))

		# all subsequent requests use the max_id parameter to prevent duplicates
		for screen_name in tweet_username_list:
			try:
				new_tweets = api.user_timeline(screen_name = screen_name, count = 200, max_id = oldest, tweet__mode = 'extended')
			except tw.TweepError:
				print("Failed to run the command on that user, Skipping...")

				# save most recent tweets
		all_tweets.extend(new_tweets)
			# update the id of the oldest tweet minus 1 
		oldest = all_tweets[-1].id -1

		print ("...%s tweets downloaded so far" % (len(all_tweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	

	outtweets = [[tweet.user.screen_name, tweet.id_str, tweet.text.encode("utf-8")] for tweet in all_tweets]
	
	print("outtweets have been stored")

	with open('all_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile) 
		for tweet in all_tweets:
			csvWriter.writerow([tweet.id_str, tweet.user.screen_name, tweet.text.encode('utf-8')])
			#print (tweet.created_at, tweet.text)
	csvFile.close()
'''
'''
def get_user_tweets(tweet_id_list, tweet_username_list, consumer_key, consumer_secret, access_token, access_token_secret):

	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth)

	all_tweets = []

	#make request through the search API for the most recent 200 tweets (200 is the max allowed)
	for screen_name in tweet_username_list:
		new_tweets = api.user_timeline(screen_name = screen_name, count = 200, tweet_mode = 'extended')
		for tweet in new_tweets:
			#print(tweet.user.screen_name,tweet.full_text)
'''

def complete_tweets(filename, username_list):
	df = pd.read_csv(filename)
	done = df.iloc[:,1]

	usernames_done = set(done)

	missing_usernames = list(set(username_list) - set(usernames_done))

	tot_users = len(username_list)
	print(tot_users)

	tot_done = len(usernames_done)
	print(tot_done)

	expected = tot_users - tot_done
	outcome = len(missing_usernames)

	if expected == outcome:
		print("Expected amount of users to fetch:", expected)
		print("Actual outcome of users to fetch: ", outcome)
		print("SUCCESS")

	else:
		print("Expected amount of users to fetch:", expected)
		print("Actual outcome of users to fetch: ", outcome)

		raise ValueError('Something very sketchy is going on here')

	return missing_usernames



def anonymise_dataset(data_frame):

	data_frame['Username'] = 'user' + pd.Series(pd.factorize(data_frame['Username'])[0] + 1).astype(str)
	
	data_frame.to_csv("anonymised.csv", index=False)

	
def make_anonymous(data_frame, username_list):

	total_users = len(username_list)
	print(total_users)

	user_codes = {}

	for name in username_list:
		print(name)
		for number in range(1, total_users):
			user_codes = {name, number}

	print (Counter(chain.from_iterable(i.itervalues() for i in user_codes.itervalues())))



def delete_duplicate_tweets(filename):

	data_frame = pd.read_csv(filename)
	data_frame.columns = ['TweetID', 'Username', 'Text']

	df = data_frame.sort_values(by='Username').drop_duplicates('Text')

	return df 



if __name__ == '__main__':



	search_terms, consumer_key, consumer_secret, access_token, access_token_secret = define_parameters()
	#get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret )

	tweet_id_list, tweet_username_list = extract('final_tweets.csv')
	name_set = set(tweet_username_list)
	username_list = list(name_set)

	#missing_usernames = complete_tweets('all_tweets.csv', username_list)

	
	#get_user_tweets(missing_usernames, consumer_key, consumer_secret, access_token, access_token_secret)


	
	data_frame = delete_duplicate_tweets("all_tweets.csv")
	data_frame_anon = anonymise_dataset(data_frame)



	
