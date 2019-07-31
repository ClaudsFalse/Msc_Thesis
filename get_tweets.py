

import pandas as pd 
import csv
import tweepy as tw


def define_parameters():
	search_terms = ["\" #myschizophreniadiagnosis\"", "\" I am schizophrenic\"", "\" #schizophrenic\"", "\" #myschizophrenia\"", "\" I've been diagnosed with schizophrenia\"", "\" I was diagnosed with paranoid schizophrenia\"", "\" I was diagnosed with schizophrenia\"", "\"I got schizophrenia\"", " \" my psychosis\"", "\"#ihaveschizophrenia\"", "\" I got diagnosed with schizoaffective disorder\"", "\" I got diagnosed as schizophrenic\"", "\" I got diagnosed with schizophrenia\"", "\"my schizophrenia\"", "\"I am schizophrenic\"", "\"I have been diagnosed with schizophrenia\""]

	consumer_key = "gzu5YvfhuFUWFHqDU8E5VQjKW"
	consumer_secret = "UvN5QcPX5808t7IIpefQWwdMo6ubOiXCIWDRGD9yZz6MD32jq5"
	access_token = "938517631828623368-AjQq5UQ1gxbq7EVRTVCZo7iXKHdFEl4"
	access_token_secret = "2JiUdWfRaFEwwbaiHAlgeCmB683ulVHVlfx4Lx8FTqExl"

	return search_terms, consumer_key, consumer_secret, access_token, access_token_secret 




def get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret ):
	
	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth, wait_on_rate_limit=True)

	with open('tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		for key_term in search_terms:
			for tweet in tw.Cursor(api.search,
				q = key_term,
				count = 200,
				lang = "en").items():

				if (not tweet.retweeted) and ('RT @' not in tweet.text):
					csvWriter.writerow([tweet.id, tweet.user.screen_name, tweet.text.encode('utf-8')])
					#print (tweet.created_at, tweet.text)

	csvFile.close()




def extract(filename):

	tweet_dataframe = pd.read_csv(filename)
	print(tweet_dataframe.shape)

	tweet_id = tweet_dataframe.iloc[:,0]
	tweet_username = tweet_dataframe.iloc[:,1]

	tweet_id_list = tweet_id.values.tolist()
	tweet_username_list = tweet_username.values.tolist() 

	return tweet_id_list, tweet_username_list





def get_user_tweets(tweet_id_list, tweet_username_list, consumer_key, consumer_secret, access_token, access_token_secret):

	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tw.API(auth)


	all_tweets = []

	#make request through the search API for the most recent 200 tweets (200 is the max allowed)
	for screen_name in tweet_username_list:
		for tweet_id in tweet_id_list:
			new_tweets = api.user_timeline(screen_name = screen_name, count = 200, since_id = tweet_id)

			# save most recent tweets 
			all_tweets.extend(new_tweets)

			# save the id of the oldest tweet minus 1 

			oldest = all_tweets[-1].id -1

	# keep grabbing tweets until there are no tweets left to grab

	while len(new_tweets) > 0:
		print ("getting tweets before %s" % (oldest))

		# all subsequent requests use the max_id parameter to prevent duplicates
		for screen_name in tweet_username_list:
			for tweet_id in tweet_id_list:
				new_tweets = api.user_timeline(screen_name = screen_name, count = 200, max_id = oldest, since_id = tweet_id)

				# save most recent tweets
				all_tweets.extend(new_tweets)
				# update the id of the oldest tweet minus 1 
				oldest = all_tweets[-1].id -1

				print ("...%s tweets downloaded so far" % (len(all_tweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	

	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in all_tweets]

	with open('all_tweets.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile) 
		for tweet in all_tweets:
			csvWriter.writerow([tweer.id_str, tweet.created_at, tweet.user.screen_name, tweet.text.encode('utf-8')])
			#print (tweet.created_at, tweet.text)
	csvFile.close()




if __name__ == '__main__':
	#pass in the username of the account you want to download
	
	search_terms, consumer_key, consumer_secret, access_token, access_token_secret = define_parameters()
	print(access_token_secret)
	get_all_tweets(search_terms, consumer_key, consumer_secret, access_token, access_token_secret )

	tweet_id_list, tweet_username_list = extract('tweets.csv')

	get_user_tweets(tweet_id_list, tweet_username_list, consumer_key, consumer_secret, access_token, access_token_secret)