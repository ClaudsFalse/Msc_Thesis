

import tweepy
from tweepy import OAuthHandler
import json 
from nltk.tokenize import word_tokenize


#----------------------------------------------------------------------------------

# HELPER FUNCTIONS 

def process_or_store(tweet):
	print(json.dumps(tweet))


#-------------------------------------------------------------------------------------

consumer_key = 'insert API KEY'
consumer_secret = ''
access_token = ''
access_secret = ''

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

for status in tweepy.Cursor(api.home_timeline).items(10):
	# process a single status 
	print(status.text)



for tweet in tweepy.Cursor(api.user_timeline).items():
	process_or_store(tweet._json)


###-------------------------------------------------------------------------
# TEXT PROCESSING 

with open ('tweets.json', 'r') as f:
	line = f.readline() # read only the first tweet / line 
	tweet = json.loads(line) # load it as Python dict 
	print(json.dumps(tweets, indent =4))

### TOKENIZE 

tweet_sample = 'RT @HELLO: this is an example #NLP '
print(word_tokenize(tweet_sample))



