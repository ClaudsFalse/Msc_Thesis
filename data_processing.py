import pandas as pd 
from  nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, words
from gensim import corpora, models, similarities
from string import punctuation, digits
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant 
import langid
from collections import Counter, defaultdict
import re 
import numpy as np 
from tqdm import tqdm_notebook
import random 
import csv 



# ---------------------------------------HELPER FUNCTIONS ----------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def get_documents(filename):

	data_frame = pd.read_csv(filename)
	data_frame_new = data_frame.iloc[:, 1:3]

	usernames = data_frame_new.iloc[:,0]

	usernames_list = set(usernames.values.tolist())


	documents = data_frame_new.groupby('Username')['Text'].apply(list).to_dict()
	
	return documents, usernames_list




def filter_noneng(lang, documents):
    doclang = [  langid.classify(str(word)) for word in documents ]
    return [documents[k] for k in range(len(documents)) if doclang[k][0] == lang]

#def filter_lang(lang, documents):
 #   doclang = [  langid.classify(doc) for doc in documents.values() ]
  #  return [documents[k] for k in range(len(documents.values())) if doclang[k][0] == lang]

def preprocessing_data(documents, usernames_list):
	"""
	Function that lowercase the text and clean it
	Break the sentences into tokens
	remove punctuations and stopwords
	Standardise the text : can't -> cannot, I'm -> I am
	"""

	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = stopwords.words('english')

	stoplist_extra=['amp','get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via',
            'one','com','new','like','great','make','top','good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice',
            'while','know', 'ngl', 'brb', 'acc', 'smh', 'fwiw', 'ftl', 'lmao', 'lol', 'omg']

    	# identify bigrams and unigrams to strip from tweets 

	counter = 0 
	
	with open('preprocessed.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		for username in documents.keys():

			#print("this is a key", username)
			counter += 1

			
			print('Number of iterations: ' + str(counter) + "  out of:  " + str(len(usernames_list)))
		
			documents[username] = tokenizer.tokenize(str(documents))
			print("Document has been tokenized")

			documents[username] = str(documents).lower() 
			print("Document has been lowercased")
			print('**********************************************************************************')
			#print("CHECK IF THIS BULLSHIT IS WORKING")
				
			#print(documents[username])
		
			print("Document is being stripped of unwanted characters.....")
			documents[username] = [re.sub(r"em", "email ", str(documents[username]))]
			documents[username] = [re.sub(r"jk", "joking ", str(documents[username]))]
			documents[username] = [re.sub(r"til", "learned ", str(documents[username]))]
			documents[username] = [re.sub(r'@[^\s]+','', str(documents[username]))]  # remove usernmaes from tweets (mentions)
			documents[username] = [re.sub(r"fb", "facebook ", str(documents[username]))]   # look for twitter most used acronyms

			print("Document has been stripped")
			# filter out non-english words
			documents[username] = filter_noneng('en', documents[username])

			print("non-english words have been filtered")
			documents[username] = [re.sub(r"(?:\@|http?\://)\S+", "", str(documents[username]))]


			print("creating unigrams and stopwords list.....")
			unigrams = [ word for doc in documents[username] for word in doc if len(word)==1]
			bigrams  = [ word for word in documents[username] for word in doc if len(word)==2]


			print("bigrams--------------", bigrams)
			stoplist  = set(stop_words + stoplist_extra + unigrams + bigrams)

			print("stoplist--------", stoplist)

			# remove stopwords 

			documents[username] = [[token for token in doc if token not in stoplist] for doc in documents[username]]
			print("Stopwords have been removed ")
			# remove numbers only words

			documents[username] = [[token for token in doc if len(token.strip(digits)) == len(token)] for doc in documents[username]]



			# Remove words that only occur once

			token_frequency = defaultdict(int)

			# count all token
			for doc in documents[username]:
				for token in doc:
					token_frequency[token] += 1
					# keep words that occur more than once

			documents[username] = [[token for token in doc if token_frequency[token] > 1] for doc in documents[username]]

			print("USERNAME-----",username,"TWEETS-----", documents[username])
			csvWriter.writerow([username, documents[username]])

	csvFile.close()

	return documents



def tokens_dictionary(documents):

	# Sort words in documents

	for key in documents.keys():
		for doc in documents[key]:
			doc.sort()

			# Build a dictionary where for each document each word has its own id (prerequisite of Gensim LDA modelling)

		tokens_dictionary = corpora.Dictionary(documents[key])
		tokens_dictionary.compactify()

	# and save the dictionary for future use
	tokens_dictionary.save('tokens_dictionary.dict')


	

	print("We now have a dictionary with %s unique tokens" % len(tokens_dictionary))

	return tokens_dictionary





'''


# create a dictionary mapping the tokens to their tf-idf values 

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']


# visualise the tokens with lowest tf-idf scores 

from wordcloud import WordCloud

def plot_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))


#### Not surprisingly, we end up with a list of very generic words. 
#These are very common across many descriptions. tfidf attributes a low score to 
# them as a penalty for not being relevant. Words likes tuesday, friday, day, time, etc...



# Now let's check out the 30 words with the highest tfidf scores.

plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))

# We end up with less common words. These words naturally 
#carry more meaning for the given description and may outline the underlying topic.

# As you've noticed, the documents have more than 7000 features 
#(see the vz shape). put differently, each document has more than 7000 dimensions.


# In order to visualise this crazy multidimensional shit, we reduce the 
# dimension to 50 by singular value decomposition

'''


if __name__ == '__main__':



	documents, usernames_list = get_documents('anonymised.csv')

	

	print("Corpus of %s documents" % len(documents.values()))

	#  Filter non english documents

#	documents = filter_lang('en', documents)

#	print("We have " + str(len(documents)) + " documents in english ")

	preprocessed_documents = preprocessing_data(documents, usernames_list)

	print("TWEET PREPROCESSING COMPLETED")
	print("---------------------------------")
'''
	dic_key = random.choice(list(preprocessed_documents))



	print ("Random key value pair from dictonary is ", dic_key, " - ", preprocessed_documents[dic_key])

	tokens_dictionary = tokens_dictionary(preprocessed_documents)

	print("ID-TOKEN DICTIONARY CREATED")
	print("END OF SCRIPT")
	'''