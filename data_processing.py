# Author: Claudia Falsetti <1431314>
# Python 3.7.3 

import pandas as pd 
from nltk.corpus import stopwords, words, wordnet 
from gensim import corpora, models, similarities
from string import punctuation, digits
from collections import Counter, defaultdict
import re 
import numpy as np 
import csv 
import string 
from nltk.stem import WordNetLemmatizer 
import nltk 
import pickle


# ---------------------------------------HELPER FUNCTIONS ----------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def get_documents(filename):

	data_frame = pd.read_csv(filename, names=["TweetID","Username", "Text"])
	data_frame_new = data_frame.iloc[:,1:3]
	#print(data_frame_new.head(5))

	usernames = data_frame_new.iloc[:,0]

	usernames_list = set(usernames.values.tolist())


	documents = data_frame_new.groupby('Username')['Text'].apply(list).to_dict()
	
	return documents, usernames_list

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


#def filter_lang(lang, documents):
 #   doclang = [  langid.classify(doc) for doc in documents.values() ]
  #  return [documents[k] for k in range(len(documents.values())) if doclang[k][0] == lang]

def preprocessing_data(documents):
	"""
	Function that lowercase the text and clean it
	Break the sentences into tokens
	remove punctuations and stopwords
	Standardise the text : can't -> cannot, I'm -> I am
	"""

	#tokenizer = RegexpTokenizer(r'\w+')

	token_frequency = defaultdict(int)
	lemmatizer = WordNetLemmatizer()
	lemmas = set(wordnet.all_lemma_names())

	stop_words = stopwords.words('english')

	stoplist_extra=['amp','youd', 'wed' ,'id', 'get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via', 'seven', 'eight', 'nine', 'ten',
            'one','com','new','like','great','make','top','good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice', ' shes', 'hes ', 'were', 'theyre', 'yous',
            'two', 'three','four', 'five', 'six', 'while','know', 'ngl', 'brb', 'acc', 'smh', 'fwiw', 'ftl', 
            'lmao', 'lol', 'omg', 'https', 'mvp', 'isnt', 'arent', 'ever','cant', 'hnd', 'sbe', 'gsa', 'bwr']

    	# identify bigrams and unigrams to strip from tweets 

	counter = 0 
	
	with open('new_one.csv', 'a') as csvFile:
		csvWriter = csv.writer(csvFile)

		docs = []

		for username in documents.keys():

			#print("this is a key", username)
			counter += 1

			
			print('Number of iterations: ' + str(counter) + "  out of:  " + str(len(usernames_list)))
		
			documents[username] = " ".join(documents[username]).lower().split()
			
			
			#documents[username] = [re.sub(r"em", "email ", str(documents[username]))]
			
			#documents[username] = [re.sub(r"til", "learned ", str(documents[username]))]
			documents[username] = [re.sub(r"fb", "facebook ", word) for word in documents[username]]   # look for twitter most used acronyms						
			documents[username] = [re.sub(r"jk", "joking ", word) for word in documents[username]]
			documents[username] = [re.sub(r'@[^\s]+','', word) for word in documents[username]]
			documents[username] = [re.sub(r'#[^\s]+','', word) for word in documents[username]]

			
			# remove punctuation and replace with space and remove links 
			documents[username] = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in documents[username]]
			documents[username] = [re.sub(r"[\"'-,.;@#?!&$:'/]+\ *", " ", word) for word in documents[username]]

			
			# remove digits 

			documents[username] = [re.sub(r'\w*\d\w*', '', word) for word in documents[username]]

			print("creating unigrams and stopwords list.....")

			# remove stopwords 
			documents[username] =  ' '.join(documents[username]).split()

			unigrams = [ word for word in documents[username] if len(word)==1]
			bigrams  = [ word for word in documents[username] if len(word)==2]
			stoplist  = set(stop_words + stoplist_extra + unigrams + bigrams)

			documents[username] = [word for word in documents[username] if word not in stoplist]


			print("Stopwords have been removed ")

			documents[username] = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in documents[username]]
			documents[username] = [word for word in documents[username] if word in lemmas]


			docs = [documents[username] for username in documents.keys()]

			csvWriter.writerow([username, documents[username]])
	csvFile.close()

		

	return docs

def get_dictionary(documents):

	#sort words in documents 
	for doc in documents:
		doc.sort()


	# Build a dictionary where for each document each word has its own id
	dictionary = corpora.Dictionary(documents)
	dictionary.compactify()
	dictionary.save('tokens_dictionary.dict')

	# We now have a dictionary with 23065 unique tokens
	print(dictionary)
	return dictionary


def get_corpus(dictionary, documents):

	'''
	This function creates a corpus where each word is represented as a vector 
	with the occurrence of each word for each document
	It then converts the tokenized documents into vectors. 
	'''
	corpus = [dictionary.doc2bow(doc) for doc in documents]

	# and save in Market Matrix format
	corpora.MmCorpus.serialize('corpus.mm', corpus)
	# this corpus can be loaded with corpus = corpora.MmCorpus('corpus.mm')

	return corpus



if __name__ == '__main__':



	documents, usernames_list = get_documents('all_tweets.csv')

	print("Corpus of %s documents" % len(documents.values()))
	
	docs = preprocessing_data(documents)

	with open("documents.txt", "wb") as file:
		pickle.dump(docs, file)

	print("TWEET PREPROCESSING COMPLETED")
	print("---------------------------------")

	dictionary = get_dictionary(docs)
	corpus = get_corpus(dictionary, docs)



