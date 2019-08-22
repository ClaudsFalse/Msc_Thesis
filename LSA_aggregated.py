# Author: Claudia Falsetti <1431314>
# Python 3.7.3 

import pandas as pd 
from nltk.corpus import stopwords, words, wordnet 
from nltk.stem import WordNetLemmatizer 

from string import punctuation, digits
from collections import Counter, defaultdict
import re 

import numpy as np 
import csv 
import matplotlib.pyplot as plt

from gensim.models import HdpModel
from modelling_aggregated import modelling_aggregated
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models, similarities




if __name__ == '__main__':

	documents_aggregated = get_documents_aggregated('all_tweets.csv')
	print("Corpus of %s documents" % len(documents_aggregated.values()))
	docs = preprocessing_data_aggregated(documents_aggregated)

	# Keep only tweets with more than 2 words 

	tweets = []

	for tweet in docs:
		if len(tweet) > 2:
			tweet = [word.replace('\\', '') for word in tweet]
			tweets.append(tweet)
	
	dictionary_aggregated = corpora.Dictionary(tweets)
	dictionary_aggregated.compactify()
	corpus_aggregated =[dictionary.doc2bow(doc) for doc in tweets]


	plot_scores_aggregated(docs, 2, 40, 2, dictionary_aggregated, corpus_aggregated) 


	# get the number of topics via HDP
	hdp = models.HdpModel(corpus_aggregated, id2word=dictionary_aggregated)
	topics = hdp.print_topics(num_words=30)
	print(len(topics))


	# Build LSA model


	#words to be printed per topic
	words = 20
	num_of_topics = topics

	LSA = LsiModel(corpus, num_topics = num_of_topics, id2word = dictionary)
	print(LSA.print_topics(num_topics = num_of_topics, num_words = words))

	coherence_model = CoherenceModel(model = LSA, texts = tweets, dictionary = dictionary, coherence = 'u_mass')
	coherence_score = coherence_model.get_coherence()
	print('\nCoherence Score: ', coherence_score)