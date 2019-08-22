# Author: Claudia Falsetti <1431314>
# Python 3.7.3 

from modelling import modelling
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



if __name__ == '__main__':
	documents = get_non_aggregated_docs('all_tweets.csv')
	print("Corpus of %s documents" % len(documents))

	processed_tweets = preprocess_non_aggregated_tweets(documents)

	# Keep only tweets with more than 2 words 

	tweets = []

	for tweet in processed_tweets:
		if len(tweet) > 2:
			tweet = [word.replace('\\', '') for word in tweet]
			tweets.append(tweet)
    
    dictionary_non_agg = corpora.Dictionary(tweets)
    dictionary_non_agg.compactify()

    corpus_non_agg = [dictionary_non_agg.doc2bow(tweet) for tweet in tweets]
    corpora.MmCorpus.serialize('corpus_non_agg.mm', corpus_non_agg)
    

    # get the number of topics via HDP
	hdp = models.HdpModel(corpus_non_agg, id2word=dictionary_non_agg)
	topics = hdp.print_topics(num_words=30)
	print(len(topics))

	lda_filename    = 'LDA_non_agg.lda'
	lda_params      = {'num_topics': topics, 'passes': 100, 'alpha': 0.001}

	# print("Running LDA with: %s  " % lda_params)

	lda = models.LdaModel(corpus_non_agg, id2word=dictionary_non_agg,
                        num_topics=lda_params['num_topics'],
                        passes=lda_params['passes'],
                        alpha = lda_params['alpha'])

	# uncomment the line below to print all the topics from the LDA model and the words distribution across them 
	
	# print(lda.print_topics(num_topics = lda_params['num_topics'], num_words = 20)) 
