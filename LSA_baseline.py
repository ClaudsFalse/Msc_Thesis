# Author: Claudia Falsetti <1431314>
# Python 3.7.3 


import re 
import csv
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation, digits
from collections import Counter, defaultdict

from nltk.corpus import stopwords, words, wordnet 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models, similarities
from gensim import corpora
from gensim.models import LsiModel

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from modelling import modelling



if __name__ == '__main__':
	
	documents = get_non_aggregated_docs('all_tweets.csv')
	print("Corpus of %s documents" % len(documents))

	# Get preprocessed tweets
	processed_tweets = preprocess_non_aggregated_tweets(documents)

	# Keep only tweets with more than 2 words 

	tweets = []

	for tweet in processed_tweets:
		if len(tweet) > 2:
			tweet = [word.replace('\\', '') for word in tweet]
			tweets.append(tweet)


## ------------------------------------------------------DESCRIPTIVE STATISTICS ----------------------------------------------##
	## compute frequency counts and plot wordclouds

	hashtags = []

	for tweet in tweets:
    for word in tweet:
        if word[0] == '#':
            hashtags.append(word)


	hash_count = defaultdict(int)

	for hashtag in hashtags:
		hash_count[hashtag] += 1


	## HASHTAG WORDCLOUD


	wc = WordCloud(background_color='white',
                  width=5000,
                  height=3000,
                  max_words=50).generate_from_frequencies(hash_count)
	fig = plt.figure(figsize=(9,9))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.show()
	fig.savefig('hashtag_wordcloud.png')


	## TOKENS WORDCLOUD

	word_count = defaultdict(int)

	for tweet in tweets:
    	for word in tweet:
        	word_count[word] += 1

        
	wc = WordCloud(background_color='white',
                  width=5000,
                  height=3000,
                  max_words=50).generate_from_frequencies(word_count)

	fig = plt.figure(figsize=(9,9))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.show()
	fig.savefig('word_freq_cloud.png')


###------------------------------- LSA MODEL ---------------------------------------------##

	## Coherence Score

	dictionary_non_agg = corpora.Dictionary(tweets)
	dictionary_non_agg.compactify()

	corpus_non_agg = [dictionary_non_agg.doc2bow(tweet) for tweet in tweets]
	corpora.MmCorpus.serialize('corpus_non_agg.mm', corpus_non_agg)


	coherence_model = CoherenceModel(model = LSA, texts = tweets, dictionary = dictionary_non_agg, coherence = 'u_mass')
	coherence_score = coherence_model.get_coherence()
    
	print('\nCoherence Score: ', coherence_score)

	#grid-search
	model_list, coherence_values = compute_coherence_values(dictionary=dictionary_non_agg, corpus=corpus_non_agg, texts=tweets, start=2, limit=40, step=2)

	start, stop,step = 2, 40, 2
	plot_scores(tweets,start,stop,step, coherence_values)

	# Build LSA model

	#words to be printed per topic
	words = 20
	num_of_topics = 5

	LSA = gensim_LSA(tweets, num_of_topics, words)

	coherence_model = CoherenceModel(model = LSA, texts = tweets, dictionary = dictionary_non_agg, coherence = 'u_mass')
	coherence_score = coherence_model.get_coherence()
    
	print('\nCoherence Score: ', coherence_score)
