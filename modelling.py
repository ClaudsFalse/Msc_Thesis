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



##------------------------HELPER FUNCTIONS-----------------------------------##







def get_non_aggregated_docs(filename):
	"""
	This function extracts the documents from the csv file where they are all saved
	It saves the data in a pandas dataframe and save the tweets in a list.

	Inputs: the name of the csv file where the corpus is stored
	Outputs: a list of tweets, where each tweet is a sublist in the tweets list. 

	"""
    data_frame = pd.read_csv(filename, names = ["TweetID", "Username", "Text"])
    df = data_frame.sort_values(by = "TweetID").drop_duplicates("Text")
    tweets = df['Text'].tolist()
    
    return tweets



def get_wordnet_pos(word):
    """
	This function map POS tag to first character that the function lemmatize() accepts

	Inputs: a word in the tweet
	Outputs: the part of speech tag for the word formatted as a TAG for the function lemmatize 
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_non_aggregated_tweets(documents):
    '''
    Function that lowercases the text and clean it
    Iterates over every tweet in the corpus and breaks the tweets into tokens
    It then removes punctuations, special characters and stopwords. 
    
    Inputs: a collection of tweets as a list  
    Outputs: a list of preprocessed tweets 
    '''
    
    lemmatizer = WordNetLemmatizer()
    lemmas = set(wordnet.all_lemma_names())
    
    stop_words = stopwords.words('english')

    ## here we define iteratively additional words to be removed from the corpus
    
    stoplist_extra=['amp','youd', 'wed' ,'id', 'get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
                    'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via', 'seven', 'eight', 'nine', 
                    'ten','one','com','new','like','great','make','top','good','wow','yes','say','yay','would',
                    'thanks','thank','going', 'new','use','should','could','best','really','see','want','nice',
                    'shes', 'hes ', 'were', 'theyre', 'yous','two', 'three','four', 'five', 'six', 'while',
                    'know', 'ngl', 'brb', 'acc', 'smh', 'fwiw', 'ftl','lmao', 'lol', 'omg', 'https', 'mvp', 
                    'isnt', 'arent', 'ever','cant', 'hnd', 'sbe', 'gsa', 'bwr', 'yeah', 'fuck', 'torture',
                    'shit','say', 'try', 'well', 'take', 'way', 'many', 'yet','never', 'may', 'come', 
                    'actually', 'much', 'thing', 'year', 'month','people', 'also', 'around', 
                    'keep', 'time', 'someone', 'give', 'back', 'even', 'every', 'tell', 'first', 'lot', 
                    'sure', 'though', 'end', 'still', 'bitch',
                	'already', 'always', 'thought','right', 'call', 'put', 'long',
                    'mak', 'cunt', 'aku', 'dia', 'yang','dah', 'masa', 'lepas', 'dari','joseph', 
                    'miller', 'ada', 'orang', 'kat', 'lah','zaman','gila','akan','tau','salah','mak',
                    'kali','ali', 'susah','lama','bahasa', 'het', 'said', 'years', 'fucking', 'things',
                   'life', 'day', 'enough', 'ass', 'u', 'made', 'make', 'makes', '']
    
    # we initialise a list to store the preprocessed tweets in
    tweets = []
    
    for tweet in documents:
    	# iterate the preprocessing steps for every tweet in the collection 
        
        tweet = tweet.lower().split()  # lowercase and tokenise the tweet

        # here we standardise common twitter abbreviations into the extended version of the token
        tweet = [re.sub(r"fb", "facebook ", word) for word in tweet]
        tweet = [re.sub(r"jk", "joking ", word) for word in tweet]

        # with this, we remove every user tag in the tweet of the form @username, as well as other @ characters in the tweet
        tweet = [re.sub(r'@[^\s]+','', word) for word in tweet]

   		 # remove punctuation and replace with space and remove hyper-links,
        tweet = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in tweet]
        tweet = [re.sub(r"[\"\'-,.;@?!&$:'/]+\ *", " ", word) for word in tweet]
    
   		 # remove standalone digits, words with digits in them  and numbers
        tweet = [re.sub(r'\w*\d\w*', '', word) for word in tweet]
        tweet =  ' '.join(tweet).split()
    
    	# identify unigrams and bigrams in the corpus to be added to the stopwords list and then removed. 
        unigrams = [word for word in tweet if len(word)==1]
        bigrams  = [word for word in tweet if len(word)==2]

        #remove stopwords 
        stoplist  = set(stop_words + stoplist_extra + unigrams + bigrams)
    	
    	# lemmatize the tweets using the NLTK function
        tweet = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tweet]

        # excluding hashtags from lemmatization
        tweet = [word for word in tweet if word in lemmas or word[0] == '#']
        tweet = [word for word in tweet if word not in stoplist]
        tweet =  ' '.join(tweet).split()
        
    	# finally, we discard empty tweets (in case the preprocessing steps stripped the whole tweet) 
    	# and append to the list of tweets 
        if tweet != []:
            tweets.append(tweet)
            
    
    return tweets



def get_dictionary(documents):
	"""
	    
    Build a dictionary where for each document each word has its own id
    Input: the list of preprocessed tweets 
    Outputs: the tokens dictionary 
	"""
    dictionary = corpora.Dictionary(documents)
    dictionary.compactify()
    dictionary.save('tokens_dictionary.dict')
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



def prepare_corpus(documents):
    '''
    This function creates the term dictionary of our corpus, where every unique term is assigned an index. 
    Input : preprocessed tweets
    Purpose: create term dictionary of our corpus and converting the list of documents 
                into a term documents matrix 
    Output: term dictionary and Document Term Matrix 
    '''
    
    dictionary = get_dictionary(documents)
    doc_term_matrix = get_corpus(dictionary, documents)
    return dictionary, doc_term_matrix




def gensim_LSA(documents, num_of_topics, words):
    """

	This function creates an LSA model using Gensim 
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    
    dictionary, doc_term_matrix = prepare_corpus(documents)
    LSA = LsiModel(doc_term_matrix, num_topics = num_of_topics, id2word = dictionary)
    print(LSA.print_topics(num_topics = num_of_topics, num_words = words))
    
    return LSA


def get_coherence_score(model, docs):

	"""
	This function computes coherence scores for the model
	Inputs: the lsa model and the list of tweets
	Outputs: the coehrence score computed through U-Mass measure
	"""    
    dictionary = get_dictionary(documents)
    coherence_model = CoherenceModel(model = model, texts = docs, dictionary = dictionary, coherence = 'u_mass')
    coherence_score = coherence_model.get_coherence()
    
    print('\nCoherence Score: ', coherence_score)
    return coherence_score


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    This function computes coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        model = models.LsiModel(corpus, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values




def plot_scores(documents, start, stop, step, coherence_values):

	"""
	This function plots the coherence scores computed for different 
	values of k (number of topics)
	-----
	Parameters: list of tweets, k starting and stop point, step of search
				list of coherence values for every model
	Returns: a matplotlib plot
	"""
    
    # plot 
    x = range(start,stop,step)
    plt.plot(x,coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show
    
    return model_list, coherence_values



