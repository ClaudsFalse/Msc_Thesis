
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




def get_documents_aggregated(filename):


    """
    This function uploads the retrieved tweets into a collection of documents
    ready to be preprocessed.

    Inputs: the name of the file where all the tweets are saved
    Outputs: the corpus as a dictionary mapping each  anonymised username to their tweets 
             and the list of users (the keys of the dictionary)

    """

    data_frame = pd.read_csv(filename, names=["TweetID","Username", "Text"])
    df = data_frame.sort_values(by='Username').drop_duplicates('Text')
    data_frame_new = df.iloc[:,1:3]
    #print(data_frame_new.head(5))

    usernames = data_frame_new.iloc[:,0]

    usernames_list = set(usernames.values.tolist())


    documents = data_frame_new.groupby('Username')['Text'].apply(list).to_dict()

    return documents


def preprocessing_data_aggregated(documents):

    """
    Function that preprocess the data. It lowercases the text and cleans it
    Break the sentences into tokens, remove punctuations and stopwords
    Standardise the text : can't -> cannot, I'm -> I am, fb -> facebook

    Inputs: a dictionary of username - tweets pairs
    Outputs: a collection of preprocessed tweets as a list of lists. 
    """

    lemmatizer = WordNetLemmatizer()
    lemmas = set(wordnet.all_lemma_names())

    stop_words = stopwords.words('english')

    stoplist_extra=['amp','youd', 'wed' ,'id', 'get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via', 'seven', 'eight', 'nine', 'ten',
            'one','com','new','like','great','make','top','good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice', ' shes', 'hes ', 'were', 'theyre', 'yous',
            'two', 'three','four', 'five', 'six', 'while','know', 'ngl', 'brb', 'acc', 'smh', 'fwiw', 'ftl', 
            'lmao', 'lol', 'omg', 'https', 'mvp', 'isnt', 'arent', 'ever','cant', 'hnd', 'sbe', 'gsa', 'bwr', 'yeah', 'fuck', 'torture',
            'shit','say', 'try', 'well', 'take', 'way', 'many', 'yet','never', 'may', 'come', 'actually', 'much', 'thing', 'year', 'month',
            'people', 'also', 'around', 'think', 'keep', 'time', 'someone', 'give', 'back', 'need', 'time', 'feel', 'look', 'even', 
            'start', 'every', 'tell', 'first', 'lot', 'sure', 'though', 'end', 'still', 'bitch',
                   'wait', 'watch', 'already', 'always', 'thought','right', 'call', 'put', 'long',
                   'mak', 'cunt', 'aku', 'dia', 'yang','dah', 'masa', 'lepas', 'dari', 
                   'joseph', 'miller', 'ada', 'orang', 'kat', 'lah','zaman','gila','akan','tau','salah','mak','kali','ali',
                   'susah','lama','bahasa', 'het', 'ass', 'fucking']

        # identify bigrams and unigrams to strip from tweets 

    counter = 0 

    docs = []

    for username in documents.keys():
        counter += 1
        documents[username] = " ".join(documents[username]).lower().split()
        documents[username] = [re.sub(r"fb", "facebook ", word) for word in documents[username]]   # look for twitter most used acronyms						
        documents[username] = [re.sub(r"jk", "joking ", word) for word in documents[username]]
        documents[username] = [re.sub(r'@[^\s]+','', word) for word in documents[username]]
        #documents[username] = [re.sub(r'#[^\s]+','', word) for word in documents[username]]

        # remove punctuation and replace with space and remove links 
        documents[username] = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in documents[username]]
        documents[username] = [re.sub(r"[\"'-,.;@?!&$:'/]+\ *", " ", word) for word in documents[username]]

        # remove digits 
        documents[username] = [re.sub(r'\w*\d\w*', '', word) for word in documents[username]]
        print("creating unigrams and stopwords list.....")

        # remove stopwords 
        documents[username] =  ' '.join(documents[username]).split()
        unigrams = [ word for word in documents[username] if len(word)==1]
        bigrams  = [ word for word in documents[username] if len(word)==2]
        stoplist  = set(stop_words + stoplist_extra + unigrams + bigrams)

        documents[username] = [word for word in documents[username] if word not in stoplist]
       
        documents[username] = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in documents[username]]
        documents[username] = [word for word in documents[username] if word in lemmas]
        documents[username] = [word for word in documents[username] if word not in stoplist]

        docs = [documents[username] for username in documents.keys()]

    return docs



    def get_dictionary_aggregated(documents):
    #sort words in documents 
    for doc in documents:
        doc.sort()


    # Build a dictionary where for each document each word has its own id
    dictionary = corpora.Dictionary(documents)
    dictionary.compactify()
    dictionary.save('tokens_dictionary.dict')

    # We now have a dictionary with 1780 unique tokens
    print(dictionary)
    return dictionary


def get_corpus_aggregated(dictionary, documents):

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


# Determine the optimum Number of Topics by generating 
# coherence scores and plot the coherence scores 

def compute_coherence_values_aggregated(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

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
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, alpha = 0.001)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



    # Can take a long time to run.
#model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=2, limit=40, step=2)


def plot_scores_aggregated(documents, start, stop, step, dictionary, corpus):
    model_list, coherence_values = compute_coherence_values_aggregated(dictionary, corpus, documents, stop,start,step)

    
    # plot 
    x = range(start,stop,step)
    plt.plot(x,coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show
    
    return model_list, coherence_values

def gensim_LSA(documents, num_of_topics, words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    
    dictionary, doc_term_matrix = prepare_corpus(documents)
    LSA = LsiModel(doc_term_matrix, num_topics = num_of_topics, id2word = dictionary)
    print(LSA.print_topics(num_topics = num_of_topics, num_words = words))
    
    return LSA