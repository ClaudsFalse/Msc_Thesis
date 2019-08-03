import pandas as pd 
from  nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, words

from string import punctuation, digits
from sklearn.feature_extraction.text import TfidfVectorizer

import langid
from collections import Counter 
import re 
import numpy as np 
from tqdm import tqdm_notebook



def get_documents(filename):

	data_frame = pd.read_csv(filename)
#	print(data_frame.shape)

#	values_expected = data_frame['Text']
#	tot_values_expected = len(values_expected)

#	values = []

#	for tweet in data_frame['Text']:
		#print(tweet)
#		values.append(tweet)

	data_frame_new = data_frame.iloc[:, 1:3]


	documents = data_frame_new.groupby('Username')['Text'].apply(list).to_dict()
	#dictionary = data_frame_new.set_index('Username').T.to_dict()

	return documents



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

	# Remove stop words
	words = set(words.words())

	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = stopwords.words('english')
	stoplist_extra=['amp','get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via',
            'one','com','new','like','great','make','top','awesome','best',
            'good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice',
            'while','know']

	#processed_tweets = [re.sub(r"(?:\@|http?\://)\S+", "", doc)
     #           for doc in documents.values() ]

	processed_tweets = [ tokenizer.tokenize(str(doc).lower()) for doc in documents.values() ]

	unigrams = [  w for doc in documents.values() for w in doc if len(w)==1]
	bigrams  = [ w for doc in documents.values() for w in doc if len(w)==2]

	stoplist  = set(stop_words + stoplist_extra + unigrams + bigrams)

	processed_tweets = [[token for token in doc if token not in stoplist]
                for doc in documents.values()]

        # rm numbers only words
	processed_tweets = [ [token for token in doc if len(token.strip(digits)) == len(token)]
                for doc in documents.values() ]

  	#return documents

def remove_nonAscii(text):
	pass


def extend_abbreviations(text):

    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = removeNonAscii(text)
    text = text.strip()
    return text
    

def filter_nonenglish(text):
	pass 

'''

vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 2), stop_words='english')
vz = vectorizer.fit_transform(list(data['tokens'].map(lambda tokens: ' '.join(tokens))))
### rows = total number of tweets 
### columns total number of unique terms (tokens) = vocabulary across the documents 
vz.shape


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

	documents = get_documents('anonymised.csv')

	print("Corpus of %s documents" % len(documents.values()))

	#  Filter non english documents

#	documents = filter_lang('en', documents)

#	print("We have " + str(len(documents)) + " documents in english ")

	preprocessing_data(documents)