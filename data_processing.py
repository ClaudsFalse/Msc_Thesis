import pandas as pd 
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None

from  nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer


from collections import Counter 
import re 
import numpy as np 

from tqdm import tqdm_notebook

tqdm_notebook().pandas()

def stopwords_remover():
	"""
	Function that lowercase the text and clean it
	Break the sentences into tokens
	remove punctuations and stopwords
	Standardise the text : can't -> cannot, I'm -> I am
	"""

	stop_words = []


def removeNonAscii(text):
	return "".join(i for i in s if ord(i)<128)

def clean_text(text):
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
