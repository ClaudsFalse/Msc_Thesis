from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
from biterm.cbtm import oBTM

def vectorize_data(path):

	texts = open(path).read().splitlines()[:50]
	vec = CountVectorizer(stop_words = 'english')

	X = vec.fit_transform(texts).toarray()

	return X


def get_vocab(X):

	vocab = np.array(vec.get_feature_names())
	biterms = vec_to_biterms(X)

	return vocab, biterms

def BTM(x, vocab, biterms):
	btm = oBTM(num_topics = 20, V = vocab)
	topics = btm.fit_transform(biterms, iterations = 100 )


x = vectorize_data('./more_tweets.csv')

