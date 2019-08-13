import pandas as pd 
from collections import defaultdict
from gensim import corpora


def filter_low_freq(filename):

	tokens_frequency = defaultdict(int)
	dataframe = pd.read_csv(filename, names=["Username","Text"])
	documents = dataframe.groupby('Username')['Text'].apply(list).to_dict()


	for username in documents.keys():

		documents[username] = " ".join(documents[username]).split()

		for token in documents[username]:
			
			tokens_frequency[token] += 1 

		documents[username] = [token if tokens_frequency[token] > 1]

		# Build a dictionary where for each document each word has its own id (prerequisite of Gensim LDA modelling)
		tokens_dictionary = corpora.Dictionary(documents[username])
	
	tokens_dictionary.compactify()
	tokens_dictionary.save('tokens_dictionary.dict')
	print("We now have a dictionary with %s unique tokens" % len(tokens_dictionary))


	return documents, tokens_frequency, tokens_dictionary


	return tokens_dictionary




if __name__ == '__main__':
	documents = filter_low_freq("new_one.csv")