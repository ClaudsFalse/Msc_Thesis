# Author: Claudia Falsetti <1431314>
# Python 3.7.3 


'''
This script loads a gensim dictionary and associated corpus
and applies an LDA model. 

'''

from gensim import corpora, models, similarities
import pyLDAvis.gensim
import numpy as np 
from time import time 
import pickle 


# ------------------- HELPER FUNCTIONS -----------------------# 

def get_docs(filename):
	with open(filename, "rb") as fp:
		docs = pickle.load(fp)

	return docs 


## ------------------## 
if __name__ == '__main__':

	docs = get_docs("documents.txt")

	print(docs)


	print("Corpus of %s documents" % len(docs))
	
	# initialise parameters 

	corpus_filename = 'corpus.mm'
	dict_filename   = 'tokens_dictionary.dict'
	lda_filename    = 'LDA.lda'
	lda_params      = {'num_topics': 40, 'passes': 20, 'alpha': 0.001}

	# Load the corpus and dictionary 
	corpus = corpora.MmCorpus(corpus_filename)
	dictionary = corpora.Dictionary.load(dict_filename)




	print("Running LDA with: %s  " % lda_params)

	lda = models.LdaModel(corpus, id2word=dictionary,
                        num_topics=lda_params['num_topics'],
                        passes=lda_params['passes'],
                        alpha = lda_params['alpha'])
	print()
	print(lda.print_topics())
	lda.save(lda_filename)
	print("lda saved in %s " % lda_filename)

	topics_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)
	pyLDAvis.display(topics_data)
	

