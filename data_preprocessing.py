from nltk.tokenize import TweetTokenizer, TreebankWordTokenizer

# imports to tokenize

nltk.download('wordnet')
nltk.download('stopwords')

class data_processor:
	def __init__(self, vectoriser = CountVectorizer(), tokeniser = None, cleaning_function = None, lemmatizer = None, model = None):

		"""
		A class to pipeline the data pre_processing. 
		Inputs are:
		vectoriser = the model used to vectorise the data
		tokeniser = the model used to tokenise the data. If set to none, the class will defaul to separate tokens by space. 
		cleaning_function = function to clean the data. 


		"""

		if not tokeniser:
			tokeniser = self.default_split

		if not cleaning_function:
			cleaning_function = self.preprocess

		self.lemmatizer = lemmatizer
		self.tokeniser = tokeniser
		self.model = model
		self.cleaning_function = cleaning_function
		self.vectoriser = vectoriser
		self._is_fit = False





	def default_split(self, text):
		"""
		Class-default tokenizer that splits token on spaces 
		"""
		return text.split(' ')

	def preprocess(self, text, tokeniser, lemmatizer):

		"""
		A function that takes as input the text, and lowercase it. It also cleans it by applying lemmatisation and tokenisation


		"""

		processed_text = []
		for tweet in text:
			processed_words = []
			for word in tokeniser(tweet):
				lowercased = word.lower()

				if lemmatizer:
					lowercased = lemmatizer.lemmatize(lowercased)

				processed_text.append(' '.join(processed_words))

		return processed_text



	def model_fit(self, text):

		"""
		cleans the data and then fits the vectoriser with the dataset 

		"""
		process_text = self.cleaning_function(text, self.tokeniser, self.lemmatizer)
		self.vectoriser.fit(process_text)

		self._is_fit = True


	def transform(self, text):
		 """
        Cleans any provided data and then transforms the data into
        a vectorized format based on the fit function. Returns the
        vectorized form of the data.
        """
        if not self._is_fit:
            raise ValueError("Must fit the models before transforming!")
        clean_text = self.cleaning_function(text, self.tokenizer, self.lemmatizer)
        return self.vectorizer.transform(clean_text)
    


    def save_pipe(self, filename):
        """
        Writes the attributes of the pipeline to a file
        allowing a pipeline to be loaded later with the
        pre-trained pieces in place.
        """
        if type(filename) != str:
            raise TypeError("filename must be a string")
        pickle.dump(self.__dict__, open(filename+".mdl",'wb'))


    def load_pipe(self, filename):
        """
        Writes the attributes of the pipeline to a file
        allowing a pipeline to be loaded later with the
        pre-trained pieces in place.
        """
        if type(filename) != str:
            raise TypeError("filename must be a string")
        if filename[-4:] != '.mdl':
            filename += '.mdl'
        self.__dict__ = pickle.load(open(filename,'rb'))


        