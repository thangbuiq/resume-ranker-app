import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import contractions
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.preprocess_text)

    def preprocess_text(self, text):
        text = text.lower()
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.remove_emojis(text)
        text = self.expand_contractions(text)
        text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
        text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
        text = re.sub(r'#\S+', '', text)  # remove hashtags
        text = re.sub(r'@\S+', '  ', text)  # remove mentions
        text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
        return text

    @staticmethod
    def remove_html_tags(text):
        return BeautifulSoup(text, "html.parser").get_text()

    @staticmethod
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    @staticmethod
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text):
        return ' '.join([word for word in word_tokenize(text) if word.lower() not in self.stop_words])

    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def expand_contractions(text):
        return contractions.fix(text)
