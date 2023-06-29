#import required packages
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Text cleaning function with lemmatization
def clean_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a sentence
    cleaned_text = ' '.join(tokens)

    return cleaned_text

#testing function
#print("RT @darreljorstad: Funny as hell! Canada demands 'gender rights' and 'climate change' in a trade deal while Soviet dairy boards untouc…Type/Paste Here\n")
#print(clean_text("RT @darreljorstad: Funny as hell! Canada demands 'gender rights' and 'climate change' in a trade deal while Soviet dairy boards untouc…Type/Paste Here"))