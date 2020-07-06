import spacy
import en_core_web_sm
from nltk import word_tokenize
import re
import pandas as pd
import nltk
spacy.require_gpu()
import string

nlp = en_core_web_sm.load()
doc = nlp("This is a sentence.")
print(doc)
print("OK")
tok = spacy.load('en_core_web_sm')

all_stopwords = tok.Defaults.stop_words

text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
new_text = ','.join(tokens_without_sw)
regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
nopunct = regex.sub(" ", new_text.lower())
ans = [token.text for token in tok.tokenizer(nopunct)]
print(tokens_without_sw)
print(ans)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


df = pd.DataFrame(['this was cheesy', 'she likes these books', 'wow this is great'], columns=['text'])
df['text_lemmatized'] = df['text'].apply(lambda x: ' '.join(lemmatize_text(x)))
print(df)
