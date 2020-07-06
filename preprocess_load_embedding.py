# Importing Sckit-learn, spacy, scikit-learn Libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import spacy
import re
import string
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
import nltk

# NLTK, Spacy To Remove Stopwords from Dataset or for Lemmatization
nltk.download('stopwords')
from nltk.corpus import stopwords

spacy.require_gpu()
tok = spacy.load('en_core_web_sm')
all_stopwords = tok.Defaults.stop_words
stop = stopwords.words('english')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


class DataPreprocessing(object):
    """
    Class Defined for Data Preprocessing, data cleaning, lemmatization, tokenization, word-embedding, words to glove
    vectors, Multilabel Binarizing for Multilabel Encoding, Base-line modelling
    """

    def __init__(self, labelled_message_file, empathies_file):
        super(DataPreprocessing, self).__init__()
        self.labeled_data = pd.read_csv(labelled_message_file)  # Read Message-Empathies file
        self.empathies = pd.read_csv(empathies_file)  # Read Empathy-polarity file
        self.counts = Counter()  # count number of occurences of each word
        self.vocab2index = {"": 0, "UNK": 1}  # Dictionary for corpus word that maps to unique number
        self.words = ["", "UNK"]  # List of (Dataset) Corpus words
        self.output_size = None  # Output Size -> Total Empathies
        self.loss_weights = None  # Weights for BCE Loss Functions to handle Imbalance in the dataset
        self.word_vectors = {}
        self.mlb = MultiLabelBinarizer()  # Initializing Multilabel Binarizing for Multilabel Encoding

        print(f'Data Shape: {self.labeled_data.shape}')
        print("First 5 Columns of Data:")
        print(self.labeled_data.head())
        print('\n')
        self.preprocess()
        self.create_dict()

        # Converts message to encoded pattern
        self.labeled_data['encoded'] = self.labeled_data['message'].apply(lambda x: np.array(self.encode_sentence(x)))

        # Fill Missing Values with 'idk' and cleans empathy labels like whitespacing
        self.labeled_data['y_encoded'] = self.labeled_data['empathy'].apply(
            lambda x: ['idk'] if pd.isnull(x) else [w.strip() for w in x.split(',')])
        self.empathies_polarity = {}

        # Creates dictionary for emapthies that maps to polarity
        self.process_polarity_empathies()

    def preprocess(self):
        del self.labeled_data['ignore']  # Ignore Column Contains all NANs values. It was deleted

        # Process text in message column for 'im'. 'im' converts to 'I am' using regex
        self.labeled_data['message'] = self.labeled_data['message'].str.replace(r"\bim\b", 'i am', regex=True)

        self.labeled_data['message'] = self.labeled_data['message'].apply(lambda x: " ".join(x.split()))

        # Removal of Stop Words in messages dataset (Corpus). Removing stop words does not help too much in test scores
        # self.labeled_data['message']=self.labeled_data['message'].apply(lambda x: ' '.join([word for word in x.split()
        #                                                                                      if word not in stop]))

        # Lemmatization of words
        # self.labeled_data['message'] = self.labeled_data['message'].apply(lambda x: ' '.join(self.lemmatize_text(x)))
        self.labeled_data['message_length'] = self.labeled_data['message'].apply(lambda x: len(x.split()))

        # Counts Occurence of the words in the corpus
        for index, row in self.labeled_data.iterrows():
            self.counts.update(self.tokenize(row['message']))

    def tokenize(self, text):
        # Tokenize the text
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]

    def describe_counts(self):
        print("Number of words in Corpus: {}".format(len(self.counts)))
        print("Message Avg Length : {}, Message Max Length : {}".format(self.labeled_data['message_length'].mean(),
                                                                        max(self.labeled_data['message_length'])))

    def create_dict(self):
        for word in self.counts:
            # Dictionary for corpus word that maps to unique number
            self.vocab2index[word] = len(self.words)
            # Append the words list
            self.words.append(word)

    def encode_sentence(self, text, N=100):
        tokenized = self.tokenize(text)
        encoded = np.zeros(N, dtype=int)

        # Creates encoding for each message
        enc1 = np.array([self.vocab2index.get(word, self.vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))  # Constraint on length of message
        encoded[:length] = enc1[:length]  # Zero padding for rest of positions  if length is less than enc1's length
        return encoded, length

    def lemmatize_text(self, text):
        # Lemmatization of text
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    def label_binarizer_get_weights(self):
        labels = self.labeled_data['y_encoded'].values.tolist()

        # Transforms y-labels to Mult-label Binarizer
        y_values = self.mlb.fit_transform(labels)

        # Total Number of Empathies
        self.output_size = len(self.mlb.classes_)
        print(f'Number of Empathies: {self.output_size}, Output Shape is: {y_values.shape}')

        # All empathy-labels in long list
        all_labels_flattened = [w for list_emp in labels for w in list_emp]

        # Counts Frequency of empathy-label in corpus
        count_labels = Counter(all_labels_flattened)
        # print(count_labels, len(count_labels))

        # Weighted Sampling for Imbalance: To give higher weight to less frequent Empathy-label
        base = np.zeros(self.output_size)
        weight = np.zeros(self.output_size)

        # Mapping from i to empathy-label
        self.d = {}
        self.rev_d = {}
        for i in range(self.output_size):
            base[i] = 1
            empathy_label = self.mlb.inverse_transform(np.array([base]))[0][0]
            self.d[i] = empathy_label
            self.rev_d[empathy_label] = i
            # Frequency of Label maps to weight
            weight[i] = count_labels[empathy_label]
            base = np.zeros(len(self.mlb.classes_))

        # Inverses weights for balance sampling
        weight = sum(weight) / weight
        weight = weight / sum(weight)
        self.loss_weights = torch.from_numpy(weight)

        # Final Y-Values for Multi-Label Encoding
        self.labeled_data['y_encoded_int'] = pd.Series(y_values.tolist())
        print('\n')
        print("First 5 Columns of Data After Preprocessing and MultiLabel Encoding:")
        print(self.labeled_data.head())
        print('\n')

        return self.output_size, self.loss_weights

    # Method to add numseen_feature in the dataset, by appending it to their respective encoding
    def add_numseen_feature(self):
        leng = len(self.labeled_data)
        for i in range(leng):
            self.labeled_data['encoded'].iloc[i][0] = np.append(self.labeled_data['encoded'].iloc[i][0],
                                                                self.labeled_data['num_seen'].iloc[i])

    def load_glove_vectors(self, glove_file="/media/HDD_2TB.1/glove.6B.50d.txt"):
        """Loading the glove word vectors"""
        with open(glove_file) as f:
            for line in f:
                split = line.split()
                # Fetching GLove vector for Word
                self.word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
        return self.word_vectors

    def get_emb_matrix(self, pretrained, emb_size=50):
        """ Creates embedding matrix from word vectors"""
        vocab_size = len(self.counts) + 2
        vocab_to_idx = {}  # dictionary from word to index
        vocab = ["", "UNK"]
        W = np.zeros((vocab_size, emb_size), dtype="float32")
        W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
        W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
        vocab_to_idx["UNK"] = 1  # "" for 0 , "UNK" for 1, rest of corpus words index starts from 2
        i = 2
        for word in self.counts:
            if word in pretrained:
                W[i] = pretrained[word]  # adding a vector for recognized word
            else:
                W[i] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
            vocab_to_idx[word] = i
            vocab.append(word)
            i += 1
        return W, np.array(vocab), vocab_to_idx

    # Methods creates empathies_polarity dic that maps empathy to polarity
    def process_polarity_empathies(self):
        emps = list(self.empathies['empathy'])
        polarity = list(self.empathies['polarity'])

        for e, p in zip(emps, polarity):
            self.empathies_polarity[e] = p

    # Retrieves Total Number of Classes
    def get_outputsize(self):
        return self.output_size

    # # Retrieves Weights for BCE-Loss Function
    def get_weights(self):
        return self.loss_weights

    # Retrieves X
    def get_X_data(self):
        return list(self.labeled_data['encoded'])

    # Retrieves Y
    def get_Y_data(self):
        return list(self.labeled_data['y_encoded_int'])

    # Retrieves Glove Word Vectors
    def get_word_vec(self):
        return self.word_vectors

    # Method Call for Baseline Modelling, Classifier RandomForest and SVM
    def modelling(self, baseclassifier, X_train, X_valid, y_train, y_valid):
        # Base Model either Random Forest or SVM, but both one are used
        if baseclassifier == 'RandomForest':
            clf = MultiOutputClassifier(RandomForestClassifier(max_depth=6, class_weight="balanced", n_jobs=2))
        elif baseclassifier == 'SVC':
            clf = MultiOutputClassifier(SVC(gamma='auto', class_weight="balanced"))

        clf_train = np.array([row[0] for row in X_train])
        clf_valid = np.array([row[0] for row in X_valid])

        # Fitting the Model and then predict on validation
        clf.fit(clf_train, y_train)
        y_pred = np.array(clf.predict(clf_valid))
        y_valid = np.array(y_valid)

        # Calculating Accuracy and AUC Scores
        return accuracy_score(y_valid.reshape(-1), y_pred.reshape(-1)), \
               roc_auc_score(y_valid.reshape(-1), y_pred.reshape(-1))


class EmpathyDataset(Dataset):
    """Data Loader while torch loading subset for epochs"""

    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return X-data, Y-value for it
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), torch.from_numpy(
            np.array(self.y[idx]).astype(np.int32)), torch.tensor(self.X[idx][1])
