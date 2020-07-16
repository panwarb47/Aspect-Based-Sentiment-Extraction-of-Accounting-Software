from string import punctuation, digits
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker
import numpy as np
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Flatten, Softmax
from sklearn import preprocessing
from keras.models import Sequential
from keras.optimizers import Adam

def extract_data_from_dict(df):
    sentence_list = []
    aspect_list = []
    for i in range(len(df)):
        sentence_list.append(df['Review'][i])
        aspect_list.append(df['ac'][i])
    return sentence_list, aspect_list


def remove_punctuation(s):
    list_punctuation = list(punctuation)
    for i in list_punctuation:
        s = s.replace(i, '')
    return s


stop_words = set(stopwords.words('english') + ['a', 'doesnt', 'dont'])


def clean_sentence(sentence):
    spell = SpellChecker()
    sentence = sentence.lower()
    # remove multiple repeat non num-aplha char !!!!!!!!!-->!
    sentence = re.sub(r'(\W)\1{2,}', r'\1', sentence)
    # removes alpha char repeating more than twice aaaa->aa
    sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
    # removes links
    sentence = re.sub(r'(?P<url>https?://[^\s]+)', r'', sentence)
    # remove @usernames
    sentence = re.sub(r"\@(\w+)", "", sentence)
    # removing stock names to see if it helps
    sentence = re.sub(r"(?:\$|https?\://)\S+", "", sentence)
    # remove # from #tags
    sentence = sentence.replace('#', '')
    sentence = sentence.replace("'s", '')
    sentence = sentence.replace("-", ' ')
    sentence = sentence.replace("€", '')
    sentence = sentence.replace("¢", '')
    sentence = sentence.replace("™", '')
    sentence = sentence.replace("â", '')
    sentence = sentence.replace("œ", '')

    #     print(sentence)
    # split into tokens by white space
    tokens = sentence.split()
    # remove punctuation from each token
    tokens = [remove_punctuation(w) for w in tokens]
    #     print(tokens)
    #     remove remaining tokens that are not alphabetic
    #     tokens = [word for word in tokens if word.isalpha()]
    # no removing non alpha words to keep stock names($ZSL)
    # filter out stop words

    #     for w in stop_words:
    #         print(w)
    #     print(tokens)
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    #     print(tokens)
    remove_digits = str.maketrans('', '', digits)
    #     print(tokens)
    tokens = [w.translate(remove_digits) for w in tokens]
    tokens = [w.strip() for w in tokens]
    tokens = [w for w in tokens if w != ""]
    tokens = [spell.correction(w) for w in tokens]
    # print('len1'+str(len(tokens)))
    tokens = [w for w in tokens if not w in stop_words]
    # print(len(tokens))
    tokens = ' '.join(tokens)
    return tokens


def convert_lables(trainY, testY, no_of_classes):
    le = preprocessing.LabelEncoder()
    le.fit(trainY+testY)
    temp1 = le.transform(trainY)
    temp2 = le.transform(testY)
    return to_categorical(temp1, no_of_classes), temp2, le.classes_


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode a list of lines
def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


def load_word2vec(file_name):
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


def get_embedding_matrix(model, tokenizer, vocab_size, emb_size):
    embedding_matrix = np.zeros((vocab_size, emb_size))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def define_model(learning_rate, dropout, lstm_out, n_hidden_layer, em, em_trainable_flag, free_em_dim, vocab_size, no_of_classes, max_length):
    model = Sequential()
    if em == 'free':
        model.add(Embedding(vocab_size, free_em_dim))
    else:
        model.add(Embedding(vocab_size, len(eval(em)[0]), weights=[eval(em)], input_length=max_length, trainable=em_trainable_flag))
    model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=dropout)))
    for i in range(n_hidden_layer):
        model.add(Dense(int((2*lstm_out+no_of_classes)/2), activation='relu'))
    model.add(Dense(no_of_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer,metrics = ['accuracy'])
    print(model.summary())
    return model