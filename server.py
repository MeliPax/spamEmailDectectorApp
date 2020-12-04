import streamlit as st
import pickle
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import codecs
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow import keras


SEQUENCE_LENGTH = 100
label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}

# Loading the model
filename = "customer_churn_model.pkl"
# model = keras.models.load_model(filename, 'rb')
model = pickle.load(open(filename, 'rb'))
# model = load_model("network.h5", 'rb')
# model = pickle.load(open('network.h5', 'rb'))
# classifier = pickle.load(model)

# loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', str(text))


def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]

# def classify(var):
#     if()


def main():
    # st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">SpamEmail Classification App</h2>
    </div>
    <br><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    input = st.text_input("Dependents (0 if no , or 1 if yes)", "")
    if st.button('Classify'):
        # if (len(text)
        # Running prediction
        st.success(classify(get_predictions(input)))

# Printing to the user


def classify(val):
    if val == 0:
        return 'Spam'
    else:
        return 'not spam'


if __name__ == '__main__':
    main()
