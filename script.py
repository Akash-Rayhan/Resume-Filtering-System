import pandas as pd
import numpy as np
import re
import string
import fitz 
import pickle
import os
import argparse

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model



model = load_model('model/LSTM_model.keras')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
tokenizer,label_encoder, max_token = pickle.load(open("model/variables.p","rb"))

def clean_text(text):
    '''Make text lowercase,remove extra whitespaces, remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\s+', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('－', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess_text(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Lemmatize all words in the sentence
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))

    return text



def categorize_resume(resume_text):
    """
    This function preprocess resume texts and passing through the model generate prediction
    """
    content = np.array([preprocess_text(resume_text)])
    text_to_sequence = tokenizer.texts_to_sequences(content)
    pad_sequence = pad_sequences(text_to_sequence, maxlen=max_token, padding='post', truncating = 'post')
    prediction = np.argmax(model.predict(pad_sequence), axis=1)
    category = label_encoder.inverse_transform(prediction)[0]
    return category

def categorize_resumes_in_directory(input_dir):
    categorized_resumes = []
    categories = set()

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            # extract content from files 
            resume_path = os.path.join(input_dir, filename)
            doc = fitz.open(resume_path)
            text = " ".join([page.get_text() for page in doc])

            # retrieve model prediction and create tuple to add data on csv
            category = categorize_resume(text)
            categorized_resumes.append((filename, category))
            categories.add(category)

            # Create category folder if it doesn't exist
            category_folder = os.path.join(input_dir, category)
            os.makedirs(category_folder, exist_ok=True)

            # Move the resume to the respective category folder
            new_path = os.path.join(category_folder, filename)
            os.rename(resume_path, new_path)

    # Write to CSV
    df = pd.DataFrame(categorized_resumes, columns=['filename', 'category'])
    df.to_csv(os.path.join('categorized_resumes.csv'), index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Categorize resumes in a directory.')
    parser.add_argument('input_directory', type=str, help='Path to the directory containing resumes')
    args = parser.parse_args()
    input_directory = args.input_directory

    categorize_resumes_in_directory(input_directory)
