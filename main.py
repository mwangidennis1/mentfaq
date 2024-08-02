import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import random
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('model\\Mental_Health_FAQ.csv')

# Preprocess function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into string
    return ' '.join(tokens)

# Apply preprocessing to questions and answers
df['Processed_Questions'] = df['Questions'].apply(preprocess_text)
df['Processed_Answers'] = df['Answers'].apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Processed_Questions'])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(question_vectors)

# Create a dictionary to store the model data
model_data = {
    'vectorizer': vectorizer,
    'question_vectors': question_vectors,
    'similarity_matrix': similarity_matrix,
    'df': df
}

# Save the model and necessary data
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model trained and saved successfully.")
intents = json.loads(open('intents.json').read())
#print(intents)
class ChatBot:
    def __init__(self,model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data['vectorizer']
        self.question_vectors = model_data['question_vectors']
        self.similarity_matrix = model_data['similarity_matrix']
        self.df = model_data['df']
    def get_response(self,user_input):
        user_input_vector = self.vectorizer.transform([user_input]).toarray()
        
        # Compute the similarity between the user input and the stored questions
        similarity_scores = cosine_similarity(self.question_vectors, user_input_vector)

        
        # Find the index of the most similar question
        most_similar_index = np.argmax(similarity_scores)
        
        # Get the corresponding answer
        response = self.df.iloc[most_similar_index]['Answers']
        
        return response
def noooticing(userInput):
    for intent in intents['intents']:
        if userInput.lower() == intent['tag'].lower():
            return intent['responses'][0]
    return 'not found'
        
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    print("get_bot_response:- " + user_input)
    noooticer=noooticing(user_input)
    if noooticer != 'not found':
        return   noooticer

    model_path='C:\\Users\\Lenovo\\Desktop\\mentalbot\\chatbot_model.pkl'
    chat_bot=ChatBot(model_path)
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
    if(user_input in negative_responses or user_input in exit_commands):
        return "Goodbye"
    chat_response=chat_bot.get_response(user_input)
    return chat_response

if __name__=="__main__":
    app.run()  
