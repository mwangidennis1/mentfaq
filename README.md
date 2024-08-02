This project implements a mental health support chatbot designed to provide information and initial guidance on mental health topics. The chatbot uses a retrieval-based approach, leveraging TF-IDF vectorization and cosine similarity to match user queries with the most relevant pre-defined responses.
and also a generative approach using seq2seq encoder-decoder model based on lstm(long short-term memory).
The report mlmentalhealth4.pdf gives a more in-depth look at the project

This chatbot aims to contribute to the United Nations Sustainable Development Goal 3: Good Health and Well-being by increasing accessibility to mental health information 
and reducing barriers to seeking help.

## Features

- Provides immediate responses to mental health-related queries
- Uses natural language processing for improved query understanding
- Offers 24/7 availability for mental health information
- Maintains user anonymity to reduce stigma barriers
## Installation
1. clone the repository
2. make sure you have pipenv installed locally
3. go to the path where you have cloned the project
4. run pipenv shell to activate virtual environment
5. run pipenv install  to install dependancies in the pipfile
6. run pipenv run python main.py to run the retreival-based chatbot
7. run pipenv run python lstm.py to run the generative-based chatbot

## Limitations

- The chatbot is limited to the scope of its training data
- It cannot maintain context over multiple interactions
- This is not a substitute for professional mental health advice or treatment

## Disclaimer

This chatbot is designed for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

