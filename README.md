# Class-project
Chatbot

A chatbot is a computer program that simulates and processes human conversation (either written or spoken), allowing humans to interact with digital devices as if they were communicating with a real person.

In this project I used the retrieval-based model, where chatbot responses are pulled from an existing corpus of dialogs. Machine learning models, such as statistical NLP models and sometimes supervised neural networks, are used to interpret the user input and determine the most fitting response to retrieve.

Libraries required

Needs to be installed and imported : numpy , tkinter , Keras, Tensorflow , NLTK(natural language tool kit)
No need to insatll, only import : pickle, string , random , json

After installing python, go to CMD(command prompt) and pip install (library name)

Steps

1) open and load the data file and analyze

  Intents.json – The data file which has predefined patterns and responses.
  prepprocess the data using tokenizing, a process of spliting a text into smaller words.
  then, lemmatize each word, meaning convert into a dictionary form of the word, and store the python object to a pickle file, which is    used while predicting
  
Words.pkl – This is a pickle file in which we store the words Python object that contains a list of our vocabulary.
Classes.pkl – The classes pickle file contains the list of categories.

2) Create training and Build the model

The training data wiil have the pattern as input and the class our pattern belong to as output and then convert the text into numbers.

Once training data is ready, we will build a deep neural network that has 3 layers using the Keras sequential. Train the model for 200 epochs and save the model as ‘chatbot_model.h5’.

Chatbot_model.h5 – This is the trained model that contains information about the model and has weights of the neurons.

3) Predict the response

Load the 'words.pkl' and 'classes.pkl'
Load the trained model 'chatbot.model and write a function to perform text proccessing and predict the class then retrieve the approperaite response . 


traindata.py – In this Python file, a written script to build the model and train our chatbot.

tk_gui.py – This is the Python script to implement GUI for chatbot.

 To apply the code to another data source, first we need to identify the file format and use the required liberary to open and load it
 usually files used in deep learning have same format 
 
Files included
- data file( intent.json)
- 2 python file (tk_gui , traindata)
- 2 pickle file ( Words , classes )
-  Trained Model ( chatbot_model.h5)
 
 
 Citations
 
 “What Is a Chatbot?” Oracle, https://www.oracle.com/chatbots/what-is-a-chatbot/.
 
 Codecademy. “What Are Chatbots.” Codecademy, Codecademy, https://www.codecademy.com/article/what-are-chatbots. 
 

