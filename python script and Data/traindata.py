
                                                      # import the neccessary liberaries
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

                                                  # create empty list to store words and catagories in classes
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)                 # open the json file with predefined pattern and responses


for intent in intents['intents']:               # loop through intent file and nested loop to get the pattens
    for pattern in intent['patterns']:

       
        w = nltk.word_tokenize(pattern).        # Tokenizers divide strings into lists of substrings
        words.extend(w)                             
        
        documents.append((w, intent['tag']))    # add them to their corresponding empty list created

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

                                                # lemmaztize and lower each word and remove duplicates                                                  
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))
                                                # documents are combination of patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))         # The dump() method converts a Python object into a byte stream and save to a disk
pickle.dump(classes,open('classes.pkl','wb'))

                                                  # training data and an empty array for the output created
training = []

output_empty = [0] * len(classes)

for doc in documents:
   
    bag = []
                                                            # list of tokenized words for the pattern
    pattern_words = doc[0]
                                                            # lemmatize each word - to return the base or dictionary form of the words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
                                                            # if word match found in current pattern, append 1 other wise 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
   
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
                                                                   
random.shuffle(training)
training = np.array(training)
                                                                    # create train and test lists with X - patterns and Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


                        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
                        # equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

                          # Compile model. Stochastic gradient descent with Nesterov accelerated gradient 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

                            #fit and save the model as chatbot_model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
