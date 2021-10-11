# This code is for training our chatbot

# import all the libraries we will need
# random is used for choosing the random response at the end
# we use pickle for serialization
import random
import json
import pickle
import numpy as np

# nltk = natural language kit, we use lemmatizer to reduce the word to its stem
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# we import tensorflow libraries for NLP for this script
# SGD stands for stochastic gradient descent
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# we are going to lemmatize individual words
lemmatizer = WordNetLemmatizer()

# we load the json files "intents"
intents = json.loads(open('intents.json').read())

# we are going to create 3 empty lists for words, classes and documents
# "documents" are the combinations (what word belongs to what class)
words = []
classes = []
documents = []

# we will create a list of letters we will ignore
ignore_letters = ["?", "!", ",", "."]

# in our function we are going to iterate through intents and patterns
# we are going to tokenize the patterns (=split patterns into individual words)
# then we are going to add these words to the word list
# we use "extend" for this, it takes the content and appends it to the list
# note: append takes the list and appends it to the list
# we are also going to append to the documents list the lists of words and the classes of each intent
# lastly we will append it to the classes list if it is not yet there
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# we can use print(documents) to check if it works and how it looks
# print(documents)

# now we will do the lemmatizing
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# we use "set" to eliminate word duplicates and "sort" to turn it back into a list
words = sorted(set(words))
# we shouldn't have any duplicates of classes, however we will include this line of code to make the code more resilient
classes = sorted(set(classes))

# we can use print(words) to get a list of lemmatized words
# print(words)

# we are going to save the words and classes for later on
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# now we get into ML part
# are going to use the bag of words to set individual word indices to 0 and 1

# we start by setting empty training list and empty output list the length of classes
training = []
output_empty = [0] * len(classes)

# for each of those word combinations we are going to create a bag of words
# we are going to lemmatize each word
# we are going to add all unique words to the bag of words
# lastly we will append all the bag of words and related classes to the training list

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag, output_row])

# we are going to shuffle the training data and turn it into np array
random.shuffle(training)
training = np.array(training)

# are going to split all the data into X and Y values (features and labels) for training NN
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# building the NN model
# we are going to add a few layers to the models: input layer will be a dense layer
# activation function = relu (rectified linear unit)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# we add dropout in order to prevent over-fitting
# we add another dense and dropouts layers
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# we add activation softmax to sum up the results (so that they add up to 1, and we get %)
model.add(Dense(len(train_y[0]), activation='softmax'))

# we will define stochastic gradient descent layer
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# we will compile the mode; with categorical loss function
# our main metric is accuracy
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# we are going to run the model with the parameters specificed above
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# we will save the model as h5
model.save('chatbotmodel.h5', hist)
print("Done")