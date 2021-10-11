# this file is to create the chatbot application
# we are going to use the trained model
# we will also deploy the model on Telegram

from lib2to3.fixes.fix_input import context

# we will import the same libararies
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# we need to import the method to load our model
from tensorflow.keras.models import load_model

# we import telegram to deploy the bot on Telegram
import telegram.ext

# we connect to Telegram using unique token that we saved in token.txt file
with open('token.txt', 'r') as f:
    TOKEN = f.read()

# we use start function for starting the conversation with our bot on Telegram
# users get this message when the start with the bot
def start(update, context):
    update.message.reply_text('Hello! Welcome to 5per5 Bot! Type something!')

# now we will go back to the content piece and run the same lemmatizer methods as on training sentence_words
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# we will load our saved words, classes and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# next we are going to define 4 functions
# the model is already trained and we can use it
# the model output is currently in numeric data, but we need words as actual outputs

# first function is to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# second function will convert a sentence into a bag of words
# bag of words is a list of 0 and 1 that indicate if the word is there or not
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

# third function is for predicting the class
# we will predict the result based on the bag of words (using 2 previous functions)
# we will set error threshold at 25% (if predicted result is below 25% we will discard it)
# we will sort the results by probability, placing the highest probability first
# we will get the return list based on intents, classes and probabilities
def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# fourth function is to get the actual response
def get_response(intents_list, intents_json):
    tag=intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

# this function handles messages and delivers users the reply on Telegram
# in this function we unite the trained model and user's input
# we get user's message, predict the class and the intent
# lastly we use "get_response" function to predict the response and reply to the user
def handle_message(update, context):
    message = str(update.message.text).lower()
    ints = predict_class(message)
    response = get_response(ints, intents)
    update.message.reply_text(response)

# now we will create the main function to connect the bot to Telegram
def main():
    updater = telegram.ext.Updater(TOKEN, use_context=True)
    disp = updater.dispatcher

    # add commands that we specified before
    disp.add_handler(telegram.ext.CommandHandler("start", start))
    disp.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handle_message))

    # updater starts the program
    # you can add seconds in (), i.e. (5), for immediate reply leave the brackets empty
    updater.start_polling()
    updater.idle()

main()


# Use the code below to test the bot inside PyCharm

# print("Go! Bot is running!")
#
# while True:
#     message= input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)