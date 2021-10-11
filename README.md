# bot_test
Telegram bot using python

The goal of 5per5 bot is to help non-native English speakers to learn trendy words to use at job interviews. Currently the bot is still in making and learning.
I wanted to share this template script in case it might be helpful to others.

The goal of this script is to create a simple bot with ML features on Telegram using python. 
I am using tensorflow.keras to train the bot to recognize patterns of natural languge and predict the appropriate response.
The code has comments to explain what each part does.

If you want to use this script, you will need to copy 3 files from this repository:

-intents.json
-training.py
-chatbot_5per5_main.py

-you will generate your own model.h5 once you run your training script.

You will also need to create your own txt file named "token.txt" where you will paste the unique token you get from botfather.
Here is the tutorial I used to create my bot on telegram https://www.youtube.com/watch?v=CNkiPN_WZfA. This tutorial walks you through token creation.

Here are 2 other tutorials that helped me along the way:

https://www.youtube.com/watch?v=PTAkiukJK7E

https://www.youtube.com/watch?v=1lwddP0KUEg&t=337s
