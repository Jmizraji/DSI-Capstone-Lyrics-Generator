import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib

#Start the program
run_program = ''

#Create lists to store text
human_generated_text = []
ai_generated_text = []
song_lyrics = []

#Create a loop to allow the user to start and stop the program
while run_program != 'STOP':

    #Get User input for the first few line of a lyric
    user_input = input("Type your lyrics: ").strip()

    #start the session
    sess = gpt2.start_tf_sess()

    #load the pre-trained model
    gpt2.load_gpt2(sess, run_name='run1')

    #save text generated as a list
    text = gpt2.generate(sess,
                  return_as_list=True,
                  length=3, #number of tokens to generate
                  temperature=0.8,
                  prefix=user_input, #start of the text
                  nsamples=9,
                  batch_size=9,
                  include_prefix=True,
                  truncate='\n'
                  )

    #store song lyrics
    song_lyrics += [user_input]

    #store human generated text in an array
    human_generated_text += [user_input]

    #create an array of all of the outputted phrases
    ai_generated_text += [phrase.replace(user_input + ' ', '') for phrase in text]

    # print('\nSuggested Lyrics\n' + ('--'*15))
    # print(text)
    print('\nAI Generated Text\n' + ('--'*15))
    print(ai_generated_text)
    print('\nHuman Generated Text\n' + ('--'*15))
    print(human_generated_text)

    #Let the user select a suggestion
    print(f"\nCurrent Lyrics:\n{' '.join(song_lyrics)}...")

    print(f"\nType a number if you would like to select a suggestion \nor press ENTER if you would ike to write your own.\n")

    #Loop through and print the suggestions
    for idx,phrase in enumerate(text):
        print(f"{idx}: {phrase}")

    #collect user response
    response = input('Response: ')

    #if user wants to accept suggestion
    try:
        if isinstance(int(response), int):

            #append to song lyrics
            song_lyrics += [(text[int(response)].replace(user_input + ' ', '')).strip()]
        else:
            user_generated_lyrics = input('Type your lyrics: ' + user_input + ' ').strip()

            #append to song lyrics
            song_lyrics += [user_generated_lyrics]

            #append to human generated text
            human_generated_text += [user_generated_lyrics]

    except ValueError:
        user_generated_lyrics = input('Type your lyrics: ' + user_input + ' ').strip()

        #append to song lyrics
        song_lyrics += [user_generated_lyrics]

        #append to human generated text
        human_generated_text += [user_generated_lyrics]


    #print out the current song lyrics
    print('\nSong Lyrics\n' + ('--'*15))
    print(' '.join(song_lyrics))
    print()
    print(song_lyrics)

    #restart the session
    gpt2.reset_session(sess)

    #either end or continue the loop
    run_program = input("Press ENTER to continue writing lyrics. Type STOP to exit. ")

#USER HAS ENDED THE PROGRAM
#Find out how much is ai generated vs human generated
def human_pct():
    #find % of human generated text in song lyrics
    pct_similar = set(song_lyrics).intersection(human_generated_text)

    return round(len(pct_similar) / len(song_lyrics),2)

def ai_pct():
    #find % of ai generated text in song lyrics
    pct_similar = set(song_lyrics).intersection(ai_generated_text)

    return round(len(pct_similar) / len(song_lyrics),2)

#Final results
print('\nFinal Song Lyrics\n' + ('--'*15))
print(' '.join(song_lyrics))
print()

print('\nAI Generated Text\n' + ('--'*15))
print(ai_generated_text)
print('\nHuman Generated Text\n' + ('--'*15))
print(human_generated_text)
print()

print(f"Percentage of Human Generated Text is: {human_pct() * 100}")
print(f"Percentage of AI Generated Text is: {ai_pct() * 100}")
