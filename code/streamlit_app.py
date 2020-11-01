import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
import streamlit as st
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

#Plotting styling  https://medium.com/@andykashyap/top-5-tricks-to-make-plots-look-better-9f6e687c1e08
style.use('seaborn-poster') #sets the size of the charts
style.use('seaborn-pastel')

#Storing the outputs
@st.cache(allow_output_mutation=True)
def get_array():
    array = []
    return array

#Create variables
ai_generated_text = get_array()
human_generated_text = get_array()
song_lyrics = get_array()

@st.cache(allow_output_mutation=True)
def get_ai_pct():
    ai_generated_ct = 0

    for phrase in set(filter(None,ai_generated_text)):
        if phrase in lyrics_box:
            ai_generated_ct += len(phrase.split(' '))

    return round(ai_generated_ct / len(lyrics_box.split(' ')), 3)


# Streamlit Code
################################
# Sidebar

#widgets
tokens = st.sidebar.slider(label='Number of Tokens', min_value=1, max_value=15,value=3, step=1)
samples = st.sidebar.slider(label='Number of Samples', min_value=1, max_value=9,value=9, step=1)
temp = st.sidebar.slider(label='Temperature', min_value=0.1, max_value=1.0,value=0.8, step=0.05)
topk = st.sidebar.slider(label='Top k', min_value=0, max_value=40,value=0, step=1)

#Help text
st.sidebar.markdown(
'''
`Number of Tokens:` number of tokens in generated text\n
`Number of Samples:` number of samples to return total\n
`Temperature:` Float value controlling randomness in boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions.\n
`Top k:` Integer value controlling diversity. 1 means only 1 word is considered for each step (token), resulting in deterministic completions, while 40 means 40 words are considered at each step. 0 (default) is a special setting meaning no restrictions. 40 generally is a good value.
''')

#title text
st.title("Song Lyrics Generator Using GPT-2")
st.markdown(
'''
This is a demo of a text generation model trained with GPT-2 to generate song lyrics that sound like John Mayer, Jack Johnson, and Shawn Mendes.

*For additional questions and inquiries, please contact Josh Mizraji via [LinkedIn](https://www.linkedin.com/in/joshuamizraji/) or [Github](https://github.com/Jmizraji/DSI-Capstone-Lyrics-Generator).*
''')

#Section to select artist model to use
st.subheader('Select an artist to write like:')

#artist dict
artist_dict = {
    "John Mayer" : "JohnMayer",
    "Jack Johnson" : "JackJohnson",
    "Shawn Mendes" : "ShawnMendes"
 }

#display artists model options
artist_selection = st.radio(label='',options=[key for key in artist_dict.keys()])

#text box
lyrics_box = st.text_area('Start writing lyrics...')
st.warning('Remember to press command + ENTER to save text before generating lyrics')


if st.button('Generate Lyrics'):

#if a valid input is typed in, run the model
    if (lyrics_box.strip() != ''):
        st.write('Generating lyrics...')

        #start the session
        sess = gpt2.start_tf_sess()

        #load the pre-trained model
        gpt2.load_gpt2(sess, run_name=artist_dict[artist_selection])

        #save text generated as a list
        text = gpt2.generate(sess,
                      return_as_list=True,
                      length=tokens, #number of tokens to generate
                      temperature=temp,
                      prefix=lyrics_box.strip(), #start of the text
                      nsamples=samples,
                      batch_size=samples,
                      include_prefix=False,
                      top_k = topk,
                      truncate='\n'
                      )

        #display generated text
        st.write([(phrase.replace(lyrics_box.strip(), '')).strip() for phrase in text])

        #create an array of all of the outputted phrases
        ai_generated_text += [(phrase.replace(lyrics_box.strip(), '')).strip() for phrase in text]

        #display human vs. ai %
        st.write('Human Generated Text: ',round(1.0 - get_ai_pct(), 3))
        st.write('AI Generated Text: ', get_ai_pct())

        #Plot Chart of % Human vs % AI
        #Create DataFrame
        chart_data = pd.DataFrame({
            'Human': [round(1.0 - get_ai_pct(), 3)],
            'AI': [get_ai_pct()]
        })

        #Plot Chart
        fig, ax = plt.subplots()
        chart_data.plot(kind='barh',stacked=True, ax=ax)
        ax.set_xticks(np.arange(0,1.1,.1))
        ax.yaxis.set_visible(False)
        ax.set_title('Percentage of Human Generated Text vs AI Generated Text')
        plt.show()
        st.pyplot(fig)




        #restart the session
        gpt2.reset_session(sess)
