import os
import openai
import streamlit as st
from retry import retry
from elevenlabs import voices, generate, save, set_api_key
import tiktoken

#######
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
#######
from thispersondoesnotexist import get_online_person, save_picture


st.sidebar.title("Assitant Factory")
st.sidebar.image("a_beautiful_person.jpeg", width=200)
st.sidebar.write("Image taken from [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/)")

st.sidebar.write("Commands:")
st.sidebar.write(":orange[/newpic] : generate a new picture for your assistant")
st.sidebar.write(":orange[/instructions] : rewrite the instructions of your assistant")
st.sidebar.write(":orange[/voice] : choose the characteristics of your assistant's voice")
st.sidebar.write("When you are ready save your Assistant")
st.sidebar.button("Save")

from audiorecorder import audiorecorder

audio = audiorecorder("Click to record", "Recording... Click when you're done", key="recorder")

set_api_key(os.getenv("ELEVENLABS_API_KEY"))

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENROUTER_REFERRER = "https://github.com/alonsosilvaallende/langchain-streamlit"

with open("instructions.txt", "r") as f:
    instructions = f.read().strip()

with open("voice.txt", "r") as f:
    voice = f.read().strip()

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="google/palm-2-chat-bison",
                 streaming=True,
                 temperature=2,
                 headers={"HTTP-Referer": OPENROUTER_REFERRER})

def template(instructions=instructions):
    template = instructions.replace("/instructions", "")
    return template

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from whispercpp import Whisper

@st.cache_resource
def do_nothing():
    return Whisper('tiny')
w = do_nothing()

import numpy as np

def inference(audio):
    # Save audio to a file:
    wav_file = open("audio.mp3", "wb")
    wav_file.write(audio.tobytes())
    result = w.transcribe("audio.mp3")
    text = w.extract_text(result)
    return text[0]

people = {
'1': 'Adam',
'2': 'Antoni',
'3': 'Arnold',
'4': 'Bella',
'5': 'Callum',
'6': 'Charlie',
'7': 'Charlotte',
'8': 'Clyde',
'9': 'Daniel',
'10': 'Dave',
'11': 'Domi',
'12': 'Dorothy',
'13': 'Elli',
'14': 'Emily',
'15': 'Fin',
'16': 'Freya',
'17': 'Gigi',
'18': 'Giovanni',
'19': 'Harry',
'20': 'James',
'21': 'Jeremy',
'22': 'Jessie',
'23': 'Joseph',
'24': 'Josh',
'25': 'Liam',
'26': 'Matilda',
'27': 'Matthew',
'28': 'Michael',
'29': 'Mimi',
'30': 'Nicole',
'31': 'Patrick',
'32': 'Rachel',
'33': 'Ryan',
'34': 'Sam',
'35': 'Serena',
'36': 'Thomas',
'37': 'None'
}

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
possible_tokens = {f"{enc.encode(f'{i}')[0]}": 100 for i in range(1,38)}

@retry(tries=10, delay=3)
def my_classifier(prompt):
    response = openai.ChatCompletion.create(
    model='openai/gpt-3.5-turbo',
    headers={"HTTP-Referer": OPENROUTER_REFERRER},
    messages=[{
        'role': 'user',
        'content': """According to the following TEXT, the user wants a voice that is
1: Adam, american, deep, narration
2: Antoni, american, well-rounded, narration
3: Arnold, american, crisp, narration
4: Bella, american, soft, narration
5: Callum, american, hoarse, video games
6: Charlie, australian, casual, conversational
7: Charlotte, english-sweden, seductive, video games
8: Clyde, american, war veteran, video games
9: Daniel, british, deep, news presenter
10: Dave, british-essex, conversational, video games
11: Domi, american, strong, narration
12: Dorothy, british, pleasant, children's stories
13: Elli, american, emotional, narration
14: Emily, american, calm, meditation
15: Fin, irish, sailor, video games
16: Freya, american, overhyped, video games
17: Gigi, american, childish, animation
18: Giovanni, english-italian, foreigner, audiobook
19: Harry, american, anxious, video games
20: James, australian, calm, news
21: Jeremy, american-irish, excited, narration
22: Jessie, american, raspy, video games
23: Joseph, british, ground reporter, news
24: Josh, american, deep, narration
25: Liam, american, neutral, narration
26: Matilda, american, warm, audiobook
27: Matthew, british, calm, audiobook
28: Michael, american, orotund, audiobook
29: Mimi, english-swedish, childish, animation
30: Nicole, american, whisper, audiobook
31: Patrick, american, shouty, video games
32: Rachel, american, calm, narration
33: Ryan, american, soldier, audiobook
34: Sam, american, raspy, narration
35: Serena, american, pleasant, interactive
36: Thomas, american, calm, meditation
37: None of the above meets the criteria
Answer only the number.
TEXT:""" + prompt
    }],
    logit_bias=possible_tokens,
    max_tokens=1,
    temperature=0).choices[0].message.content
    return f"{people[response]}"

@retry(tries=10, delay=1, backoff=2, max_delay=4)
def llm1(text: str, instructions: str) -> str:
    response = openai.ChatCompletion.create(
        model='google/palm-2-chat-bison',
        headers={
            "HTTP-Referer": OPENROUTER_REFERRER
        },
        messages=[
            {
                "role": "system",
                "content": f"{instructions}"
            },
            {
                'role': 'user',
                'content': f'{text}'
            }
        ],
        temperature=0).choices[0].message.content
    return response, instructions

if (prompt := st.chat_input("Your message")) or (len(audio)>0):
    if len(audio)>0:
        prompt = inference(audio)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        if "/voice" in prompt:
            voice_aux = my_classifier(prompt)
            if voice_aux == "None":
                full_response = "None of our current voices matches that criteria. Please, try again"
            else:
                voice = voice_aux
                with open("voice.txt","w") as f:
                    f.write(voice_aux)
                full_response = "Is this OK for you?"
        elif "/instructions" in prompt:
            instructions = prompt.replace("/instructions", "")
            with open("instructions.txt", "w") as f:
                f.write(instructions)
            full_response = "New instructions have been copied to memory"
        elif "/newpic" in prompt:
            picture = get_online_person()
            save_picture(picture, "a_beautiful_person.jpeg")
            st.experimental_rerun()
            full_response = "Is this picture OK for you?"
        else:
            full_response, instructions = llm1(prompt, instructions)
        message_placeholder.markdown(full_response)
        audio = generate(text=f"{full_response}", voice=voice, model="eleven_monolingual_v1")
        save(audio, "./hola.mp3")
        audio_file = open('./hola.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mpeg')
    st.session_state.messages.append({"role": "assistant", "content": full_response})

style_stuff = f"""
<style>
    *, html {{
      scroll-behavior: smooth !important;
    }}
    recorder {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(style_stuff, unsafe_allow_html=True)
