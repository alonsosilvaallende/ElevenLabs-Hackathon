import os
import openai
import streamlit as st
from retry import retry
from elevenlabs import voices, generate, save, set_api_key

#######
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
#######

st.sidebar.title("Assitant Store Factory")

st.sidebar.write("Commands:")
st.sidebar.write("/voice : choose the characteristics of your assistant's voice")
st.sidebar.write("/instructions : rewrite the instructions of your assistant")
st.sidebar.write("When you are ready save your Assistant")
st.sidebar.button("Save")

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

# Accept user input or example
# import whisper

# from audiorecorder import audiorecorder

#@st.cache_resource
#def do_nothing():
#    model = whisper.load_model("tiny")
#    return model
#model = do_nothing()
#audio = audiorecorder("Click to record", "Recording... Click when you're done")

def inference(audio):
    # To save audio to a file:
    wav_file = open("audio.mp3", "wb")
    wav_file.write(audio.tobytes())
    audio = whisper.load_audio("audio.mp3")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    input_language = max(probs, key=probs.get)
    result = whisper.transcribe(audio=audio, model=model,language=input_language, fp16=True, verbose=False)
    return result['text']


def my_classifier(prompt, default=voice):
    prediction = llm.predict("""\
According to the following TEXT, the user wants a voice that is
1: male and american
2: female and american
3: male and british
4: female and british
5: the user didn't specify which voice he wants
Answer only the number: 1, 2, 3, 4, or 5.
TEXT:"""+prompt)
    if '1' in prediction:
        voice = "Adam"
    elif '2' in prediction:
        voice = "Bella"
    elif '3' in prediction:
        voice = "Daniel"
    elif '4' in prediction:
        voice = "Dorothy"
    else:
        voice = default
    return voice

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

if (prompt := st.chat_input("Your message")):
#if len(audio)>0:
#    prompt = inference(audio)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        if "/voice" in prompt:
            voice = my_classifier(prompt)
            with open("voice.txt","w") as f:
                f.write(voice)
            full_response = "Is this OK for you?"
        elif "/instructions" in prompt:
            instructions = prompt.replace("/instructions", "")
            with open("instructions.txt", "w") as f:
                f.write(instructions)
            full_response = "I have copied your new instructions to memory"
        else:
            full_response, instructions = llm1(prompt, instructions)
        message_placeholder.markdown(full_response)
        audio = generate(text=f"{full_response}", voice=voice, model="eleven_monolingual_v1")
        save(audio, "./hola.mp3")
        audio_file = open('./hola.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mpeg')
    st.session_state.messages.append({"role": "assistant", "content": full_response})
