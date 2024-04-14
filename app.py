import streamlit as st
import openai
import os
import random
import string
from PIL import Image

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
from pydub import AudioSegment


# Set up Langmith tracking
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize OpenAI API
openai_api_key = os.getenv("OPEN_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125", temperature=0.5)

# Load environment variables
load_dotenv()

# Generate a random code once when the script runs
random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))



# Function to get OpenAI response
def get_openai_response(feedback):
    chat_messages = [
        SystemMessage(content='As a healthcare professional specializing on pregnancy and psychology, you are tasked with summarizing the pregnancy and medical intervention experiences of patients. Your goal is to provide a concise summary that helps other healthcare providers understand the patients needs, worries, concerns, and any potential red flags.  Please utilize the patients input to create a comprehensive summary. This should include details about the pregnancy journey, any medical interventions undergone, the patients worries or concerns, current threats to health at the time of the feedback. Furthermore consider if the text covers the following list of topics: 1.pain relief/management, 2. birth injuries, 3. anesthesia and side-effects, 4. drugs and medicine, 5. equipments, 6. location and surrounding environment, 7. personnel, 8. cultural biases around pregnancy, 9. fear and worries. If there are mentions that concern a certain point, put a short mention under this topic. The summary should be empathetic, clear, and informative, offering valuable insights to guide the healthcare provider in providing the best possible care for the patient. Thank you for your dedication to patient care. Your efforts in summarizing their experiences will greatly contribute to their well-being.'),
        HumanMessage(content=f'Please provide a short and concise summary of the following feedback, about 250 words total, plus extra lists under titles of mentioned topics if exist. Ensure to mark any unanswered worries with a star (*) to indicate they need attention. Additionally, check for signs of depression or threats to health and add a red flag if detected. Return in a format: 1.Summary. 2.Red flags. 3. Topics with points.  \n TEXT: {feedback}')
    ]
    response = llm(chat_messages).content
    return response


# Function to record audio
def record_audio():
    from audiorecorder import audiorecorder
    audio = audiorecorder("Click to Record", "Click to Stop Recording")
    return audio

# Function to save audio
def save_audio(audio, filename=f"audio_{random_code}.wav"):
    audio.export(filename, format="wav")

# Function to transcribe audio to text
def transcribe_audio(audio_data):
    transcription = openai.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_data,
        response_format="text")
    return transcription

# Function to split audio into parts
def split_audio(audio, segment_length_ms=60000):
    audio_parts = []
    total_duration = len(audio)
    start = 0
    while start < total_duration:
        end = min(start + segment_length_ms, total_duration)
        audio_part = audio[start:end]
        audio_parts.append(audio_part)
        start = end
    return audio_parts

# Function to save transcribed text to a file
def save_transcribed_text(text, filename=f"transcribed_text_{random_code}.txt"):
    with open(filename, "w") as file:
        file.write(text)

# Function to save OpenAI response to a CSV file
def save_openai_response(response, filename=f"openai_response_{random_code}.csv"):
    import csv
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Response"])
        writer.writerow([response])

# Streamlit app initialization
st.set_page_config(
    page_title="Child Birth Feedback",
    page_icon='🎈',
    layout='wide',  # Set layout to wide
    initial_sidebar_state='expanded'  # Keep sidebar expanded
)

# Sidebar

img = Image.open('welcome.png')
st.sidebar.image(img, use_column_width=True)
st.sidebar.write("")
st.sidebar.write("Share your story! The summary will help during your last meeting with the midwife.")
st.sidebar.write("You also support a nation wide research on childbirth service improvement and help all new and future parents in Sweden!")
st.sidebar.write("Thank you!")

selected_option = st.sidebar.selectbox(f"Your important feedback: #{random_code}", ("Record and Summarize", "Summarize from Text", "Summarize from Audio"))

# Option: Record and Summarize
if selected_option == "Record and Summarize":
    audio = record_audio()

    if len(audio) > 0:
        # Display and save audio
        st.sidebar.audio(audio.export().read())  
        save_audio(audio)

        # Transcribe each part of the audio and concatenate the transcribed texts
        audio_parts = split_audio(audio)
        transcribed_texts = []
        total_parts = len(audio_parts)
        for i, audio_part in enumerate(audio_parts, start=1):
            st.write(f"Transcribing Part {i} out of {total_parts}", end="\r")  # Print on the same line
            st.empty()  
            # Export audio part as WAV format
            audio_part.export(f"audio_part_{i}.wav", format="wav")
            
            # Transcribe audio part to text
            transcription = transcribe_audio(open(f"audio_part_{i}.wav", "rb"))
            transcribed_texts.append(transcription)


        # Concatenate transcribed texts into a single text
        feedback_text = " ".join(transcribed_texts)

        # Display transcribed text
        st.write("Transcribed Text:")
        st.write(feedback_text)

        # Save transcribed text to a file
        save_transcribed_text(feedback_text)

        # Get AI response
        response = get_openai_response(feedback_text)
        st.write("AI Response:")
        st.write(response)

        # Save OpenAI response to a CSV file
        save_openai_response(response)

# Option: Summarize from Text File
elif selected_option == "Summarize from Text":
    st.sidebar.subheader("Summarizing from the text file")
    if os.path.exists("transcribed_feedback.txt"):
        st.write("Transcription in proces:")
        with open("transcribed_feedback.txt", "r") as file:
            feedback_text = file.read()
            response = get_openai_response(feedback_text)
            #assessment= check_summary(response)
            st.write("AI Response:")
            st.write(response)
           #st.write(assessment)
            save_openai_response(response)
    else:
        st.sidebar.write("Error: File not found. Please make sure that transcribed_feedback.txt exists.")

# Option: Summarize from Audio
elif selected_option == "Summarize from Audio":
    st.sidebar.subheader("Summarizing from the audio file")
    if os.path.exists("audio_feedback.wav"):
        audio_file_path = "audio_feedback.wav"
        audio = AudioSegment.from_file(audio_file_path)

# Split audio into parts if it's too long
        audio_parts = split_audio(audio)

# Transcribe each part of the audio and concatenate the transcribed texts
        transcribed_texts = []
        total_parts = len(audio_parts)
        for i, audio_part in enumerate(audio_parts, start=1):
            st.write(f"Transcribing Part {i} out of {total_parts}", end="\r")
    
    # Export audio part as WAV format
            audio_part.export(f"audio_part_{i}.wav", format="wav")
    
    # Transcribe audio part to text
            transcription = transcribe_audio(open(f"audio_part_{i}.wav", "rb"))
            transcribed_texts.append(transcription)

    # Save transcribed text to a file
            save_transcribed_text(transcription)

# Concatenate transcribed texts into a single text
        feedback_text = " ".join(transcribed_texts)

# Display transcribed text
        st.write("Transcribed Text:")
        st.write(feedback_text)

# Save transcribed text to a file
        save_transcribed_text(feedback_text)

# Get OpenAI response
        response = get_openai_response(feedback_text)
        st.write("OpenAI Response:")
        st.write(response)

# Save OpenAI response to a CSV file
        save_openai_response(response)

    else:
        st.sidebar.write(f"Error: File not found. Please make sure that audio_feedback.wav exists.")