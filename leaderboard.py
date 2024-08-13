# pip install streamlit datasets huggingface-hub ulid-py bittensor replicate google-cloud-storage ffmpeg-python

import asyncio
import os
import time
import json
import requests
import random
import tempfile

import streamlit as st
import pandas as pd
import threading
from urllib.parse import urlparse
import ffmpeg
#from moviepy.editor import VideoFileClip, concatenate_videoclips

from neurons.mm_model_scoring import get_caption_from_model, get_mm_response
from tune_recipes.imagebind_api import embed_modality

JSON_FILE = "leaderboard.json"
CACHE_FILE = "omega_dataset_examples.json"

class CountingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0
        self._count_lock = threading.Lock()

    def acquire(self):
        with self._count_lock:
            self._count += 1
        self._lock.acquire()
        with self._count_lock:
            self._count -= 1

    def release(self):
        self._lock.release()

    def waiting_threads(self):
        with self._count_lock:
            return self._count
        
    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

# Singleton instance
_mutex_instance = None

#@st.cache_resource
#def get_mutex():
    #global _mutex_instance
    #if _mutex_instance is None:
       # _mutex_instance = CountingLock()
    #return _mutex_instance

@st.cache_resource
def get_mutex():
    return threading.Lock()

def load_models():
    # Load models from JSON file
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
            models = {model['name']: model for model in data}
    else:
        models = {}

    # Sort models by rank (lowest to highest)
    models_to_sort = models.values()
    sorted_models = sorted(models_to_sort, key=lambda x: x.get('rank', float('inf')))
    sorted_models = {model['name']: model for model in sorted_models}

    return sorted_models

@st.cache_data(ttl=1800) # Cache for 30 minutes
def load_video_metadata():
    # Load video metadata from JSON file
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            video_metadata = json.load(f)
    else:
        video_metadata = []

    return video_metadata

# Helper function to convert seconds to mm:ss format
def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

#def stitch_videos(video_files):
#    clips = [VideoFileClip(file.name) for file in video_files]
#    final_clip = concatenate_videoclips(clips)
#    
#    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
#        final_clip.write_videofile(tmpfile.name, codec="libx264")
#        return tmpfile.name

from google.cloud import storage
from google.cloud.exceptions import NotFound
GCS_BUCKET_NAME = "omega-a2a-mm-chat"
def gcs_upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket if it doesn't exist, otherwise returns the existing URL."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    # Check if the blob already exists
    try:
        blob.reload()
    except NotFound:
        # If we reach here, the blob doesn't exist, so we upload it
        blob.upload_from_filename(source_file_name)

    # Generate the public URL
    url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{destination_blob_name}"
    return url

def get_file_type_from_url(url):
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        return 'image'
    elif extension in ['.mp3', '.wav', '.ogg', '.flac']:
        return 'audio'
    elif extension in ['.mp4', '.avi', '.mov', '.wmv']:
        return 'video'
    else:
        return 'unknown'

def process_file(uploaded_file, process_audio=True):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        url = gcs_upload_blob(tmp_file_path, uploaded_file.name)

        if url:
            try:
                embed_type = get_file_type_from_url(url)
                embedding = embed_modality(url, embed_type)
                
                if embedding is not None:
                    st.session_state.embeddings.append({embed_type: embedding})
                    
                    # Check if the file is a video and we want to process audio from the video
                    if embed_type == 'video' and process_audio:
                        # Check for audio stream
                        probe = ffmpeg.probe(tmp_file_path)
                        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
                        
                        if audio_stream:
                            # Extract audio
                            audio_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
                            (
                                ffmpeg
                                .input(tmp_file_path)
                                .output(audio_output, acodec='libmp3lame', ac=2, ar='44100')
                                .overwrite_output()
                                .run(capture_stdout=True, capture_stderr=True)
                            )
                            
                            # Upload extracted audio
                            audio_url = gcs_upload_blob(audio_output, f"{os.path.splitext(uploaded_file.name)[0]}.mp3")
                            
                            # Generate audio embedding
                            audio_embedding = embed_modality(audio_url, 'audio')
                            
                            if audio_embedding is not None:
                                st.session_state.embeddings.append({'audio': audio_embedding})
                                st.session_state.processed_files.append({
                                    'name': f"{os.path.splitext(uploaded_file.name)[0]}.mp3",
                                    'type': 'audio',
                                    'url': audio_url
                                })
                            
                            # Clean up
                            os.unlink(audio_output)
                    
                    st.session_state.processed_files.append({
                        'name': uploaded_file.name,
                        'type': embed_type,
                        'url': url
                    })
                    return f"Successfully processed {uploaded_file.name}"
                else:
                    return f"Issue processing {uploaded_file.name}"
            except Exception as e:
                return f"Error processing {uploaded_file.name}: {str(e)}"
    finally:
        os.unlink(tmp_file_path)

    return f"Failed to process {uploaded_file.name}"


async def main():
    st.set_page_config(
        layout="wide",
        page_title="OMEGA Any-to-Any Leaderboard",
        page_icon="omega_favico.png"
    )
    mutex = get_mutex()
    models = load_models()
    video_metadata = load_video_metadata()
    filtered_metadata = [
        {
            "youtube_id": entry["youtube_id"],
            "description": entry["description"],
            "views": entry["views"]
        }
        for entry in video_metadata
    ]
    
    # Custom CSS for centering the title and table data
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://storage.googleapis.com/omega-a2a-mm-chat/omega-background.png');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            background-color: rgba(0, 0, 0, 0.5);  /* Black background with 50% opacity */
        }
                
        section.main > div {max-width:1400px;}

        .centered-title {
            text-align: center;
        }
        .centered-data td {
            text-align: center;
        }
        .logo {
            display: block; /* Use block to apply margin auto for centering */
            width: 75px; /* Set the width of the logo container */
            height: 75px; /* Set the height of the logo container */
            margin: 0 auto; /* Center the logo horizontally */
            margin-top: 0rem; /* Add space above the logo */
        }

        .logo svg {
            width: 100%; /* Make the SVG fill the container */
            height: 100%; /* Make the SVG fill the container */
        }
                
        .intro-text {
            margin-top: 1rem;
            margin-bottom: 1rem;
            font-size: 18px;
        }
                
        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Roboto', sans-serif;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        button {
            background-color: #068AC7;
            color: white;
            border: none;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        button:hover {
            background-color: #005f8a;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add the SVG logo above the title
    st.markdown("""
        <div class="logo">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 75 75">
                <!-- Define the drop shadow filter -->
                <defs>
                    <filter id="text-shadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
                        <feOffset in="blur" dx="2" dy="2" result="offsetBlur"/>
                        <feMerge>
                            <feMergeNode in="offsetBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                <text x="50%" y="70%" dominant-baseline="middle" text-anchor="middle" font-family="Roboto" font-size="100" fill="#068AC7" filter="url(#text-shadow)">Ω</text>
            </svg>
        </div>
    """, unsafe_allow_html=True)

    # Center the title
    st.markdown('<h1 class="centered-title">OMEGA Any-to-Any Demo</h1>', unsafe_allow_html=True)

    if 'show_intro' not in st.session_state:
        st.session_state.show_intro = False
    def toggle_intro():
        st.session_state.show_intro = not st.session_state.show_intro

    # Toggle button with icon
    col1, col2 = st.columns([0.05, 0.95])
    with col1:
        st.button("Ω", on_click=toggle_intro, help="Toggle Introduction")

    # Intro text (hidden by default)
    with col2:
        if st.session_state.show_intro:
            st.markdown('<p class="intro-text">Welcome to OMEGA Labs\' Any-to-Any multi-modal chat, model demos, and leaderboard. This streamlit showcases a multi-modal chat and video captioning capabilities from the latest models on Bittensor\'s subnet 21.<br /><strong>*Please note most models are undertrained right now (Q2 2024) given the early days of the subnet.</strong></p>', unsafe_allow_html=True)
            st.markdown('<p class="intro-text">On the "MM Chat" tab, upload video, audio, and/or image files and chat with our model to demonstrate how it understands multiple modalities.', unsafe_allow_html=True)
            st.markdown('<p class="intro-text">On the "Model Demos" tab, select a miner\'s model from the dropdown and then browse recent video submissions from subnet 24. Interact with the model by pressing the "Generate Caption for Video" button.</p>', unsafe_allow_html=True)
            st.markdown('<p class="intro-text">On the "Leaderboard" tab, checkout the latest rankings.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Multi-Modal Chat", "Model Demos", "Leaderboard"])

    # Main column for chat
    with tab1:
        st.title("Ω OMEGA Multi-Modal Chat")

        tcol1, tcol2 = st.columns([0.4, 0.6])
        with tcol1:
            model_names = list(models.keys())[:5]
            model_names = ["omega-anytoany-test"] + model_names

            # Define the partial model name you're looking for
            partial_model_name = "omega"

            # Find the index of the first model name that contains the partial name
            model_index = None
            for i, name in enumerate(model_names):
                if partial_model_name in name:
                    model_index = i
                    break

            selected_model = st.selectbox(
                "Select a model", 
                model_names,
                index=model_index,
                placeholder="- Select a model -"
            )

        with tcol2:
            if selected_model and selected_model != "- Select a model -":
                if selected_model == "omega-anytoany-test":
                    model_path = "briggers/omega_a2a_test4"
                    model_path_display = "n/a"
                    model_incentive = "n/a"
                    model_rank = "n/a"
                else:
                    model_path = models[selected_model]['model_path']
                    model_path_display = model_path
                    model_incentive = models[selected_model]['incentive']
                    model_rank = models[selected_model]['rank']

                st.markdown(f"""
                <p>
                <strong>Model Path:</strong> <a href="http://huggingface.co/{model_path_display}">{model_path_display}</a><br>
                <strong>Incentive:</strong> {model_incentive}<br>
                <strong>Rank:</strong> {model_rank}
                </p>
                """, unsafe_allow_html=True)

        if "user_prompt_history" not in st.session_state:
            st.session_state["user_prompt_history"] = []
        if "chat_answers_history" not in st.session_state:
            st.session_state["chat_answers_history"] = []
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "embeddings" not in st.session_state:
            st.session_state["embeddings"] = []
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []

        tcol1, tcol2 = st.columns([0.7, 0.3])

        with tcol1:
            # Create a container for chat history
            chat_container = st.container()

            # Create a container for file upload and user input
            input_container = st.container()
        with tcol2:
            # Create a container for processed files
            files_container = st.container()
            with files_container:
                # Display processed files
                if st.session_state.processed_files:
                    #st.divider()
                    #st.subheader("Processed Files")
                    for pf in st.session_state.processed_files:
                        file_name = pf['name']
                        file_type = pf['type']
                        url = pf['url']
                        if file_type == 'image':
                            st.image(url, caption=file_name)
                        elif file_type == 'video':
                            st.video(url)
                            st.text(f"{file_name}")
                        elif file_type == 'audio':
                            st.audio(url)
                            st.text(f"{file_name}")
        
        # Inside the input container, add file uploader and chat input
        with input_container:
            st.divider()
            uploaded_file = st.file_uploader("Upload audio, video, or image files", type=["mp4", "mov", "avi", "mp3", "wav", "png", "jpg", "jpeg"], accept_multiple_files=False)

            # Add checkbox for audio processing
            process_audio = st.checkbox("Process audio if present in video file", value=True)

            # Process uploaded files
            if uploaded_file:
                #for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if not any(file_info['name'] == uploaded_file.name for file_info in st.session_state.processed_files):
                        result = process_file(uploaded_file, process_audio)
                        st.write(result)
                        st.rerun()

            prompt = st.chat_input("Enter your questions here", disabled=not input)
            if prompt:
                st.session_state.user_prompt_history.append(prompt)
                
                # Process user input
                with st.spinner("Generating response..."):
                    if mutex.locked():
                        st.write("Waiting to generate response...")
                        while mutex.locked():
                            time.sleep(0.1)

                    with mutex:
                        try:
                            history_limit = 5  # for example, last 5 exchanges
                            limited_history = st.session_state.chat_history[-history_limit*2:]  # *2 because each exchange has 2 messages
                            assistant = "\n".join([
                                f"Human: {msg}" if i % 2 == 0 else f"Assistant: {msg}"
                                for i, msg in enumerate(limited_history)
                            ])

                            mm_response = get_mm_response(model_path, prompt, st.session_state.embeddings, assistant)
                            if mm_response is None:
                                st.error("Issue processing your prompt, please try again.")
                            else:
                                st.session_state["chat_answers_history"].append(mm_response)
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

        # Display chat history in the chat container
        with tcol1:
            with chat_container:
                if st.session_state.chat_answers_history:
                    for user_msg, bot_msg in zip(st.session_state.user_prompt_history, st.session_state.chat_answers_history):
                        message1 = st.chat_message("user")
                        message1.write(user_msg)
                        message2 = st.chat_message("assistant")
                        message2.write(bot_msg)

       
    # Main column for model demo
    with tab2:
        st.header("Model Demo")
        tcol1, tcol2 = st.columns([0.4, 0.6])
        with tcol1:
            model_names = list(models.keys())

            # Define the partial model name you're looking for
            partial_model_name = ""

            # Find the index of the first model name that contains the partial name
            model_index = None
            for i, name in enumerate(model_names):
                if partial_model_name in name:
                    model_index = i
                    break

            selected_model = st.selectbox(
                "Select a model", 
                model_names,
                index=model_index,
                placeholder="- Select a model -"
            )

        if selected_model and selected_model != "- Select a model -":
            model_info = models[selected_model]
            st.write(f"**Model Path:** [{model_info['model_path']}](http://huggingface.co/{model_info['model_path']})")
            st.write(f"**Incentive:** {model_info['incentive']}")
            st.write(f"**Rank:** {model_info['rank']}")
            
            st.markdown('<h2 class="centered-title">Recent Video Metadata</h2>', unsafe_allow_html=True)
            st.divider()

            # Iterate over the DataFrame rows and create a button for each row
            for index, row in enumerate(video_metadata):
                # Create a three-column layout
                ccol1, ccol2, ccol3 = st.columns([0.45, 0.1, 0.4])
                with ccol1:
                    st.write(f"**YouTube ID:** {row['youtube_id']}")
                    st.write(f"**Description:** {row['description']}")
                    if st.button(f"Generate Caption for Video {row['youtube_id']}", key=f"button_{index}"):

                        with st.container(height=250):
                            #if mutex.waiting_threads() >= 5:
                                #st.warning("Too many concurrent threads, cannot use at this time. Please try again soon.")
                                #return

                            if mutex.locked():
                                with st.spinner("Waiting to start your generation..."):
                                    while mutex.locked():
                                        time.sleep(0.1)

                            with mutex:
                                try:
                                    with st.spinner('Generating caption...'):
                                        generated_caption = get_caption_from_model(model_info['model_path'], row['video_embed'])
                                    st.markdown(f"Generated Caption: {generated_caption}")

                                except Exception as e:
                                    st.exception(e)
                            
                with ccol3:
                    youtube_url = f"https://www.youtube.com/embed/{row['youtube_id']}"
                    st.video(youtube_url, start_time=row['start_time'])

                st.divider()
            
                            
    # tab for leaderboard
    with tab3:
        st.header("Leaderboard")

        # Prepare data for the table
        table_data = []
        for model_name, model_info in models.items():
            table_data.append({
                "rank": model_info['rank'],
                "UID": model_info['uid'],
                "model_path": f'https://huggingface.co/{model_info["model_path"]}',
                "incentive": str(round(float(model_info['incentive']), 3))
            })
        
        df = pd.DataFrame(table_data)

        st.dataframe(
            df, 
            width=800,
            hide_index=True,
            column_config={
                "model_path": st.column_config.LinkColumn(
                    "Repo",
                    validate="^https://huggingface\\.co/.*",
                    max_chars=100,
                    display_text="https://huggingface\\.co/(.*)",
                )
            }
        )

if __name__ == "__main__":
    asyncio.run(main())
