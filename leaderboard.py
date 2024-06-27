# pip install streamlit datasets huggingface-hub ulid-py bittensor

import asyncio
import os
import time
import json

import streamlit as st
import pandas as pd
import threading

from neurons.model_scoring import get_caption_from_model

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
        section.main > div {max-width:1200px;}

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
    st.markdown('<h1 class="centered-title">OMEGA Any-to-Any Leaderboard</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="intro-text">Welcome to OMEGA Labs\' Any-to-Any model demos and leaderboard. This streamlit showcases video captioning capabilities from the latest models on Bittensor\'s subnet 21.<br /><strong>*Please note most models are undertrained right now (Q2 2024) given the early days of the subnet.</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="intro-text">On the "Model Demos" tab, select a miner\'s model from the dropdown and then browse recent video submissions from subnet 24. Interact with the model by pressing the "Generate Caption for Video" button.</p>', unsafe_allow_html=True)
    st.markdown('<p class="intro-text">On the "Leaderboard" tab, checkout the latest rankings.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Model Demos", "Leaderboard"])

    # Main column for model demo
    with tab1:
        st.header("Model Demo")
        tcol1, tcol2 = st.columns([0.4, 0.6])
        with tcol1:
            model_names = list(models.keys())

            # Define the partial model name you're looking for
            partial_model_name = "UID 31"

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
    with tab2:
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
