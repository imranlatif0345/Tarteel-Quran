import gradio as gr
import numpy as np
import os
from huggingface_hub import login
import logging
import xml.etree.ElementTree as ET
import html
import re
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
mytoken = os.getenv("Token_HUG")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Authentication (if using private models)
login(token=mytoken)  # Replace with your token

# Model Selection
# Initialize the Hugging Face ASR pipeline
asr_model = pipeline("automatic-speech-recognition", model="tarteel-ai/whisper-base-ar-quran")

# Function to load and parse the Quran XML file
def load_quran_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        surahs = []

        for sura in root.findall('sura'):
            sura_index = sura.get('index')
            sura_name = sura.get('name')
            verses = []

            for aya in sura.findall('aya'):
                aya_index = aya.get('index')
                aya_text = aya.get('text')
                verses.append({'index': aya_index, 'text': aya_text})

            surahs.append({'index': sura_index, 'name': sura_name, 'verses': verses})

        return surahs
    except ET.ParseError as e:
        logging.error(f"Error parsing XML: {e}")
        return []

# Load the Quran XML data at startup
quran_file_path = '/media/imran/Office1/MU-AI Works/MU Projects/Quran Tutor/quran-simple.xml'  # Update with your XML file path
surahs = load_quran_xml(quran_file_path)

# Create a list of Surah options for the dropdown
surah_options = [f"{sura['index']}: {sura['name']}" for sura in surahs]

# Function to transcribe and validate audio continuously
def transcribe_and_validate_live(selected_surah, selected_ayah, audio_chunk):
    if not selected_surah or not selected_ayah:
        return "Please select both a Surah and an Ayah.", "", ""

    # Get the selected Ayah text
    reference_text = get_selected_ayah_text(selected_surah, selected_ayah)
    if not reference_text:
        return "Selected Ayah not found.", "", ""

    # Normalize the reference text for comparison
    def normalize(text):
        text = re.sub(r'[\u064B-\u0652]', '', text)  # Remove diacritics
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()

    try:
        # Process the audio chunk using the ASR model
        transcription = asr_model(audio_chunk)['text']
        transcription_words = normalize(transcription).split()
        reference_words = normalize(reference_text).split()

        # Compare each word
        highlighted_text = ""
        for idx, word in enumerate(transcription_words):
            if idx < len(reference_words) and word == reference_words[idx]:
                highlighted_text += f"<span style='background-color: #90ee90;'>{html.escape(word)}</span> "
            else:
                highlighted_text += f"<span style='background-color: #ffcccb;'>{html.escape(word)}</span> "

        highlighted_text_html = f"<p>{highlighted_text.strip()}</p>"
        return transcription, "Live transcription and validation ongoing...", highlighted_text_html

    except Exception as e:
        logging.error(f"ASR Transcription Error: {e}")
        return "Error during live transcription.", "", ""

# Function to get Ayah options based on selected Surah
def get_ayah_options(selected_surah):
    if not selected_surah:
        return []
    try:
        surah_index = selected_surah.split(":")[0].strip()
        sura = next((s for s in surahs if s['index'] == surah_index), None)
        if not sura:
            return []
        ayah_options = [f"{aya['index']}: {aya['text']}" for aya in sura['verses']]
        return ayah_options
    except Exception as e:
        logging.error(f"Error getting Ayah options: {e}")
        return []

# Function to get the selected Ayah text

def get_selected_ayah_text(selected_surah, selected_ayah):
    if not selected_surah or not selected_ayah:
        return ""
    try:
        # Ensure the inputs are strings
        if isinstance(selected_ayah, list):
            selected_ayah = selected_ayah[0]  # Get the first element if it's a list
        if isinstance(selected_surah, list):
            selected_surah = selected_surah[0]  # Get the first element if it's a list
        
        # Now process the values as strings
        surah_index = selected_surah.split(":")[0].strip()
        ayah_index = selected_ayah.split(":")[0].strip()

        sura = next((s for s in surahs if s['index'] == surah_index), None)
        if not sura:
            return "Selected Surah not found."
        aya = next((a for a in sura['verses'] if a['index'] == ayah_index), None)
        if not aya:
            return "Selected Ayah not found."
        return aya['text']
    except Exception as e:
        logging.error(f"Error getting selected Ayah text: {e}")
        return "Error retrieving Ayah text."


# Clear function to reset the interface
def clear_interface():
    return gr.update(value=None), "", "", ""

# Design the Gradio interface using Blocks for better layout control
with gr.Blocks() as demo:
    gr.Markdown("# Quranic Arabic Corpus Ayah Practice (Live Mode)")
    gr.Markdown("""
    **Instructions:**
    1. Select a Surah from the first dropdown menu.
    2. Select an Ayah (verse) from the second dropdown menu.
    3. Click the microphone button to start live streaming and speak the selected Ayah clearly.
    4. The transcribed text will appear along with validation, highlighting words in real-time as correct or incorrect.
    """)

    # Surah and Ayah Selection
    with gr.Row():
        surah_dropdown = gr.Dropdown(choices=surah_options, label="Select Surah", value=None, interactive=True)
        ayah_dropdown = gr.Dropdown(choices=[], label="Select Ayah", value=None, interactive=True)

    ayah_text_display = gr.HTML(label="Selected Ayah Text")

    # Audio Input for Live Speech
    audio_input = gr.Audio(sources="microphone", type="numpy", label="Speak Your Ayah", streaming=True)

    # Transcribed Output
    transcribed_output = gr.Textbox(label="You Recited", interactive=False)

    # Validation Message
    validation_output = gr.Markdown(label="Validation Result")

    # Define the interaction: when a Surah is selected, update Ayah dropdown
    surah_dropdown.change(fn=get_ayah_options, inputs=[surah_dropdown], outputs=[ayah_dropdown])

    # Define the interaction: when an Ayah is selected, display its text
    ayah_dropdown.change(fn=get_selected_ayah_text, inputs=[surah_dropdown, ayah_dropdown], outputs=[ayah_text_display])

    # Define the interaction: when live audio is streamed, perform transcription and validation
    audio_input.stream(
        fn=transcribe_and_validate_live,
        inputs=[surah_dropdown, ayah_dropdown, audio_input],
        outputs=[transcribed_output, validation_output, ayah_text_display]
    )

    # Clear button
    clear_button = gr.Button("Clear")
    clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_output, validation_output, ayah_text_display])

# Launch the Gradio interface
demo.launch()
