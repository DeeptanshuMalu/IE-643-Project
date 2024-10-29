import base64
from pathlib import Path
import subprocess
import tempfile
import streamlit as st
from pydub import AudioSegment
import io
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Remove the file if it exists
filename = "recorded_audio.wav"
if os.path.exists(filename):
    os.remove(filename)


def init_pipe():
    if "pipe_whisper" not in st.session_state:
        with st.spinner("Loading whisper..."):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model_id = "openai/whisper-large-v3"

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)

            st.session_state.pipe_whisper = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )

    if "pipe_model" not in st.session_state:
        with st.spinner("Loading model..."):
            pipe_model = pipeline(
                "text2text-generation",
                model="Hyeonsieun/MathSpeech_T5_base_translator",
                max_length=1000,
            )
            st.session_state.pipe_model = pipe_model


def get_transcription():
    with st.spinner("Transcribing..."):
        output = st.session_state.pipe_whisper("recorded_audio.wav")["text"]
    return output


def get_openai_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. Correct the input latex code to make it valid. Only return the corrected code. The returned code should be enclosed in a pair of $. For example, if the input is "x^2", the output should be "$x^2$". """,
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def compile_latex_to_pdf(latex_code):

    latex_content = f"""
            \\documentclass{{article}}
            \\begin{{document}}

            Hello, this is a sample LaTeX document!
            {latex_code}
            \\end{{document}}
        """

    tex_file = "latex/document.tex"

    # Write LaTeX content to .tex file
    with open(tex_file, "w") as f:
        f.write(latex_content)

    # Compile the LaTeX file into a PDF
    subprocess.run(
        ["pdflatex", "-output-directory", "latex", tex_file], check=True, cwd="latex"
    )

    # Path to the resulting PDF
    pdf_path = "latex/document.pdf"

    # Return PDF data
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)


init_pipe()

st.title("IE643 NeuroTwins Demo")

with st.container(border=True):
    audio_option = st.selectbox("Give audio input", ["Upload", "Record"])

    if audio_option == "Upload":
        uploaded_file = st.file_uploader("Choose a sound file", type=["wav", "mp3"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            with open(filename, "wb") as f:
                f.write(uploaded_file.getbuffer())

    if audio_option == "Record":
        audio = st.experimental_audio_input(label="Record audio")

        if audio:
            audio_data = audio.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
            louder_audio_segment = audio_segment + 20

            output_buffer = io.BytesIO()
            louder_audio_segment.export(output_buffer, format="wav")
            output_buffer.seek(0)

            st.audio(output_buffer, format="audio/wav")

            with open(filename, "wb") as f:
                f.write(output_buffer.read())

    if st.button("Transcribe"):
        st.session_state.transcription = get_transcription()

    if st.session_state.get("transcription"):
        st.session_state.transcription = st.text_input(
            label="Transcription", value=st.session_state.transcription
        )

        if st.button("Convert to LaTeX"):
            with st.spinner("Generating LaTeX..."):
                st.session_state.latex = st.session_state.pipe_model(
                    st.session_state.transcription
                )[0]["generated_text"]

        if st.session_state.get("latex"):
            st.text_input("LaTeX before API", value=st.session_state.latex)

            if st.button("Send to API"):
                with st.spinner("Sending to API..."):
                    response = get_openai_response(st.session_state.latex)

                st.text_input("LaTeX after API", value=response)
                compile_latex_to_pdf(response)
