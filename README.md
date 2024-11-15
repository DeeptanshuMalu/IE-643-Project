# IE643 NeuroTwins Project

This project is a demonstration of the IE643 NeuroTwins application, which transcribes audio input into LaTeX code, corrects it using OpenAI's GPT-4o-mini model, and compiles it into a PDF. Below is a description of each file in the project:

## Files

### `app.py`
This is the main application file that sets up the Streamlit interface and handles the core functionality:
- Initializes the Whisper and T5 models for transcription and LaTeX generation.
- Provides options to upload or record audio.
- Transcribes audio input into text.
- Uses OpenAI's GPT-4o-mini model to correct the LaTeX code.
- Compiles the corrected LaTeX code into a PDF.
- Calculates the TeXBLEU score for the generated LaTeX code.

### `api_latex.ipynb`
A Jupyter notebook that demonstrates how to use OpenAI's GPT-4o-mini model to correct LaTeX code. It includes:
- Setting up the OpenAI API key.
- Defining a function to get a response from the OpenAI API.

### `evaluation.ipynb`
A Jupyter notebook for evaluating the performance of the models used in the project.

### `fineuned_T5_testing.ipynb`
A Jupyter notebook for testing the fine-tuned T5 model. It includes:
- Loading the fine-tuned T5 model and tokenizer.
- Generating LaTeX code from sample inputs.

### `mathbridge_t5_huggingface.ipynb`
A Jupyter notebook for training and saving the T5 model using the Hugging Face library. It includes:
- Saving the fine-tuned T5 model and tokenizer.

### `mathbridge_to_speech.ipynb`
A Jupyter notebook for converting mathematical text to speech using the trained models.

### `saytex_mathspeech_testing.ipynb`
A Jupyter notebook for testing the SayTeX and MathSpeech models.

### `whisper.ipynb`
A Jupyter notebook for testing the Whisper model for automatic speech recognition.

### `imp_term.txt`
A text file containing important terms related to the project.

### `TeXBLEU/`
A directory containing the implementation of the TeXBLEU metric used to evaluate the quality of the generated LaTeX code.

### `fine_tuned_t5/`, `fine_tuned_t5_2L_e5_cb_ca/`, `fine_tuned_t5_2L_e5_cb_ca_spl/`, `fine_tuned_t5_2L_e5_noc/`
Directories containing different versions of the fine-tuned T5 model.

## Usage
1. Add your OpenAI API key to the environments variables of your computer. The variable should be named as **OPENAI_API_KEY**.

    The steps to setup the api key can be found [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

2. To install all the required libraries
    ```sh
    pip install -r requirements.txt
    ```

3. To run the application, execute the following command:
    ```sh
    streamlit run app.py
    ```