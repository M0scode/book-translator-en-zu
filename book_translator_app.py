# book_translator_app.py

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load MarianMT English to isiZulu translation model
@st.cache_resource

def load_model():
    model_name = 'Helsinki-NLP/opus-mt-en-zu'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Translation function that handles batching

def translate_text(text, tokenizer, model):
    sentences = sent_tokenize(text)
    translated_sentences = []

    for sentence in sentences:
        if sentence.strip():
            batch = tokenizer.prepare_seq2seq_batch([sentence], return_tensors="pt", truncation=True)
            output = model.generate(**batch)
            translated = tokenizer.decode(output[0], skip_special_tokens=True)
            translated_sentences.append(translated)

    return " ".join(translated_sentences)

# Streamlit UI
st.set_page_config(page_title="English to isiZulu Book Translator", layout="wide")
st.title("📚 English to isiZulu Book Translator")
st.markdown("This app translates English book text to isiZulu using a pre-trained MarianMT model.")

# File uploader for .txt files
uploaded_file = st.file_uploader("Upload your English book (.txt)", type=["txt"])

if uploaded_file:
    # Read the uploaded text file
    english_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("📖 Sample from Uploaded Book")
    st.text(english_text[:500])  # Show a preview of the text

    # Translate when button is pressed
    if st.button("Translate to isiZulu"):
        st.info("Translating... This may take a few moments.")

        # Break the input into paragraphs
        paragraphs = english_text.split("\n\n")
        translated_paragraphs = []

        for para in paragraphs:
            if para.strip():
                translated_paragraph = translate_text(para, tokenizer, model)
                translated_paragraphs.append(translated_paragraph)

        # Join translated paragraphs
        translated_text = "\n\n".join(translated_paragraphs)

        st.subheader("🔁 isiZulu Translation")
        st.text_area("Translated isiZulu Text", translated_text, height=300)

        # Download button for translation
        st.download_button(
            label="📥 Download Translated Text",
            data=translated_text,
            file_name="translated_book.txt",
            mime="text/plain"
        )
else:
    st.warning("Please upload a .txt file to begin.")
