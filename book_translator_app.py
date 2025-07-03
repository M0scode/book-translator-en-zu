import streamlit as st
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Translation function
def translate_en_to_zulu(text):
    tokenizer.src_lang = "eng_Latn"
    zul_lang_token_id = tokenizer.convert_tokens_to_ids("zul_Latn")
    model.config.forced_bos_token_id = zul_lang_token_id

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    tokens = model.generate(
        **inputs,
        max_length=200,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("📘 English to isiZulu Translator - Batch Mode")
st.markdown("Upload a `.txt` file or paste multiple English paragraphs to translate into isiZulu.")

# Text input or file upload
input_method = st.radio("Input Method", ["Paste Text", "Upload .txt File"])

if input_method == "Paste Text":
    input_text = st.text_area("Enter English text (one or more paragraphs):", height=200)
    paragraphs = [p.strip() for p in input_text.split("\n") if p.strip()]
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    paragraphs = []
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8")
        paragraphs = [p.strip() for p in file_text.split("\n") if p.strip()]
        st.text_area("File Content Preview:", file_text, height=200)

if paragraphs and st.button("Translate"):
    results = []
    with st.spinner("Translating..."):
        for p in paragraphs:
            translation = translate_en_to_zulu(p)
            results.append({"English": p, "isiZulu": translation})

    # Display results
    st.success("Translations complete!")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # File download options
    st.markdown("### 📥 Download Translations")
    
    # CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", csv, "translations.csv", "text/csv")

    # JSON
    json_data = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button("Download as JSON", json_data, "translations.json", "application/json")

    # TXT
    txt_lines = [f"{row['English']}\n→ {row['isiZulu']}\n" for row in results]
    txt_content = "\n".join(txt_lines)
    st.download_button("Download as TXT", txt_content.encode("utf-8"), "translations.txt", "text/plain")
