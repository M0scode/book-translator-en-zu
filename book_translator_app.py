import streamlit as st
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

    inputs = tokenizer(text, return_tensors="pt")
    tokens = model.generate(
        **inputs,
        max_length=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("📘 English to isiZulu Translator")
st.markdown("Translate English sentences into isiZulu using NLLB-200")

input_text = st.text_area("Enter English text:", height=150)

if st.button("Translate"):
    if input_text.strip():
        translation = translate_en_to_zulu(input_text)
        st.success("isiZulu Translation:")
        st.write(translation)
    else:
        st.warning("Please enter some text to translate.")
