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

# Translate
def translate_en_to_zulu(text):
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("zul_Latn"),
        max_length=200,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #f4f1ee;
        color: #1b1b1b;
    }
    .stButton>button {
        background-color: #228b22;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #b22222;
        color: white;
    }
    .stTextArea textarea {
        background-color: #fffaf0;
        border: 1px solid #8b4513;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Translator Settings")
input_method = st.sidebar.radio("Input Method", ["Paste Text", "Upload .txt File"])

paragraphs = []

if input_method == "Paste Text":
    user_input = st.sidebar.text_area("Enter your text:", height=200)
    paragraphs = [p.strip() for p in user_input.split("\n") if p.strip()]
else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")
        paragraphs = [p.strip() for p in file_text.split("\n") if p.strip()]
        st.sidebar.success(f"Loaded {len(paragraphs)} paragraphs.")

# App title
st.title("🌍 English to isiZulu Book Translator")

# Tabs
tabs = st.tabs(["📝 Input", "🔁 Translation", "📥 Downloads"])

# Tab 1: Input Review
with tabs[0]:
    st.subheader("📄 Input Preview")
    if paragraphs:
        for i, p in enumerate(paragraphs):
            st.markdown(f"**Paragraph {i+1}:** {p}")
    else:
        st.info("Please paste text or upload a .txt file from the sidebar.")

# Tab 2: Translation
with tabs[1]:
    if paragraphs:
        if st.button("Translate"):
            results = []

            with st.spinner("Translating paragraphs..."):
                for i, p in enumerate(paragraphs, start=1):
                    translation = translate_en_to_zulu(p)

                    results.append({
                        "English": p,
                        "isiZulu": translation
                    })

            st.success("✅ Translation complete!")

            # Display each paragraph in text boxes
            for i, result in enumerate(results, start=1):
                st.subheader(f"Paragraph {i}")

                st.text_area(
                    "English",
                    value=result["English"],
                    height=180,
                    key=f"eng_{i}"
                )

                st.text_area(
                    "isiZulu",
                    value=result["isiZulu"],
                    height=180,
                    key=f"zu_{i}"
                )

                st.divider()

            # Save results for download later
            df = pd.DataFrame(results)
            st.session_state["translations"] = df

    else:
        st.warning("No input found. Please add text on the sidebar.")

# Tab 3: Downloads
with tabs[2]:
    if "translations" in st.session_state:
        df = st.session_state["translations"]

        st.subheader("📁 Download Translations")
        
        # CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "translations.csv", "text/csv")

        # JSON
        json_data = df.to_dict(orient="records")
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        st.download_button("Download JSON", json_str, "translations.json", "application/json")

        # TXT
        txt_lines = [f"{row['English']}\n→ {row['isiZulu']}\n" for _, row in df.iterrows()]
        txt_output = "\n".join(txt_lines)
        st.download_button("Download TXT", txt_output.encode("utf-8"), "translations.txt", "text/plain")
    else:
        st.info("No translations available yet.")

