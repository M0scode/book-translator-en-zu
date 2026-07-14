"""
Main Streamlit application.

User interface only.
"""


import streamlit as st

from services.translation_service import (
    translate_content
)

st.set_page_config(
    page_title="English - isiZulu Translator",
    page_icon="🌍",
    layout="wide"
)



st.title(
    "🌍 English to isiZulu Curriculum Translator"
)


st.write(
    """
    Translate curriculum material into isiZulu
    learning resources.
    """
)



st.sidebar.header(
    "📚 Curriculum Information"
)



grade = st.sidebar.selectbox(
    "Grade",
    [
        8,
        9,
        10,
        11,
        12
    ]
)


subject = st.sidebar.text_input(
    "Subject"
)


term = st.sidebar.selectbox(
    "Term",
    [
        1,
        2,
        3,
        4
    ]
)

metadata = {

    "grade": grade,

    "subject": subject,

    "term": term

}

st.header(
    "Learning Material Input"
)


input_method = st.radio(
    "Choose input method:",
    [
        "Type/Paste Text",
        "Upload File"
    ]
)



text_content = None



if input_method == "Type/Paste Text":

    text_content = st.text_area(
        "Enter English text:",
        height=250,
        placeholder=
        """
        Paste curriculum content here...
        
        Example:
        Photosynthesis is the process by which plants
        produce food using sunlight.
        """
    )



else:

    uploaded_file = st.file_uploader(
        "Upload document",
        type=[
            "txt",
            "pdf"
        ]
    )


    if uploaded_file:

        if uploaded_file.type == "text/plain":

            text_content = (
                uploaded_file
                .read()
                .decode("utf-8")
            )

        else:

            st.info(
                "PDF processing will be connected in the next milestone."
            )

if text_content:

    st.text_area(
        "Content",
        text_content,
        height=200,
        key="input_text"
    )

if st.button("🚀 Translate"):

    if not text_content:
        st.warning(
            "Please enter text before translating."
        )

    else:

        with st.spinner(
            "Translating content..."
        ):

            resource = translate_content(
                text_content,
                metadata,
                input_type="text"
            )


        st.session_state["resource"] = resource


        st.success(
            "Translation completed!"
        )

if "resource" in st.session_state:

    resource = st.session_state["resource"]


    st.header(
        "📚 Translation Result"
    )


    for index, item in enumerate(resource["content"], start=1):

        st.subheader(
            f"Paragraph {index}"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.text_area(
                "English",
                item["english"],
                height=150,
                key=f"english_{index}"
            )


        with col2:

            st.text_area(
                "isiZulu",
                item["zulu"],
                height=150,
                key=f"zulu_{index}"
            ) 

        st.header(
            "⬇️ Download Learning Resources"
        )


        files = resource["files"]


        with open(
            files["docx"],
            "rb"
        ) as file:

            st.download_button(
                label="📄 Download DOCX",
                data=file,
                file_name="translation.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


        with open(
            files["pdf"],
            "rb"
        ) as file:

            st.download_button(
                label="📕 Download PDF",
                data=file,
                file_name="translation.pdf",
                mime="application/pdf"
            )


        with open(
            files["csv"],
            "rb"
        ) as file:

            st.download_button(
                label="📊 Download CSV",
                data=file,
                file_name="translation.csv",
                mime="text/csv"
            )


        with open(
            files["json"],
            "rb"
        ) as file:

            st.download_button(
                label="🗂 Download JSON",
                data=file,
                file_name="translation.json",
                mime="application/json"
            )


        with open(
            files["mobile"],
            "rb"
        ) as file:

            st.download_button(
                label="📱 Download Learner Mobile Resource",
                data=file,
                file_name="learner_resource.json",
                mime="application/json"
            )