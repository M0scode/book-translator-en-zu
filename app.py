"""
Main Streamlit application.

User interface only.
"""


import streamlit as st



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

    st.subheader(
        "📄 Text Preview"
    )


    st.text_area(
        "Content",
        text_content,
        height=200
    )