import streamlit as st


def render_input():

    st.header(
        "📝 Learning Material Input"
    )


    method = st.radio(
        "Choose input method:",
        [
            "Type/Paste Text",
            "Upload File"
        ]
    )


    text_content = None


    if method == "Type/Paste Text":

        text_content = st.text_area(
            "Enter English text:",
            height=250
        )


    else:

        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt","pdf"]
        )


        if uploaded_file:

            text_content = (
                uploaded_file
                .read()
                .decode("utf-8")
            )


    return text_content