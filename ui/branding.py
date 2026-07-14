import streamlit as st


def render_branding():

    st.set_page_config(
        page_title="2S0 Curriculum Translator",
        page_icon="🌍",
        layout="wide"
    )


    st.title(
        "🌍 2S0 Curriculum Translator"
    )


    st.subheader(
        "English → isiZulu AI-powered learning resources"
    )


    st.write(
        """
        Transform curriculum material into isiZulu
        learning resources for teachers and learners.
        """
    )