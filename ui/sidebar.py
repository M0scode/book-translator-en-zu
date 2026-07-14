import streamlit as st


def render_sidebar():

    st.sidebar.header(
        "📚 Curriculum Information"
    )


    grade = st.sidebar.selectbox(
        "Grade",
        [8,9,10,11,12]
    )


    subject = st.sidebar.text_input(
        "Subject"
    )


    term = st.sidebar.selectbox(
        "Term",
        [1,2,3,4]
    )


    topic = st.sidebar.text_input(
        "Topic"
    )


    return {

        "grade": grade,

        "subject": subject,

        "term": term,

        "topic": topic

    }