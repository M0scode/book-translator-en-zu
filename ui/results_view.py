import streamlit as st


def render_results(resource):

    st.header(
        "📚 Translation Result"
    )


    for index,item in enumerate(
        resource["content"],
        start=1
    ):

        st.subheader(
            f"Paragraph {index}"
        )


        st.markdown(
            f"""
**🇬🇧 English**

{item['english']}


**🇿🇦 isiZulu**

{item['zulu']}

---
"""
        )