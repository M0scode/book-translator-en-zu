import streamlit as st


def render_results(resource):

    st.header(
        "📚 Translation Result"
    )
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "⏱ Translation Time",
            f"{resource.get('translation_time', 0)} s"
        )

    with col2:

        st.metric(
            "📄 Paragraphs",
            len(resource["content"])
        )

    with col3:

        characters = sum(
            len(item["english"])
            for item in resource["content"]
        )

        st.metric(
            "📝 Characters",
            characters
        )

    st.divider()

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