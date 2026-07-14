import streamlit as st


def render_learner_mode(resource):

    st.header(
        "📚 Learn in isiZulu"
    )


    metadata = resource["metadata"]


    st.info(
        f"""
        Grade: {metadata.get('grade')}

        Subject: {metadata.get('subject')}
        """
    )


    for index, item in enumerate(
        resource["content"],
        start=1
    ):

        st.subheader(
            f"📖 Section {index}"
        )


        st.markdown(
            "**🇿🇦 isiZulu Learning Content**"
        )

        st.write(
            item["zulu"]
        )


        st.divider()


        st.markdown(
            "**🇬🇧 English Reference**"
        )

        st.write(
            item["english"]
        )