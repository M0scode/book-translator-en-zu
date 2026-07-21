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


    for section in resource["sections"]:

        st.subheader(
            f"📖 Section {section['section_id']}"
        )


        st.markdown(
            "### 🇿🇦 Lesson Content"
        )

        st.write(
            section["lesson_content"]
        )


        with st.expander(
            "🇬🇧 View English Reference"
        ):

            st.write(
                section["english_reference"]
            )
        
        if section["key_terms"]:

            st.divider()

            st.subheader(
                "📘 Key Terms"
            )


            for term in section["key_terms"]:

                st.markdown(
                    f"""
                    **{term['zulu_term']}**

                    🇬🇧 {term['term']}

                    {term['meaning']}
                    """
            )