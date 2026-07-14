import streamlit as st


def render_downloads(resource):

    st.header(
        "⬇️ Download Learning Resources"
    )


    files = resource["files"]


    col1, col2 = st.columns(2)


    # DOCX
    with col1:

        with open(
            files["docx"],
            "rb"
        ) as file:

            st.download_button(
                label="📄 Download Word Document",
                data=file,
                file_name="translation.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx"
            )


    # PDF
    with col2:

        with open(
            files["pdf"],
            "rb"
        ) as file:

            st.download_button(
                label="📕 Download PDF",
                data=file,
                file_name="translation.pdf",
                mime="application/pdf",
                key="download_pdf"
            )


    col3, col4 = st.columns(2)


    # CSV
    with col3:

        with open(
            files["csv"],
            "rb"
        ) as file:

            st.download_button(
                label="📊 Download CSV",
                data=file,
                file_name="translation.csv",
                mime="text/csv",
                key="download_csv"
            )


    # JSON
    with col4:

        with open(
            files["json"],
            "rb"
        ) as file:

            st.download_button(
                label="🗂 Download JSON",
                data=file,
                file_name="translation.json",
                mime="application/json",
                key="download_json"
            )


    st.divider()


    # Learner resource
    st.subheader(
        "📱 Learner Mobile Resource"
    )


    st.write(
        """
        This file is designed for learner applications
        and future mobile learning platforms.
        """
    )


    with open(
        files["mobile"],
        "rb"
    ) as file:

        st.download_button(
            label="📱 Download Learner Resource",
            data=file,
            file_name="learner_resource.json",
            mime="application/json",
            key="download_mobile"
        )