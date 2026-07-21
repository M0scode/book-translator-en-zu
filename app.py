import streamlit as st
import time

from services.translation_service import translate_content

from ui.branding import render_branding
from ui.sidebar import render_sidebar
from ui.input_tab import render_input
from ui.results_view import render_results
from ui.download_tab import render_downloads
from ui.theme import load_css
from ui.learner_view import render_learner_mode

render_branding()
load_css()


metadata = render_sidebar()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📝 Translate",
        "📚 Results",
        "📖 Learner Mode",
        "⬇️ Downloads"
        
    ]
)


with tab1:

    text_content = render_input()


    if st.button(
        "🚀 Translate",
        key="translate_button"
    ):

        if not text_content:

            st.warning(
                "Please enter text before translating."
            )

        else:
            start_time = time.perf_counter()

            with st.spinner(
                "Translating content... This may take a minute for longer lessons."
            ):

                resource = translate_content(
                    text_content,
                    metadata,
                    input_type="text"
                )

            end_time = time.perf_counter()

            elapsed_time = round(
                end_time - start_time,
                2
            )

            resource["translation_time"] = elapsed_time

            st.session_state["resource"] = resource


            st.success(
                "Translation completed!"
                    )
                    


with tab2:

    if "resource" in st.session_state:

        render_results(
            st.session_state["resource"]
        )

    else:

        st.info(
            "Translate content first to view results."
        )



with tab3:

    if "resource" in st.session_state:

        render_learner_mode(
            st.session_state["resource"]
        )

    else:

        st.info(
            "Translate content first."
        )



with tab4:

    if "resource" in st.session_state:

        render_downloads(
            st.session_state["resource"]
        )

    else:

        st.info(
            "Translate content first to enable downloads."
        )