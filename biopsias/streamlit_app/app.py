import streamlit as st
from biopsias.streamlit_app import forms_page, snomed_page


class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.write("# Bio-analysis")
        st.sidebar.write("## Seleccione la página deseada")

        st.markdown(
            """
            <style>
            [data-baseweb="select"] {
                margin-top: -25px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Drodown to select the page to run
        page = st.sidebar.selectbox(
            "", self.pages, format_func=lambda page: page["title"]
        )

        # run the app function
        page["function"]()

        st.sidebar.write("## Info")
        st.sidebar.write(
            "Demostración del funcionamiento del los modelos utilizados para extraer información de biopsias."
        )

        st.sidebar.write(" ")
        st.sidebar.write(" ")

        st.sidebar.image("biopsias/streamlit_app/logo_datalab_header.png")


if __name__ == "__main__":
    st.set_page_config("Biopsies analysis tool", layout="wide")

    app = MultiPage()

    app.add_page("Estudio de formularios", forms_page.app)
    app.add_page("Clasificación SNOMED", snomed_page.app)

    app.run()
