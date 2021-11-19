import random
from collections import namedtuple

import numpy as np
import pandas as pd
import regex as re
import streamlit as st
from biopsias.config import (
    all_biopsias_path,
    histological_degree_model_path,
    histological_type_model_path,
    intervention_type_model_path,
    is_cdi_model_path,
    tumor_size_labels,
    tumor_size_automatic_model_path,
)
from biopsias.form.generate_dataset import generate_dataset
from biopsias.form.interventions import clean_x_no_deep_learning
from biopsias.ner_huggingface import HuggingfaceNerModel, decode_predictions
from biopsias.train import split_tokenizer
from joblib import load
from spacy import displacy
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
import toml

Models = namedtuple(
    "Models", "intervention histological_type is_cdi histological_degree tumor_size"
)


@st.cache(allow_output_mutation=True)
def load_all_data():
    parquet = pd.read_parquet(all_biopsias_path)
    biopsias_df = parquet[parquet["valid"]]
    return biopsias_df


def load_random_text():
    biopsias_df = load_all_data()

    return random.choice(biopsias_df["Diagnóstico"].tolist())


def load_models():
    size_model = HuggingfaceNerModel(tumor_size_labels, tumor_size_automatic_model_path)

    return Models(
        load(intervention_type_model_path),
        load(histological_type_model_path),
        load(is_cdi_model_path),
        load(histological_degree_model_path),
        size_model,
    )


def clean_text_to_display(text):
    text = re.sub(r"\#\#ENTRY_\d+\#\#", "", text)
    text = text.replace("\n", "").strip()

    return text


def intervention_inference(all_model, text):
    model = all_model[0]
    labels = all_model[1]

    clean_text = clean_x_no_deep_learning(text)
    index = model.predict([clean_text]).squeeze().astype(bool)

    return ",".join(np.array(labels)[index])


def simple_inference(model, text):
    clean_text = clean_x_no_deep_learning(text)
    pred = model.predict([clean_text])[0]

    return pred


st.cache(allow_output_mutation=True)


def get_theme_colors():
    with open(".streamlit/config.toml", "r") as f:
        return toml.load(f)["theme"]


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    theme_colors = get_theme_colors()
    back_color = theme_colors["secondaryBackgroundColor"]

    WRAPPER = (
        '<div style="overflow-x: auto; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem;background-color:'
        + back_color
        + '">{}</div>'
    )
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def ner_visualizer(words, ner, st_container):
    ents = []

    for pos, tag in enumerate(ner):
        if tag.startswith("B"):
            ents.append({"start": pos, "end": pos + 1, "label": tag.split("-")[1]})
        elif tag.startswith("I"):
            ents[-1]["end"] = pos + 1

    doc = Doc(Vocab(strings=set(words)), words=words, spaces=[True] * len(words))

    for ent in ents:
        doc.ents = list(doc.ents) + [Span(doc, **ent)]

    # visualize_ner(doc, labels=["TUMOR_SIZE"], show_table=False, title="Tumor size")
    html = displacy.render(
        doc,
        style="ent",
        options={
            "ents": ["TUMOR_SIZE"],
            "colors": {"TUMOR_SIZE": get_theme_colors()["backgroundColor"]},
        },
    )

    style = "<style>mark.entity { display: inline-block }</style>"
    st_container.write("### Tumor size:")
    st_container.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


def tumor_size_inference(model, text, st_container):
    text = clean_text_to_display(text)
    predictions = model.predict(text)
    pred_labels = predictions.labels
    tokens_hugg = predictions.tokens

    text_splited = split_tokenizer(text)

    labels_text_splited = decode_predictions(tokens_hugg, pred_labels, text_splited)

    ner_visualizer(text_splited, labels_text_splited, st_container)


def execute_all_inference(models, text):
    names = [
        "### Intervención",
        "### Tipo histológico",
        "### Grado histológico",
        "### Carcinoma ductal infiltrante",
    ]

    values = (
        intervention_inference(models.intervention, text),
        simple_inference(models.histological_type, text),
        str(simple_inference(models.histological_degree, text)),
        "Sí"
        if simple_inference(models.is_cdi, text) == "carcinoma ductal infiltrante"
        else "No",
    )
    cols = st.columns(4)
    tumor_size_container = st.container()

    for i, n in enumerate(names):
        cols[i].write(" ")
        cols[i].write(n)

    for i, v in enumerate(values):
        if i == 0:
            for e in v.split(","):
                cols[i].warning(f"{e}")
        else:
            cols[i].warning(f"{v}")

    tumor_size_inference(models.tumor_size, text, tumor_size_container)


def app():
    st.title("Estudio de formularios")
    with st.spinner("Cargando modelos..."):
        models = load_models()

    option = st.selectbox(
        "Selecciona la forma del texto de entrada:", ("Random", "Manual")
    )

    if option == "Random":
        text_cleaned_to_display = clean_text_to_display(load_random_text())

        with st.form("random"):
            text = st.text_area("Texto aleatorio", text_cleaned_to_display)

            run_button = st.form_submit_button("Random")

            st.write("## Resultados")
            execute_all_inference(models, text)

        if run_button:
            pass

    elif option == "Manual":
        with st.form("manual"):
            text = st.text_area("Escribe un diagnóstico")

            run_button = st.form_submit_button("Run")

        if run_button:

            st.write("## Resultados")
            execute_all_inference(models, text)


if __name__ == "__main__":
    app()
