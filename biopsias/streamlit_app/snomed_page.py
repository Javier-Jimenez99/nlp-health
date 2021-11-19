import random
from collections import namedtuple

import pandas as pd
import regex as re
from regex.regex import sub
import streamlit as st
from biopsias.config import all_snomed_data_path, morf_save_path, top_save_path
from biopsias.form.interventions import clean_x_no_deep_learning
from joblib import load
from biopsias.snomed.generate_snomed_dataset import generate_snomed_dataset, get_topk_XY

Models = namedtuple("Models", "morf top")


@st.cache(allow_output_mutation=True)
def load_dataset():
    dataset = generate_snomed_dataset(all_snomed_data_path).dropna(
        subset=["x_Diagnostico", "y_CodTopografico", "y_ConceptID_Morphologic"]
    )

    morf_dataset = get_topk_XY(
        dataset,
        "x_Diagnostico",
        "y_ConceptID_Morphologic",
        translate_labels_path="data/clean/id_desc_morf.json",
    ).rename(columns={"y": "y_morf"})

    topo_dataset = (
        get_topk_XY(
            dataset,
            "x_Diagnostico",
            "y_CodTopografico",
            translate_labels_path="data/clean/id_desc_top.json",
        )
        .rename(columns={"y": "y_top"})
        .drop(columns=["x"])
    )

    result_df = pd.concat([topo_dataset, morf_dataset], axis=1)
    return result_df


def load_random_value():
    dataset = load_dataset().dropna()
    dataset_no_otro = dataset[~dataset.isin(["otro"]).any(axis=1)]
    data = dataset_no_otro.sample()

    return {
        "x": data["x"].iloc[0],
        "y_morf": data["y_morf"].iloc[0],
        "y_top": data["y_top"].iloc[0],
    }


def load_models():
    return Models(
        load(morf_save_path),
        load(top_save_path),
    )


def clean_text_to_display(text):
    text = re.sub(r"\#\#ENTRY_\d+\#\#", "", text)
    text = text.replace("\n", "").strip()

    return text


def simple_inference(model, text):
    clean_text = clean_x_no_deep_learning(text)
    pred = model.predict([clean_text])[0]

    return pred


def execute_all_inference(models, text, predictions=None):
    cols = st.columns(2)

    names = [
        "### Morfológico",
        "### Topológico",
    ]

    values = (
        simple_inference(models.morf, text),
        simple_inference(models.top, text),
    )

    for i, n in enumerate(names):
        cols[i].write(" ")
        cols[i].write(n)

    for i, v in enumerate(values):
        if predictions is None:
            cols[i].warning(f"## {v}")
        elif predictions[i] == v:
            cols[i].success(f"## {v}")
        else:
            cols[i].error(f"## {v}")
            cols[i].warning(f"## {predictions[i]}")


def app():
    st.title("Clasificación SNOMED")
    with st.spinner("Cargando modelos..."):
        models = load_models()

    option = st.selectbox(
        "Selecciona la forma del texto de entrada:", ("Random", "Manual")
    )

    if option == "Random":
        error = True
        while error:
            try:
                random_value = load_random_value()

                diagnostico = random_value["x"]
                y_morf = random_value["y_morf"]
                y_top = random_value["y_top"]

                text_cleaned_to_display = clean_text_to_display(diagnostico)

                with st.form("random"):
                    text = st.text_area("Texto aleatorio", text_cleaned_to_display)

                    run_button = st.form_submit_button("Random")

                    st.write("## Resultados")
                    execute_all_inference(models, text, (y_morf, y_top))

                if run_button:
                    pass

                error = False
            except TypeError:
                pass

    elif option == "Manual":
        error = True
        while error:
            try:
                with st.form("manual"):
                    text = st.text_area("Escribe un diagnóstico")

                    run_button = st.form_submit_button("Run")

                if run_button:

                    st.write("## Resultados")
                    execute_all_inference(models, text)
                error = False
            except TypeError:
                pass


if __name__ == "__main__":
    app()
