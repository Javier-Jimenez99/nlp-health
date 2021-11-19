import numpy as np
import pandas as pd
from biopsias.form.generate_dataset import (
    generate_Y_histological_degree,
    generate_Y_caracter,
    generate_Y_is_multiple,
    join_X_data,
)


def test_join_X_data():
    df_list = {
        "Datos clínicos": [
            "Texto datos clinicos1",
            "Texto datos clinicos2",
        ],
        "Diagnóstico": ["Texto diagnostico1", "Texto diagnostico2"],
        "Macro": ["Texto macro1", "Texto macro2"],
        "Estudio": ["B17-06626", "B17-06624"],
    }

    s1 = join_X_data(pd.DataFrame(df_list))

    s2 = pd.Series(
        [
            "\nDatos clínicos: Texto datos clinicos1 \nDiagnóstico: Texto diagnostico1 \nMacro: Texto macro1",
            "\nDatos clínicos: Texto datos clinicos2 \nDiagnóstico: Texto diagnostico2 \nMacro: Texto macro2",
        ],
        index=["B17-06626", "B17-06624"],
    )

    pd.testing.assert_series_equal(s1, s2, check_names=False)


def test_histological_degree():
    df_list = {
        "tumor_features.histological_degree": [
            np.nan,
            "grado i ( 35 puntos ) bien diferenciado",
            "grado iii ( 9 puntos ) pobremente diferenciado",
            "grado ii moderadamente diferenciado",
            "grado ii ( 35 puntos ) moderadamente diferenciado",
            "grado i bien diferenciado",
            "gradoii ( 67 puntos ) moderadamente diferenciado",
            "carcinoma ductal infiltrante pobremente diferenciado",
            "invasion vascular peritumoral no",
        ]
    }

    s1 = generate_Y_histological_degree(pd.DataFrame(df_list))
    s2 = pd.Series([0, 1, 3, 2, 2, 1, 2, 0, 0])

    pd.testing.assert_series_equal(s1, s2, check_names=False)


def test_caracter():
    df_list = {
        "tumor_features.caracter": [
            np.nan,
            "bien circunscrito",
            "microinvasor ( foco / s infiltrante / s menores de 0 1 cm )",
            "microinvasor dos focos microinfiltrantes menores de 1 mm",
            "infiltrante ( con abundante componente in situ hasta un tamano tumoral total de 5 cm )",
            "infiltrant",
            "infiltrante + microinvasor",
        ]
    }

    s1 = generate_Y_caracter(pd.DataFrame(df_list))
    s2 = pd.Series([np.NaN, np.NaN, 0, 0, 1, 1, 1])

    pd.testing.assert_series_equal(s1, s2, check_names=False)


def test_is_multiple():
    df_list = {
        "tumor_features.is_multiple": [
            np.nan,
            "si ( cdi + focos de cdis )",
            "no",
            "si",
            "si ( multipes focos )",
            "si ( cdi en uce + cdis en ucs )",
            "no hay multiples focos de carcinoma lobulillar infiltrante + entre ellos ca lobulillar in situ",
        ]
    }

    s1 = generate_Y_is_multiple(pd.DataFrame(df_list))
    s2 = pd.Series([np.NaN, 1, 0, 1, 1, 1, 0])

    pd.testing.assert_series_equal(s1, s2, check_names=False)
