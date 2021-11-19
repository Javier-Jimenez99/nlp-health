from biopsias.form.parse_form_2 import parse_form_2

form_example1 = """TIPO DE INTERVENCIÓN:  
TUMORECTOMÍA DE MAMA IZQUIERDA CON AMPLIACIÓN DE MÁRGENES INFEROSUPEROINTERNO

Nº BIOPSIAS/ CITOLOGÍAS PREVIAS: B18-13629 

DATOS DEL TUMOR:

 1.- TIPO:                    
                 CARCINOMA DUCTAL IN SITU 
CON MICROINVASIÓN
       
2.- LOCALIZACIÓN:  CSI

3- TAMAÑO: 1,6 CM   (Solo para el CDIS. La medida es la histológica para tumores pequeños y la 
                               macro-microscópica para los grandes; si difieren prevalece la medida histológica)

 -  TAMAÑO FOCO MICROINVASOR <0,2 MM
 

4.- GRADO NUCLEAR DE VAN NUYS  (Solo CDIS):  INTERMEDIO            

5.- NECROSIS  (Solo CDIS. Es independiente del patrón arquitectural. No incluye células individuales necróticas descamadas):  PRESENTE (TIPO COMEDO)

6.- PATRÓN ARQUITECTURAL (SOLO CDIS):  CRIBIFORME / SÓLIDO   

7.- MÁRGENES DE LA BIOPSIA (Solo CDIS. Siempre marcados con tinta china):

 - CERCANO ( de 0,1 a 1 cm ): FOCO MICROINVASOR A 0,5 CM DEL MARGEN
 INFERIOR (SIN INCLUIR AMPLIACIÓN)
                                         CARCINOMA IN SITU A 0,2 CM DEL MARGEN INFERIOR (SIN INCLUIR AMPLIACIÓN)
                                                   
8.- MICROCALCIFICACIONES:  PRESENTES

A

PRESENCIA DE ARPÓN:   
         SI    - COINCIDENCIA CON EL CARCINOMA

DETERMINACION INMUNOHISTOQUÍMICA DE RECEPTORES HORMONALES:

RECEPTORES DE ESTROGENOS: POSITIVOS EN COMPONENTE IN SITU   (100% de células positivas)

COMENTARIOS / NOTAS ADICIONALES: PENDIENTE DE RECEPTORES HORMONALES, HER2 Y KI67 EN FOCO MICROINVASOR
"""
expected_dict1 = {
    "intervention_raw": "tumorectomia de mama izquierda con ampliacion de margenes inferosuperointerno",
    "intervention_automatic": ["tumorectomia"],
    "intervention_manual": ["tumorectomia"],
    "tumor_features": {
        "histological_type": "carcinoma ductal in situ \ncon microinvasión",
        "location": "csi",
        "size": "1,6 cm   (solo para el cdis. la medida es la histológica para tumores pequeños y la \n                               macro-microscópica para los grandes; si difieren prevalece la medida histológica)\n\n -  tamaño foco microinvasor <0,2 mm",
        "histological_degree": "intermedio",
        "tumor_necrosis": "(solo cdis. es independiente del patrón arquitectural. no incluye células individuales necróticas descamadas):  presente (tipo comedo)",
        "architectural_patern_insitu": "cribiforme / sólido",
        "margin": "- cercano ( de 0,1 a 1 cm ): foco microinvasor a 0,5 cm del margen\n inferior (sin incluir ampliación)\n                                         carcinoma in situ a 0,2 cm del margen inferior (sin incluir ampliación)",
    },
}

form_example2 = """TIPO DE INTERVENCIÓN:  Tumorectomia guiada por arpón y BSGC
Nº BIOPSIAS/ CITOLOGÍAS PREVIAS:  B10-00572
DATOS DEL TUMOR:
 1.- TIPO:                    
                 CARCINOMA DUCTAL IN SITU 
2.- LOCALIZACIÓN:  No consta en hoja de petición
3- TAMAÑO:    (Solo para el CDIS. La medida es la histológica para tumores pequeños y la 
                               macro-microscópica para los grandes; si difieren prevalece la medida histológica)
	-    0,6     cm
4.- GRADO NUCLEAR DE VAN NUYS  (Solo CDIS):   INTERMEDIO                  
5.- NECROSIS  (Solo CDIS. Es independiente del patrón arquitectural. No incluye células individuales necróticas descamadas):  PRESENTE (TIPO COMEDO) 
6.- PATRÓN ARQUITECTURAL (SOLO CDIS):  CRIBIFORME / SÓLIDO  / PAPILAR       
7.- MÁRGENES DE LA BIOPSIA (Solo CDIS. Siempre marcados con tinta china):
	- LIBRE (a 1 cm o más):  1,4    cm ( Con ampliación de margen )
8.- MICROCALCIFICACIONES:  PRESENTES 
ASOCIACIÓN CON OTRAS LESIONES:
	- MASTOPATÍA NO PROLIFERATIVA: Fibrosis
INDICE PRONÓSTICO DE VAN NUYS (SOLO CDIS):
                - GRADO BAJO  (puntuación 3-4)
PRESENCIA DE ARPÓN:      
			      SI    - COINCIDENCIA CON EL CARCINOMA
DETERMINACION INMUNOHISTOQUÍMICA DE RECEPTORES HORMONALES:
RECEPTORES DE ESTROGENOS:       Positivo                      (  100% de células positivas)
RECEPTORES DE PROGESTERONA:    Positivo                     ( 90% de células positivas)"""

expected_dict2 = {
    "intervention_raw": "tumorectomia guiada por arpon + bsgc",
    "intervention_automatic": ["bsgc", "tumorectomia"],
    "intervention_manual": ["bsgc", "tumorectomia"],
    "tumor_features": {
        "histological_type": "carcinoma ductal in situ",
        "location": "no consta en hoja de petición",
        "size": "(solo para el cdis. la medida es la histológica para tumores pequeños y la \n                               macro-microscópica para los grandes; si difieren prevalece la medida histológica)\n\t-    0,6     cm",
        "histological_degree": "intermedio",
        "tumor_necrosis": "(solo cdis. es independiente del patrón arquitectural. no incluye células individuales necróticas descamadas):  presente (tipo comedo)",
        "architectural_patern_insitu": "cribiforme / sólido  / papilar",
        "margin": "- libre (a 1 cm o más):  1,4    cm ( con ampliación de margen )",
    },
}


text_forms = [
    form_example1,
    form_example2,
]
parsed_forms = [
    expected_dict1,
    expected_dict2,
]


def test_parse_form_2_hardcoded():
    for form, expected_dict in zip(text_forms, parsed_forms):
        result_dict = parse_form_2(form)
        assert (
            result_dict == expected_dict
        ), f"Returned dict: {result_dict}\n and expected dict: {expected_dict}\n are different"


def test_parse_form_2_empty():
    form_example_empty = "test no debe encontrar nada"
    form = form_example_empty
    expected_dict = {}

    result_dict = parse_form_2(form)
    assert (
        result_dict == expected_dict
    ), f"Returned dict: {result_dict}\n and expected dict: {expected_dict}\n are different"
