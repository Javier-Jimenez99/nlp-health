from biopsias.form.measure_rules import define_nlp


def test_regex_measure_extraction():
    expected_span = [
        {"start": 430, "end": 434, "label": "MEASURE"},
        {"start": 444, "end": 450, "label": "MEASURE"},
        {"start": 460, "end": 466, "label": "MEASURE"},
        {"start": 916, "end": 928, "label": "MEASURE"},
        {"start": 1180, "end": 1184, "label": "MEASURE"},
        {"start": 1312, "end": 1318, "label": "MEASURE"},
        {"start": 1435, "end": 1443, "label": "MEASURE"},
        {"start": 1618, "end": 1624, "label": "MEASURE"},
    ]

    nlp = define_nlp()
    text = "\nDatos clínicos: ##ENTRY_1## Carcinoma ductal infiltrante de mama izquierda que afecta a unión de cuadrantes inferiores, a unión de cuadrantes externos y cuadrante superoexterno.\nInterveción: Mastectomía radical modificada. Niveles I y II de Berg. \nDiagnóstico: ##ENTRY_1## PIEZA DE MASTECTOMIA RADICAL IZQUIERDA Y  LINFADENECTOMIA AXILAR CON:\n   -   CARCINOMA DUCTAL INFILTRANTE MODERADAMENTE DIFERENCIADO ( GII ), MÚLTIPLE,  DE 6 CM ( UCE ), 3.2 CM (CSE ) Y 2.5 CM (UCI ) DE DIAMETROS MÁXIMOS, CON FOCOS DE CARCINOMA INTRADUCTAL ( MENOS DEL 25% DE MASA TUMORAL ), TIPO HISTOLOGICO SOLIDO-CRIBIFORME, GRADO NUCLEAR MEDIO, CON COMEDONECROSIS. EXTENSA PERMEACIÓN VASCULAR POR CÉLULAS NEOPLASICAS.\n   -  METÁSTASIS DE CARCINOMA EN NUEVE DE DIECIOCHO GANGLIOS LINFATICOS AXILARES.\n   -  MAMA NO TUMORAL: FIBROSIS \nMacro: ##ENTRY_1## Pieza de mastectomía radical y linfadenectomía axilar cuya porción cutánea mide entre 20 x 5,5 cms y no presenta alteraciones macroscópicas. Realizados cortes seriados del parénquima mamario apreciamos incluidos dentro de abundante estroma adiposo tres nódulos de aspecto tumoral. El primero de ellos está  situado en cuadrante supero externo y mide 4 cm de diámetro máximo y está constituído por un tejido blanquecino y grisáceo.Todos los nódulos tumorales están situados a más de 1.5 cm de margen profundo. Se toman cortes representativos 1 al 4. A nivel de cuadrante externos apreciámos otro nódulo de 7 x 4 cm constituído por un tejido blanquecino firme y grisáceo. Se toman cortes representativos, 5 al 9. A nivel de cuadrantes inferiores observamos  un nódulo de aspecto tumoral de 2,5 cm de diámetro máximo. Se toman cortes representativos del mismo 10 y 11. Con el nº12 Pezón. Tejido adiposo de prolongación axilar: Aislamos múltiples formaciones  nodulares que impresionan como ganglios linfáticos. Se toman cortes representativos de los mismos con los números 13 al 21 (los dos últimos corresponden al nivel II).TOTAL : 21BHE."
    spans = []
    doc = nlp(text)
    for ent in doc.ents:
        span = {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        spans.append(span)

    assert (
        expected_span == spans
    ), f"Generated spans: {span}\n and expected dict: {expected_span}\n are different"
