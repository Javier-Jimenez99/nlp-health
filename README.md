## Configuración del entorno
Para construir el enviroment de conda hay que ejecutar el siguiente código.
```
conda env create -f env.yml
conda activate nlp-health
pip install -r requirements.txt
```
Para actualizar requirements.txt: `pip-compile --extra=dev`
# Estructuración de diagnóstigos patológicos en formularios

## Librería
### form
- `config.py`:fichero de configuración que contiene el diccionario de clases de intervención.
#### Módulos
- `string_matching` contiene funciones encargadas del procesamiento de strings.

- `parse_form` contiene las funciones utilizadas para parsear el primer tipo de formularios.
    - Ejemplo: 
    ```
    TIPO DE INTERVENCIÓN:  
    TUMORECTOMÍA Y BSGC

    NÚMERO DE BIOPSIAS / CITOLOGÍAS PREVIAS: H. DE CEUTA 

    1- CARACTERÍSTICAS DEL TUMOR:

    - TAMAÑO:  1,7   cm     INDETERMINABLE (fragmentado, artefactado)        LOCALIZACIÓN: UC INF 

    - CARACTER:  INFILTRANTE 
    - MÚLTIPLE:    NO

    - 
    -INFILTRACIÓN DEL MARGEN QUIRÚRGICO / MÁRGENES DE LA BIOPSIA:

    LIBRE (a más de 1cm):  INCLUYENDO LAS AMPLIACIONES DE MUSCULO PECTORAL.

    - TIPO HISTOLÓGICO:  
    CARCINOMA DUCTAL INFILTRANTE

    ...
    ```

- `parse_form_2` contiene las funciones utilizadas para parsear el segundo tipo de formularios.
    - Ejemplo: 
    ```
    TIPO DE INTERVENCIÓN:  Exéresis

    LOCALIZACIÓN:  Región inframamaria

    1. TIPO TUMORAL
    Melanoma maligno extensión superficial

    2. NIVEL Y PROFUNDIDAD
    Nivel de Clark  II
    Espesor de Breslow  0,8 mm.

    ...
    ```

- `interventions.py` agrupa todas las funciones relacionadas con el procesamiento de las intervenciones.

#### Scripts
1. `xls_raw_preprocessing`: contiene funciones para procesar los archivos crudos. Se ejecuta para unificar, filtrar y procesar los estudios dentro de los archivos `xls`.
    - Input: ruta a la carpeta que contiene archivos `xls` de diferentes años. 
    - Output: archivo `Biopsias_HUPM_all.csv` que unifica todos los arhivos eliminando las apariciones extras de estudios y seleccionando los elementos que tienen formularios, y `parsed_forms_raw` que contiene los formularios parseados. 

2. `generate_dataset`: crea el dataset para entrenar un modelo para cada uno de los campos del formulario. Por ahora los campos incluidos son: `intervention`, `is_multiple`, `caracter`, `histological_type` and `histological_degree`. 
    - Input: ruta a `Biopsias_HUPM_all.csv` y `parsed_forms_raw`.
    - Output: DataFrame que contiene el dataset necesario para enternar un modelo de cada campo del formulario.

### multilabel classification
- `multilabel_classification.py`: contiene las funciones necesarias para entrenar, validar y estudiar los resultados de un modelo de clasificación multilabel:
    - Input: dataset para entrenar el modelo.
    - Output: modelo entrenado y resultados que este produce.

# Clasificación de diagnósticos patológicos en formularios con códigos SNOMED

### Recursos
En el caso de querer realizar un entrenamiento sencillo o inferencia, es recomendable hacer uso de la clase ```SNOMED_BERT_model.py```. Esta incluye todo lo necesario para funcionar correctamente, sin la necesidad de utilizar un cuaderno ```jupyter```.


# Anotación
Será necesario clonar https://github.com/Komorebi-AI/prodigy_annotator, ya que contiene los archivos necesarios para lanzar una herramienta de anotación.

Primero habrá que instalar prodigy desde https://github.com/Komorebi-AI/prodigy_annotator/nightly: `pip install prodigy-1.11.0a8-cp36.cp37.cp38.cp39-cp36m.cp37m.cp38.cp39-linux_x86_64.whl`

Este repositorio cuenta con varias herramientas de anotación para los datos de biopsias. Todas tienen un archivo de configuración, para definir parámetros como: la base de datos para guardar los resultados, las etiquetas utilizadas o el puerto en el que será lanzado. 

**Importante:** cambiar el parámetro de localización de la base de datos (`db_settings`).

También existe un archivo de instrucciones para dar una explicación clara al anotador y un script de bash para lanzar la herramienta. Este script recibe como parámetro la ruta al archivo que contiene los datos que van a ser anotados.
 
### Histological type
Esta herramienta se utiliza para anotar la clasificación del tipo histológico. En este caso, como son muchas clases se ha añadido un dropdown desde el que el anotador podrá seleccionar la opción correcta.

Los datos para anotar se encuentran en `nlp-health/data/clean/prodigy/to_annotate/nan_histological_type_prodigy.jsonl`. Se trata de un archivo `JSON lines` en el que cada línea se corresponde con el diagnostico de un estudio específico.

Los archivos de configuración, instrucciones y el script de ejecución están en el repositorio de `prodigy_annotator` mencionado anteriormente. 

- **Configuración:** `prodigy_annotator/config_files/histological_type/prodigy.json`
- **Instrucciones:** `prodigy_annotator/instructions/instructions_histological_type_classification.html`
- **Script ejecución:** `prodigy_annotator/run_scripts/run_histological_degree_classification.sh`

Para lanzar la herramienta de anotación se ejecuta:
```
nohup bash run_scripts/run_histological_degree_classification.shrun_.sh DIR_TO_ANNOTATION_DATA &
```

Finalmente, para extraer los datos anotados de la base de datos se debe ejecutar:
```
export PRODIGY_HOME="./config_files/histological_type"
prodigy db-out histological_type > ./annotations.jsonl
```

### Tumor size
Esta herramienta se utiliza para anotar mediante NER el tamaño del tumor.

Los datos para anotar se encuentran en `nlp-health/data/clean/prodigy/to_annotate/measure_ner.jsonl`. Se trata de un archivo `JSON lines` en el que cada línea se corresponde con todo el texto de un estudio y el inicio y final de medidas detectadas automáticamente. Ejemplo:
```
{   
    "text": "\nDatos clínicos: ##ENTRY_1## Carcinoma ductal infiltrante de mama izquierda...", 
    "estudio": "##ENTRY_1## B10-00020", 
    "spans": [
        {"start": 430, "end": 434, "label": "MEASURE"}, 
        {"start": 444, "end": 450, "label": "MEASURE"},
        ...
    ]
}
```
Esto hace que a la hora de anotar aparezcan señaladas automáticamente todas las medidas del texto, para que el anotador solo tenga que seleccionar alguna de ellas.

Los archivos de configuración, instrucciones y el script de ejecución están en el repositorio de `prodigy_annotator` mencionado anteriormente. 

- **Configuración:** `prodigy_annotator/config_files/tumor_size/prodigy.json`
- **Instrucciones:** `prodigy_annotator/instructions/instructions_ner.html`
- **Script ejecución:** `prodigy_annotator/run_scripts/run_tumor_size_ner.sh`

Para lanzar la herramienta de anotación se ejecuta:
```
nohup bash run_scripts/run_tumor_size_ner.sh DIR_TO_ANNOTATION_DATA &
```

Finalmente, para extraer los datos anotados de la base de datos se debe ejecutar:
```
export PRODIGY_HOME="./config_files/tumor_size"
prodigy db-out ner_measures> ./annotations.jsonl
```

# Streamlit app
Se ha creado una demo para ver el funcionamiento de los modelos. Primero hay que tener los modelos entrenados, si no es el caso se ha creado un script que se encarga de esto:
```
python biopsias/generate_models.py
```
La ruta de guardado de los modelos se puede cambiar en el archivo `config.py`.

Una vez se tengan los modelos se lanzará la aplicación ejecutando:
```
streamlit run biopsias/streamlit_app/app.py
```