DIC_CLASSES = {
    "mrm": {
        "mastectomia radical modificada",
        "mrm",
        "m.r.m.",
    },
    "mr": {"mastectomia radical"},
    "ms": {"mastectomia simple", "mastectomia"},
    "tumorectomia": {"tumorectomia"},
    "bsgc": {
        "bsgc",
        "biopsia selectiva ganglio centinela",
        "bsg centinela",
        "bs gc",
        "biopsia ganglio centinela",
        "bsg",
    },
    "linfadenectomia": {"linfadenectomia"},
    "cuadrantectomia": {"cuadrantectomia"},
    # "ganglio intramamario": {"ganglio intramamario", "ganglio mamaria interna"},
    "invalid": {"exeresis", "amputacion"},
    "": {"extirpacion", "ampliacion margenes"},
}

tumor_size_labels = ["B-TUMOR_SIZE", "I-TUMOR_SIZE", "O"]

hardcoded_invalid_studies = ["B17-10176"]

all_biopsias_path = "data/clean/Biopsias_HUPM_all.parquet"
parsed_forms_path = "data/clean/parsed_forms_raw.json"

all_snomed_data_path = "data/raw/Biopsias_HUPM_2010-2018_mor_codes-v1.csv"

figures_path = "reports"

# MODELS PATHS
histological_type_model_path = "Models/form/histological_type.joblib"
histological_degree_model_path = "Models/form/histological_degree.joblib"
is_cdi_model_path = "Models/form/is_cdi.joblib"
intervention_type_model_path = "Models/form/intervention_type.joblib"
tumor_size_automatic_model_path = "Models/form/tumor_size_automatic_annotated"
tumor_size_manual_model_path = "Models/form/tumor_size_manual_annotated"
morf_save_path = "Models/snomed/morf_all.joblib"
top_save_path = "Models/snomed/top_all.joblib"
