from setuptools import setup, find_packages

setup(
    name="biopsias",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "pandas>=1.2.4",
        "numpy>=1.20.3",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.12.2",
        "torch>=1.8.1",
        "transformers>=4.5.1",
        "datasets>=1.8.0",
        "xlrd>=2.0.1",
        "matplotlib>=3.4.2",
        "scikit-learn>=0.24.2",
        "spacy>=3.0.6",
        "spaczz>=0.5.2",
        "nltk>=3.6.2",
        "streamlit>=0.88.0",
        "seqeval>=1.2.2",
        "pytokenizations>=0.8.3",
        "jsonlines>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pip-tools",
            "pytest>=6.2.3",
            "black>=20.8b1",
            "isort",
            "flake8",
        ]
    },
)
