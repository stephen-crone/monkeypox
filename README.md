![Alt](/monkeypox-small.jpg "monkeypox-image")

# Monkeypox misinformation detector project

This repository contains source code and other materials developed as part of a machine learning project on **monkeypox misinformation detection**. The project itself involved the following main outputs:

1. The creation of an [annotated dataset of monkeypox](https://www.kaggle.com/datasets/stephencrone/monkeypox) related Twitter misinformation.
2. The development and evaluation of a [transformer model](https://huggingface.co/smcrone/monkeypox-misinformation) trained to accurately classify monkeypox misinformation.
3. The proposal of a measure designed to identify misinformation 'superspreaders' on Twitter.
4. The development of a Streamlit web in which the aforementioned transformer model and superspreader measure could be integrated.

Links to the aforementioned dataset and transformer model are provided above. The contents of this repository are as follows:

1. The Python script used to gather and preprocess tweets (**tweet-collector.py**).
2. The Jupyter notebook used to conduct model experiments (**model-experiments.ipynb**).
3. The experimental data itself (stored in a series of sub-folders labelled according to learning paradigm).
4. The Python script used to optimise hyperparameters for the superspreader function (**ss-hyperparam-calc.py**).
5. The source code for the Streamlit web app.
