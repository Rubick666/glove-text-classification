# Sentiment Analysis with Pretrained GloVe Embeddings

This project implements binary sentiment classification on the IMDB movie reviews dataset using a neural network with pretrained GloVe word embeddings. The workflow includes data loading, preprocessing, embedding matrix preparation, model building, training, and evaluation. The code is provided as a Jupyter Notebook and is compatible with Google Colab.

## Features
- Loads and preprocesses the IMDB movie reviews dataset
- Tokenizes and pads text data
- Loads pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors
- Prepares an embedding matrix for Keras
- Builds a neural network with a non-trainable embedding layer
- Trains and evaluates the model

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy

## How to Run
1. Download and extract the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and the [GloVe embeddings (glove.6B.100d.txt)](https://nlp.stanford.edu/data/glove.6B.zip).
2. Update the paths in the notebook to point to your local copies of the datasets.
3. Open the notebook (`Pretrained_GloVe_text_classification.ipynb`) in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
4. Run all cells in order.
5. (Optional) If running locally, first install the dependencies listed in `requirements.txt`:

    ```
    pip install -r requirements.txt
    ```

## Results
- The model achieves strong binary classification performance on the IMDB dataset.
- Training and validation accuracy/loss curves are shown in the notebook.

## Dataset & Embeddings
- [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/)

## Author
- [Amirfarhad](https://github.com/Rubick666)
