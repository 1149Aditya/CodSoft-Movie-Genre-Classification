# CodSoft-Movie-Genre-Classification


This project builds a machine learning model that predicts the genre of a movie based on its plot summary or other textual information. Two main approaches are explored:

1. **TF-IDF + Naive Bayes Classifier**
2. **Word Embeddings + LSTM Neural Network**

## Project Overview

This project tackles the challenge of movie genre classification as a multi-class text classification task. The goal is to predict a movie's genre using its plot summary. The project employs two distinct approaches: 
- **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization for feature extraction followed by the **Naive Bayes** classifier.
- **Word Embeddings** for dense word representations combined with an **LSTM (Long Short-Term Memory)** neural network for sequence-based classification.

## Dataset

You can use publicly available datasets like the [IMDB](https://www.imdb.com/interfaces/) or [TMDB](https://www.themoviedb.org/documentation/api) datasets for movie plot summaries and genre labels.

**Features:**
- **Plot Summary**: A brief summary of the movie’s storyline (text).
- **Genre**: The target genre label (e.g., Comedy, Drama, Sci-Fi, etc.).

## Approaches

### 1. TF-IDF + Naive Bayes

- The plot summaries are cleaned and vectorized using **TF-IDF**.
- A **Naive Bayes classifier (MultinomialNB)** is trained to predict the genre.
- Performance is evaluated using accuracy, precision, recall, and F1-score.

### 2. Word Embeddings + LSTM

- The plot summaries are tokenized and transformed into word embeddings using Keras' **Tokenizer** and **Embedding** layers.
- An **LSTM neural network** is built to learn from the sequence of words in the plot summaries.
- Additional layers like **Dropout** are used to prevent overfitting, and **EarlyStopping** is employed to stop training when validation loss no longer improves.



### Required Libraries

Here’s the full list of libraries required for this project:
- **pandas**: Data manipulation.
- **numpy**: Numerical operations.
- **matplotlib, seaborn**: For data visualization.
- **re, string**: Regular expressions and string manipulation for text preprocessing.
- **nltk**: Natural language processing (stopwords, stemming).
- **sklearn**: Machine learning (TF-IDF, Naive Bayes, train-test split, evaluation metrics).
- **tensorflow/keras**: For the LSTM neural network.


### Evaluation:

- After training, the model’s performance will be evaluated based on metrics like accuracy, precision, recall, and F1-score.


The performance metrics can be adjusted based on the specific dataset used.
