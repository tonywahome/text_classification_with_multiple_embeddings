# Text Classification with Multiple Word Embeddings

## Project Overview

This project implements and compares GRU-based text classification models using multiple word embedding techniques on the Amazon Fine Food Reviews dataset. The goal is to predict review ratings (1-5 stars) and analyze which embedding methods work best for sentiment classification.

## Embedding Techniques Implemented

1. **TF-IDF** - Traditional statistical representation
2. **Word2Vec Skip-gram** - Predictive context-based embeddings
3. **Word2Vec CBOW** - Continuous Bag of Words embeddings
4. **GloVe** - Global word co-occurrence vectors
5. **FastText** - Subword-aware embeddings

## Project Structure

```
text_classification_with_multiple_embeddings/
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_tfidf_gru.ipynb                  # TF-IDF + GRU
│   ├── 03_word2vec_skipgram_gru.ipynb      # Skip-gram + GRU
│   ├── 04_word2vec_cbow_gru.ipynb          # CBOW + GRU
│   ├── 05_glove_gru.ipynb                  # GloVe + GRU
│   ├── 06_fasttext_gru.ipynb               # FastText + GRU
│   └── 07_comparative_analysis.ipynb       # Results comparison
├── src/
│   ├── preprocessing.py                    # Text preprocessing utilities
│   ├── data_loader.py                      # Data loading and splitting
│   ├── model_builder.py                    # GRU model architectures
│   ├── embeddings.py                       # Embedding generation
│   ├── trainer.py                          # Training utilities
│   └── evaluator.py                        # Evaluation metrics
├── models/                                 # Saved trained models
├── results/                                # Performance metrics and plots
├── report/                                 # Academic report
├── Reviews.csv                             # Dataset
└── requirements.txt                        # Dependencies

```

## Dataset

**Amazon Fine Food Reviews**

- Source: Amazon product reviews
- Size: ~568,000 reviews
- Target: Rating scores (1-5 stars)
- Text Fields: Review text and summary
- Domain: Food products

## Methodology

### Phase 1: Exploratory Data Analysis

- Class distribution analysis
- Text length patterns
- Vocabulary statistics
- Data cleaning requirements

### Phase 2: Model Development

Each embedding technique follows this pipeline:

1. Text preprocessing (HTML removal, tokenization)
2. Embedding generation
3. GRU model training
4. Hyperparameter tuning
5. Performance evaluation

### Phase 3: Comparative Analysis

- Unified performance metrics
- Statistical significance testing
- Visualization of results
- Discussion with research citations

## GRU Architecture

```python
Input Layer → Embedding Layer → Bidirectional GRU → Dropout → Dense → Softmax (5 classes)
```

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/tonywahome/text_classification_with_multiple_embeddings.git
cd text_classification_with_multiple_embeddings
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

4. **Run notebooks in order**
   Start with `01_eda.ipynb` and proceed sequentially through the embedding-specific notebooks.

## Results

Results will be documented in the comparative analysis notebook and final report, including:

- Accuracy, Precision, Recall, F1-Score per embedding
- Confusion matrices
- Training curves
- Statistical analysis
- Embedding space visualizations

## References

Key papers and resources:

- Mikolov et al. (2013) - Word2Vec
- Pennington et al. (2014) - GloVe
- Bojanowski et al. (2017) - FastText
- Cho et al. (2014) - GRU Networks

## License

MIT License

## Authors

Academic research project for text classification comparison study.
