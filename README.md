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
│   ├── .ipynb_checkpoints/
│   ├── 01_eda.ipynb
│   ├── 02_tfidf_dense.ipynb
│   └── complete_text_classification_project.ipynb
├── report/
│   └── academic_report_template.md
├── .gitattributes
├── .gitignore
├── hashes.txt
├── QUICK_EXECUTION.md
├── README.md
├── requirements.txt
└── TRAINING_GUIDE.md
```

## Dataset

**Amazon Fine Food Reviews**

- **Source**: Amazon product reviews
- **Size**: ~568,000 reviews
- **Target**: Rating scores (1-5 stars)
- **Text Fields**: Review text and summary
- **Domain**: Food products
- **Challenge**: Highly imbalanced dataset with majority positive reviews

## Methodology

### Phase 1: Exploratory Data Analysis

- Class distribution analysis
- Text length patterns
- Vocabulary statistics
- Data cleaning requirements
- Feature engineering opportunities

### Phase 2: Model Development

Each embedding technique follows this pipeline:

1. **Text preprocessing** (HTML removal, tokenization, lemmatization)
2. **Embedding generation** (domain-specific training or pre-trained models)
3. **GRU model training** (bidirectional architecture with dropout)
4. **Hyperparameter tuning** (grid search for optimal parameters)
5. **Performance evaluation** (comprehensive metrics and analysis)

### Phase 3: Comparative Analysis

- Unified performance metrics (accuracy, precision, recall, F1-score)
- Statistical significance testing
- Visualization of results (confusion matrices, training curves)
- Discussion with research citations
- Computational efficiency comparison

## GRU Architecture
```python
Input Layer → Embedding Layer → Bidirectional GRU → Dropout → Dense → Softmax (5 classes)
```

**Architecture Details:**
- Embedding dimension: 100
- GRU units: 96 (bidirectional)
- Dropout rate: 0.3
- Batch size: 128
- Optimizer: Adam (lr=0.001)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/tonywahome/text_classification_with_multiple_embeddings.git
cd text_classification_with_multiple_embeddings
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- TensorFlow >= 2.10.0
- scikit-learn >= 1.0.0
- gensim >= 4.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- nltk >= 3.6.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run notebooks in order

Start with `01_eda.ipynb` and proceed sequentially through the embedding-specific notebooks. Each notebook is self-contained and includes detailed explanations of the methodology.

## Results

Results will be documented in the comparative analysis notebook and final report, including:

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score per embedding
- **Confusion Matrices**: Detailed error analysis for each model
- **Training Curves**: Loss and accuracy progression
- **Statistical Analysis**: Significance testing between models
- **Embedding Space Visualizations**: t-SNE and PCA projections
- **Computational Efficiency**: Training time and resource usage comparison

## Key Features

-  Systematic comparison of 5 different embedding techniques
-  Reproducible experimental framework with fixed random seeds
-  Comprehensive preprocessing pipeline
-  Class imbalance handling with weighted loss
-  Early stopping and learning rate scheduling
-  Detailed logging and experiment tracking
-  Professional visualizations and reporting

## Expected Outcomes

This project demonstrates:
- Trade-offs between traditional (TF-IDF) and neural embeddings
- Impact of subword information (FastText) on review classification
- Computational efficiency vs. performance balance
- Effectiveness of different architectures on imbalanced data

## References

Key papers and resources:

- **Mikolov et al. (2013)** - Word2Vec: Efficient Estimation of Word Representations in Vector Space
- **Pennington et al. (2014)** - GloVe: Global Vectors for Word Representation
- **Bojanowski et al. (2017)** - FastText: Enriching Word Vectors with Subword Information
- **Cho et al. (2014)** - GRU Networks: Learning Phrase Representations using RNN Encoder-Decoder

## Future Work

- Integration of transformer-based models (BERT, RoBERTa)
- Ensemble methods combining multiple embeddings
- Cross-domain transfer learning experiments
- Multi-task learning for sentiment and aspect detection

## License

MIT License

## Authors

Academic research project for text classification comparison study.

## Acknowledgments

- Amazon for providing the Fine Food Reviews dataset
- Open-source community for excellent NLP libraries
- Academic researchers whose foundational work made this project possible

---