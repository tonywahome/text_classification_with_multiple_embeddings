# Comparative Analysis of Word Embedding Techniques for Text Classification: A GRU-based Approach

**Research Project: Amazon Fine Food Reviews Sentiment Classification**

---

## Abstract

This study presents a comprehensive comparative analysis of five different word embedding techniques applied to sentiment classification of Amazon Fine Food Reviews using Gated Recurrent Unit (GRU) neural networks. We evaluate TF-IDF, Word2Vec (Skip-gram and CBOW), GloVe, and FastText embeddings across multiple performance metrics including accuracy, precision, recall, and F1-score. Our analysis reveals the strengths and limitations of each embedding approach in capturing semantic relationships for sentiment analysis tasks. The findings contribute to understanding which embedding techniques are most suitable for review-based sentiment classification and provide insights into the trade-offs between computational efficiency and classification performance.

**Keywords**: Text Classification, Word Embeddings, GRU, Sentiment Analysis, Natural Language Processing

---

## 1. Introduction

### 1.1 Background

Text classification remains a fundamental task in Natural Language Processing (NLP), with applications ranging from sentiment analysis to document categorization (Mikolov et al., 2013). The quality of text representation significantly impacts classification performance, making the choice of embedding technique crucial for model success.

### 1.2 Research Questions

This study addresses the following research questions:

1. **RQ1**: How do different word embedding techniques compare in sentiment classification accuracy?
2. **RQ2**: What are the trade-offs between traditional statistical methods (TF-IDF) and neural embedding approaches?
3. **RQ3**: How do context-free embeddings (Word2Vec, GloVe) compare to subword-aware embeddings (FastText)?
4. **RQ4**: What is the impact of pre-trained versus domain-specific embeddings on classification performance?

### 1.3 Contributions

Our contributions include:

- Comprehensive comparison of five embedding techniques on a large-scale review dataset
- Analysis of embedding effectiveness in handling class imbalance
- Insights into computational trade-offs between embedding approaches
- Reproducible experimental framework for text classification research

---

## 2. Related Work

### 2.1 Word Embedding Techniques

**TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure that reflects word importance in documents relative to a corpus (Salton & Buckley, 1988). While interpretable and computationally efficient, TF-IDF lacks semantic understanding.

**Word2Vec**: Introduced by Mikolov et al. (2013), Word2Vec learns distributed representations through two architectures:

- **Skip-gram**: Predicts context words from target word
- **CBOW (Continuous Bag of Words)**: Predicts target word from context

These models capture semantic relationships through vector arithmetic (e.g., king - man + woman ≈ queen).

**GloVe (Global Vectors)**: Pennington et al. (2014) proposed GloVe, which combines global matrix factorization with local context window methods, capturing both global statistical information and local context.

**FastText**: Bojanowski et al. (2017) extended Word2Vec by incorporating subword information, enabling better handling of rare words and out-of-vocabulary (OOV) terms through character n-grams.

### 2.2 Recurrent Neural Networks for Text Classification

Gated Recurrent Units (GRUs), introduced by Cho et al. (2014), provide an efficient alternative to LSTMs for sequential data processing. GRUs use gating mechanisms to control information flow, making them particularly effective for text classification tasks (Chung et al., 2014).

### 2.3 Sentiment Analysis

Sentiment analysis on product reviews has been extensively studied (Pang & Lee, 2008; Liu, 2012). However, comprehensive comparisons of modern embedding techniques on large-scale review datasets remain limited.

---

## 3. Methodology

### 3.1 Dataset

**Amazon Fine Food Reviews Dataset**

- **Source**: Amazon product reviews (McAuley & Leskovec, 2013)
- **Size**: [TO BE FILLED: total number of reviews]
- **Classes**: 5 (1-5 star ratings)
- **Features**: Review text, summary, rating score
- **Challenge**: Significant class imbalance (heavily skewed toward positive reviews)

**Data Splits**:

- Training: 70%
- Validation: 10%
- Test: 20%

Stratified sampling ensures balanced class representation across splits.

### 3.2 Preprocessing Pipeline

1. **HTML Tag Removal**: Clean markup from review text
2. **Tokenization**: NLTK word tokenization
3. **Lowercasing**: Convert all text to lowercase
4. **Stopword Removal**: Remove common English stopwords (optional, tested both ways)
5. **Lemmatization**: WordNet lemmatization for base form reduction
6. **Sequence Padding**: Pad/truncate sequences to maximum length of 200 tokens

### 3.3 Embedding Generation

#### 3.3.1 TF-IDF

- **Max Features**: 5,000
- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Min Document Frequency**: 2
- **Max Document Frequency**: 0.95

#### 3.3.2 Word2Vec (Skip-gram & CBOW)

- **Embedding Dimension**: 100
- **Window Size**: 5
- **Min Word Frequency**: 2
- **Training Epochs**: 10
- **Algorithm**: Skip-gram (sg=1) or CBOW (sg=0)

#### 3.3.3 GloVe

- **Pre-trained Model**: glove.6B.100d
- **Source**: Stanford NLP
- **Vocabulary**: 400K words
- **Training Corpus**: Wikipedia 2014 + Gigaword 5

#### 3.3.4 FastText

- **Embedding Dimension**: 100
- **Window Size**: 5
- **Min Word Frequency**: 2
- **Character N-grams**: 3-6
- **Training Epochs**: 10

### 3.4 Model Architecture

#### GRU-based Classifier

```
Input Layer (max_length=200)
    ↓
Embedding Layer (dim=100 or pre-trained)
    ↓
Bidirectional GRU Layer 1 (units=128, recurrent_dropout=0.2)
    ↓
Dropout (rate=0.5)
    ↓
Bidirectional GRU Layer 2 (units=128, recurrent_dropout=0.2)
    ↓
Dropout (rate=0.5)
    ↓
Dense Layer (units=5, activation='softmax')
```

#### Dense Classifier (TF-IDF)

```
Input Layer (features=5,000)
    ↓
Dense Layer (units=512, activation='relu')
    ↓
Dropout (rate=0.5)
    ↓
Dense Layer (units=256, activation='relu')
    ↓
Dropout (rate=0.5)
    ↓
Dense Layer (units=128, activation='relu')
    ↓
Dropout (rate=0.5)
    ↓
Dense Layer (units=5, activation='softmax')
```

### 3.5 Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Max Epochs**: 50
- **Early Stopping**: Patience of 5 epochs on validation loss
- **Learning Rate Reduction**: Factor of 0.5, patience of 3 epochs
- **Class Weights**: Computed to handle class imbalance

### 3.6 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-Score**: Harmonic mean of precision and recall (macro and weighted)
- **Confusion Matrix**: Detailed error analysis
- **Training Time**: Computational efficiency comparison

---

## 4. Results

### 4.1 Overall Performance Comparison

[TABLE TO BE FILLED AFTER EXPERIMENTS]

| Embedding Technique | Accuracy | F1 (Weighted) | F1 (Macro) | Precision | Recall | Training Time |
| ------------------- | -------- | ------------- | ---------- | --------- | ------ | ------------- |
| TF-IDF              | X.XXXX   | X.XXXX        | X.XXXX     | X.XXXX    | X.XXXX | XX min        |
| Word2Vec Skip-gram  | X.XXXX   | X.XXXX        | X.XXXX     | X.XXXX    | X.XXXX | XX min        |
| Word2Vec CBOW       | X.XXXX   | X.XXXX        | X.XXXX     | X.XXXX    | X.XXXX | XX min        |
| GloVe               | X.XXXX   | X.XXXX        | X.XXXX     | X.XXXX    | X.XXXX | XX min        |
| FastText            | X.XXXX   | X.XXXX        | X.XXXX     | X.XXXX    | X.XXXX | XX min        |

### 4.2 Per-Class Performance

[DETAILED TABLES TO BE FILLED]

### 4.3 Confusion Matrices

[INSERT CONFUSION MATRIX VISUALIZATIONS]

### 4.4 Training Curves

[INSERT LOSS AND ACCURACY CURVES]

---

## 5. Discussion

### 5.1 Comparative Analysis

[TO BE FILLED AFTER EXPERIMENTS]

**Expected Insights**:

1. **TF-IDF Performance**: While computationally efficient, TF-IDF may struggle with semantic understanding, particularly for nuanced sentiment expressions.

2. **Word2Vec Skip-gram vs CBOW**: Skip-gram typically performs better on semantic tasks due to its focus on predicting context, while CBOW may be faster to train.

3. **Pre-trained GloVe**: Expected to show strong performance due to extensive pre-training, but may have limited coverage of domain-specific food terminology.

4. **FastText Advantages**: Should demonstrate superior OOV handling through subword embeddings, particularly valuable for misspellings and rare food product names common in reviews.

### 5.2 Class Imbalance Impact

The dataset's heavy skew toward positive reviews (4-5 stars) presents challenges for minority class prediction. Analysis should reveal:

- Whether class weights effectively mitigate imbalance
- Which embedding techniques better capture minority class patterns
- Trade-offs between overall accuracy and balanced class performance

### 5.3 Computational Considerations

- **Training Time**: TF-IDF fastest, FastText potentially slowest
- **Memory Requirements**: Pre-trained embeddings require less training time but more storage
- **Inference Speed**: All approaches suitable for production deployment

### 5.4 Practical Implications

For practitioners:

1. **Resource-Constrained Scenarios**: TF-IDF or CBOW may be preferable
2. **High-Accuracy Requirements**: FastText or pre-trained GloVe recommended
3. **Domain Adaptation**: Custom-trained Word2Vec/FastText on domain data
4. **OOV Handling**: FastText essential for noisy user-generated content

---

## 6. Limitations

1. **Dataset Specificity**: Results may not generalize to other domains beyond product reviews
2. **Computational Constraints**: Limited hyperparameter tuning due to computational budget
3. **Language Coverage**: Analysis limited to English reviews
4. **Temporal Bias**: Dataset represents historical reviews; language patterns may evolve
5. **Class Imbalance**: Despite mitigation strategies, minority classes remain underrepresented

---

## 7. Conclusion

This study provides a comprehensive comparison of five word embedding techniques for sentiment classification on Amazon reviews. Our findings contribute to understanding the trade-offs between different representation methods and inform embedding selection for text classification tasks.

**Key Takeaways**:

- [TO BE FILLED BASED ON RESULTS]
- No single embedding technique dominates across all metrics
- Embedding choice should balance performance requirements with computational constraints
- Subword-aware embeddings (FastText) offer advantages for noisy, user-generated content

### 7.1 Future Work

1. **Transformer-based Models**: Compare with BERT, RoBERTa, and other transformer embeddings
2. **Ensemble Approaches**: Combine multiple embeddings for improved performance
3. **Domain Adaptation**: Transfer learning from general to domain-specific embeddings
4. **Multi-task Learning**: Joint training on sentiment and aspect extraction
5. **Cross-lingual Analysis**: Extend comparison to multilingual review datasets

---

## 8. References

Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. _Transactions of the Association for Computational Linguistics_, 5, 135-146.

Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_.

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. _arXiv preprint arXiv:1412.3555_.

Liu, B. (2012). Sentiment analysis and opinion mining. _Synthesis lectures on human language technologies_, 5(1), 1-167.

McAuley, J., & Leskovec, J. (2013). Hidden factors and hidden topics: understanding rating dimensions with review text. In _Proceedings of the 7th ACM conference on Recommender systems_ (pp. 165-172).

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_.

Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. _Foundations and Trends in Information Retrieval_, 2(1–2), 1-135.

Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vectors for word representation. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ (pp. 1532-1543).

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. _Information processing & management_, 24(5), 513-523.

---

## Appendix A: Hyperparameters

[COMPLETE HYPERPARAMETER TABLES]

## Appendix B: Code Repository

Complete implementation available at: [GitHub Repository URL]

---

**Acknowledgments**: [TO BE FILLED]

**Ethics Statement**: This research uses publicly available Amazon review data. No personally identifiable information was collected or analyzed.
