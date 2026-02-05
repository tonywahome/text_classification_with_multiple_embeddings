# GRU Model Training Guide

## Text Classification with Multiple Word Embeddings

This guide explains how to train and evaluate GRU models with different word embeddings for Amazon Food Reviews classification.

---

## 📋 Overview

The notebook `01_eda.ipynb` now contains a complete implementation for:

- **Data preprocessing** and stratified splitting
- **Multiple embedding techniques**: TF-IDF, Word2Vec (CBOW & Skip-gram), FastText
- **Bidirectional GRU models** for sequence-based embeddings
- **Dense neural networks** for TF-IDF features
- **Systematic experiment tracking** with JSON logging
- **Hyperparameter tuning** framework
- **Comprehensive evaluation** with metrics and visualizations

---

## 🚀 Quick Start

### Step 1: Run Baseline Experiments

Execute the following cells in order:

1. **Cell: "PART 1: Data Preparation"**
   - Loads 50,000 reviews (or modify `SAMPLE_SIZE` for full dataset)
   - Performs stratified train/val/test split (70%/10%/20%)
   - Preprocesses all text data
   - Computes class weights for imbalance handling

2. **Cell: "PART 2: TF-IDF + Dense Model"**
   - Generates TF-IDF features (5,000 dimensions)
   - Trains dense neural network
   - Evaluates and logs results
   - **Expected time**: ~10-15 minutes

3. **Cell: "PART 3: Word2Vec CBOW + GRU"**
   - Trains Word2Vec CBOW embeddings (100d)
   - Builds Bidirectional GRU model
   - Trains and evaluates
   - **Expected time**: ~20-30 minutes

4. **Cell: "PART 4: Word2Vec Skip-gram + GRU"**
   - Trains Word2Vec Skip-gram embeddings (100d)
   - Builds Bidirectional GRU model
   - Trains and evaluates
   - **Expected time**: ~20-30 minutes

5. **Cell: "PART 5: FastText + GRU"** (Optional)
   - Trains FastText embeddings with subword information
   - Builds Bidirectional GRU model
   - Trains and evaluates
   - **Expected time**: ~25-35 minutes

6. **Cell: "PART 6: Comparative Analysis"**
   - Displays summary table of all experiments
   - Generates comparison visualizations
   - Saves results to CSV

---

## 🎯 Understanding the Results

### Experiment Tracking

All experiments are automatically logged to:

- **File**: `results/experiments.json`
- **Contains**:
  - Experiment ID and timestamp
  - Hyperparameters used
  - Training history (loss/accuracy per epoch)
  - Evaluation metrics (accuracy, F1-score, precision, recall)
  - Per-class performance
  - Training time
  - Confusion matrix

### Metrics to Compare

When evaluating embeddings, focus on:

1. **Accuracy**: Overall correct predictions (but can be misleading with class imbalance)
2. **F1-Score (Macro)**: Average F1 across all classes (treats classes equally)
3. **F1-Score (Weighted)**: Weighted by class support (better for imbalanced data)
4. **Per-class Performance**: Which ratings (1-5 stars) are hardest to predict?
5. **Training Time**: Computational efficiency consideration

### Expected Performance Range

Based on similar sentiment classification tasks:

- **TF-IDF + Dense**: Accuracy ~55-65%
- **Word2Vec CBOW**: Accuracy ~58-68%
- **Word2Vec Skip-gram**: Accuracy ~60-70%
- **FastText**: Accuracy ~62-72%

_Note: Actual performance depends on data quality and hyperparameters_

---

## 🔧 Hyperparameter Tuning

### Step 2: Tune Best Models

After reviewing baseline results, tune the top 2 performing embeddings:

1. **Review** the summary table from PART 6
2. **Identify** the 2 best embeddings by F1-score
3. **Uncomment** the hyperparameter tuning cell in PART 7
4. **Modify** to tune your selected embeddings

Example (if Word2Vec CBOW performed best):

```python
best_config_cbow, best_model_cbow, tuned_results_cbow, tuning_df_cbow = hyperparameter_tuning(
    embedding_type='word2vec_cbow',
    X_train=X_train_cbow,
    X_val=X_val_cbow,
    X_test=X_test_cbow,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    embedding_matrix=embedding_matrix_cbow,
    vocab_size=vocab_size_cbow,
    hp_space=hp_search_space,
    base_config=TRAIN_CONFIG,
    class_weights=class_weight_dict
)
```

**Expected time**: ~2-4 hours per embedding (8 configurations tested)

### Hyperparameter Search Space

Current search space (modify in cell if needed):

```python
{
    'GRU_UNITS': [64, 128, 256],
    'DROPOUT_RATE': [0.3, 0.5],
    'BATCH_SIZE': [32, 64],
    'LEARNING_RATE': [0.001, 0.0001]
}
```

**Total combinations**: 24  
**Tested**: 8 random combinations (for efficiency)

---

## 📊 Generated Outputs

### Files Created

1. **Models** (`models/`)
   - `tfidf_dense_best.h5`
   - `word2vec_cbow_gru_best.h5`
   - `word2vec_skipgram_gru_best.h5`
   - `fasttext_gru_best.h5`

2. **Results** (`results/`)
   - `experiments.json` - Complete experiment log
   - `experiment_summary.csv` - Summary table
   - `*_hyperparameter_tuning.csv` - Tuning results per embedding

3. **Plots** (`results/plots/`)
   - `*_history.png` - Training/validation curves
   - `*_confusion_matrix.png` - Confusion matrices
   - `model_comparison.png` - Comparative bar charts

### Visualizations

Each model generates:

- **Training History**: Loss and accuracy curves showing convergence
- **Confusion Matrix**: Which classes are confused with each other
- **Comparison Charts**: Accuracy, F1-score, training time, and epochs across all models

---

## 🎓 Academic Report Preparation

### Key Findings to Document

1. **Dataset Characteristics**
   - Class imbalance (heavily skewed toward 5-star reviews)
   - Text length statistics (mean ~79 words)
   - Data quality issues (HTML tags, variable length)

2. **Embedding Comparison**
   - Which embedding achieved highest accuracy?
   - Which had best F1-score (macro vs weighted)?
   - Trade-offs between performance and training time

3. **Model Architecture**
   - Bidirectional GRU advantages for sentiment analysis
   - Impact of dropout and recurrent dropout
   - Why 2 layers vs 1 or 3?

4. **Class Imbalance Handling**
   - Effectiveness of class weights
   - Per-class performance differences
   - Are minority classes (1-2 stars) being learned?

5. **Hyperparameter Impact**
   - Which hyperparameters had biggest effect?
   - Optimal configuration for your best embedding
   - Relationship between model capacity and overfitting

### Statistical Significance

For academic rigor, consider:

- Running multiple random seeds (3-5 runs)
- Computing confidence intervals
- Statistical tests (e.g., paired t-test) between embeddings

---

## ⚙️ Configuration Options

### Modify Sample Size

For faster experimentation:

```python
TRAIN_CONFIG['SAMPLE_SIZE'] = 10000  # Smaller sample
```

For full dataset:

```python
TRAIN_CONFIG['SAMPLE_SIZE'] = None  # Use all reviews
```

### Adjust Training Epochs

Default: 30 epochs with early stopping

To increase:

```python
TRAIN_CONFIG['EPOCHS'] = 50
```

### Change Model Architecture

Modify GRU units:

```python
TRAIN_CONFIG['GRU_UNITS'] = 256  # More capacity
```

Adjust dropout:

```python
TRAIN_CONFIG['DROPOUT_RATE'] = 0.3  # Less regularization
```

### Sequence Length

Current: 200 tokens (captures most reviews)

For longer texts:

```python
TRAIN_CONFIG['MAX_SEQUENCE_LENGTH'] = 300
```

---

## 🐛 Troubleshooting

### Memory Issues

If running out of memory:

1. Reduce `SAMPLE_SIZE` to 20,000 or 10,000
2. Reduce `BATCH_SIZE` to 16
3. Reduce `MAX_SEQUENCE_LENGTH` to 150
4. Use Google Colab with GPU for larger experiments

### Slow Training

To speed up:

1. Reduce `EPOCHS` to 20
2. Increase `BATCH_SIZE` to 64 (if memory allows)
3. Use smaller `SAMPLE_SIZE` for initial testing
4. Skip FastText (slowest embedding to train)

### Poor Performance

If accuracy is low (<50%):

1. Check class weights are being used
2. Verify preprocessing is working (print sample)
3. Increase model capacity (`GRU_UNITS` = 256)
4. Train longer or reduce early stopping patience
5. Try different learning rate (0.0005 or 0.002)

### Import Errors

If missing packages:

```bash
pip install gensim nltk beautifulsoup4 tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## 📈 Next Steps After Training

1. **Analyze Results**
   - Which embedding performed best overall?
   - Which is best for minority classes (1-2 stars)?
   - What's the accuracy-time trade-off?

2. **Error Analysis**
   - Examine misclassified examples
   - Are certain review types consistently wrong?
   - What patterns do confused classes share?

3. **Advanced Techniques** (Optional)
   - Try pre-trained GloVe embeddings
   - Ensemble multiple models
   - Fine-tune BERT or RoBERTa for comparison
   - Add attention mechanisms

4. **Academic Report**
   - Document methodology thoroughly
   - Include all metrics and visualizations
   - Discuss limitations and future work
   - Cite relevant papers (Word2Vec, FastText, GRU)

---

## 📚 Key References

**Embeddings:**

- Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- Bojanowski et al. (2017) - "Enriching Word Vectors with Subword Information" (FastText)
- Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"

**RNN Architectures:**

- Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder" (GRU)
- Schuster & Paliwal (1997) - "Bidirectional Recurrent Neural Networks"

**Sentiment Analysis:**

- Zhang et al. (2018) - "Deep Learning for Sentiment Analysis: A Survey"
- Socher et al. (2013) - "Recursive Deep Models for Semantic Compositionality"

---

## ✅ Checklist

Before finalizing your project:

- [ ] Run all baseline experiments (4 embeddings)
- [ ] Review and compare results
- [ ] Perform hyperparameter tuning on best 2 embeddings
- [ ] Generate all visualizations
- [ ] Save all models and results
- [ ] Document findings in academic report
- [ ] Include statistical significance tests
- [ ] Cite all relevant papers
- [ ] Discuss limitations and future work
- [ ] Proofread and format report

---

**Good luck with your experiments! 🚀**

For questions or issues, review the notebook cells for comments and documentation.
