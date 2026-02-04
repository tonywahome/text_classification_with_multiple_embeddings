# Quick Execution Instructions

## Run All Baseline Experiments in Sequence

Open [01_eda.ipynb](notebooks/01_eda.ipynb) and execute these cells:

### 1️⃣ Setup (Run Once)
```
✅ Cell: Import libraries (#VSC-b6ab101a)
✅ Cell: Training configuration (#VSC-2e440eb5)
✅ Cell: Data loading functions (#VSC-1ee40d6c)
✅ Cell: EmbeddingGenerator (#VSC-874d04fc)
✅ Cell: Model builders (#VSC-c69f2404)
✅ Cell: Training & evaluation (#VSC-05fc2ba9)
✅ Cell: Experiment tracker (#VSC-d35fcf82)
```

### 2️⃣ Load Data
```
▶️ Cell: "PART 1: Data Preparation" (#VSC-690f9817)
   ⏱️ Time: ~30 seconds
   📊 Output: 35,000 train / 5,000 val / 10,000 test samples
```

### 3️⃣ Train Models (Run each experiment)

**Experiment 1: TF-IDF**
```
▶️ Cell: "PART 2: TF-IDF + Dense Model" (#VSC-a8ddd4ae)
   ⏱️ Time: ~10-15 minutes
   💾 Saves: models/tfidf_dense_best.h5
```

**Experiment 2: Word2Vec CBOW**
```
▶️ Cell: "PART 3: Word2Vec CBOW + GRU" (#VSC-19c9b74c)
   ⏱️ Time: ~20-30 minutes
   💾 Saves: models/word2vec_cbow_gru_best.h5
```

**Experiment 3: Word2Vec Skip-gram**
```
▶️ Cell: "PART 4: Word2Vec Skip-gram + GRU" (#VSC-839ae2ce)
   ⏱️ Time: ~20-30 minutes
   💾 Saves: models/word2vec_skipgram_gru_best.h5
```

**Experiment 4: FastText (Optional)**
```
▶️ Cell: "PART 5: FastText + GRU" (#VSC-7666d392)
   ⏱️ Time: ~25-35 minutes
   💾 Saves: models/fasttext_gru_best.h5
```

### 4️⃣ Compare Results
```
▶️ Cell: "PART 6: Comparative Analysis" (#VSC-2fb85968)
   ⏱️ Time: ~5 seconds
   📊 Displays summary table
   💾 Saves: results/experiment_summary.csv
```

---

## Total Estimated Time

- **3 Embeddings (minimum)**: ~60-90 minutes
- **4 Embeddings (recommended)**: ~85-125 minutes
- **With Hyperparameter Tuning**: +4-8 hours

---

## Quick Commands (Terminal)

### Install Dependencies
```bash
pip install tensorflow gensim nltk beautifulsoup4 scikit-learn pandas matplotlib seaborn
```

### Download NLTK Data
Already included in notebook, but if needed:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Check GPU Availability
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

---

## After Training - View Results

### 1. Summary Table
```python
tracker.get_summary_df()
```

### 2. Best Model
```python
summary_df.sort_values('Accuracy', ascending=False).head(1)
```

### 3. All Experiment Details
```python
import json
with open('../results/experiments.json', 'r') as f:
    experiments = json.load(f)
print(json.dumps(experiments[-1], indent=2))  # Last experiment
```

---

## Hyperparameter Tuning (After Baseline)

1. Review baseline results
2. Identify top 2 embeddings
3. Uncomment tuning cell in PART 7
4. Run for each embedding:
   ```python
   best_config, best_model, results, tuning_df = hyperparameter_tuning(...)
   ```

---

## Files to Check

### Results
- `results/experiments.json` - Complete log
- `results/experiment_summary.csv` - Summary table
- `results/*_hyperparameter_tuning.csv` - Tuning results

### Models
- `models/*_best.h5` - Saved model weights

### Plots
- `results/plots/*_history.png` - Training curves
- `results/plots/*_confusion_matrix.png` - Confusion matrices
- `results/plots/model_comparison.png` - Model comparison

---

## Troubleshooting

### Out of Memory
```python
TRAIN_CONFIG['SAMPLE_SIZE'] = 20000  # Reduce sample
TRAIN_CONFIG['BATCH_SIZE'] = 16      # Reduce batch size
```

### Too Slow
```python
TRAIN_CONFIG['EPOCHS'] = 20          # Reduce epochs
TRAIN_CONFIG['SAMPLE_SIZE'] = 10000  # Smaller sample
```

### Check Progress During Training
Look for these outputs:
- Epoch progress bar
- Training/validation loss and accuracy
- Early stopping messages
- Best model checkpoint saves

---

## Expected Outputs Per Experiment

Each experiment produces:
1. ✅ Embedding generation confirmation
2. 📊 Model architecture summary
3. 🚀 Training progress (epoch-by-epoch)
4. 📈 Final evaluation metrics
5. 🖼️ Training history plot
6. 🎯 Confusion matrix plot
7. 💾 Experiment logged confirmation

---

## Keyboard Shortcuts (Jupyter)

- `Shift + Enter` - Run cell and move to next
- `Ctrl + Enter` - Run cell and stay
- `Alt + Enter` - Run cell and insert below
- `Esc + A` - Insert cell above
- `Esc + B` - Insert cell below
- `Esc + D D` - Delete cell
- `Esc + M` - Convert to Markdown
- `Esc + Y` - Convert to Code

---

**Happy Training! 🎉**
