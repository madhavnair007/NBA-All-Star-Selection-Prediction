# NBA All-Star Selection Prediction

Predicting whether an NBA player will be selected as an All-Star in a given season using machine learning classification models trained on player performance statistics.

**Course:** DS 4400 - Machine Learning and Data Mining I (Spring 2026)

**Team:** Madhav Nair, Ryan Greene, Ray Gutierrez

---

> **IMPORTANT: Which notebook to use**
>
> **`NBA-AllStar-Final.ipynb`** is the **FINAL** notebook. It contains the complete pipeline: data cleaning, EDA, all 7 models, threshold tuning, and position normalization. All results in the report are produced by this notebook.
>
> `Data-Cleaning.ipynb` is an **earlier version** containing only data cleaning, EDA, and the 5 baseline models. It is kept in the repository for contribution history purposes and is **not** the final deliverable.

---

## Problem

All-Star selection is often debated and influenced by factors beyond raw performance. We apply machine learning to determine how much of All-Star selection can be explained by measurable statistics alone, framing it as a binary classification task.

## Dataset

- **Source:** [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) by Sumit Rodatta
- **Size:** 52,984 player-season records, 61 features after merging per-game stats, advanced metrics, and All-Star labels
- **Filtered dataset:** 29,093 records (1980-present, minimum 20 games played), 46 numeric features
- **Class imbalance:** ~4.5% All-Stars vs 95.5% non-All-Stars

## Models

We trained seven classification models spanning discriminative, generative, and ensemble approaches:

| Model | AUC-ROC | F1 (All-Star) | Precision | Recall |
|-------|---------|---------------|-----------|--------|
| Random Forest | 0.9886 | 0.7792 | 0.8618 | 0.7110 |
| Lasso Logistic (L1) | 0.9874 | 0.7588 | 0.8288 | 0.6996 |
| Logistic Regression | 0.9873 | 0.7578 | 0.8318 | 0.6958 |
| Gradient Boosting | 0.9871 | 0.7581 | 0.8069 | 0.7148 |
| LDA | 0.9829 | 0.7323 | 0.7164 | 0.7490 |
| QDA | 0.9803 | 0.4271 | 0.2739 | 0.9696 |
| Gaussian Naive Bayes | 0.9767 | 0.4885 | 0.3269 | 0.9658 |

Additional experiments include threshold tuning (optimizing F1 via precision-recall curves) and position normalization (z-score features within position groups).

## Repository Structure

```
.
├── NBA-AllStar-Final.ipynb      # *** FINAL notebook - complete pipeline ***
├── Data-Cleaning.ipynb          # Earlier version (kept for contribution history)
├── DS4400_Final_Report.docx     # Final project report
├── Player Per Game.csv          # Per-game statistics
├── Advanced.csv                 # Advanced metrics
├── All-Star Selections.csv      # All-Star selection labels
└── Full-Dataset/                # Additional raw data files
```

## Setup

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Run the final notebook:
```bash
jupyter notebook NBA-AllStar-Final.ipynb
```

## Key Findings

- All-Star selection is highly predictable from performance stats (AUC-ROC > 0.97 across all models)
- Random Forest achieved the best overall performance (AUC-ROC = 0.989, F1 = 0.779)
- Discriminative models outperformed generative models in precision
- Top predictive features: VORP, Win Shares, points per game, minutes per game, usage rate
- Threshold tuning improved F1 from 0.779 to 0.788 on the best model
- Position normalization had minimal impact, suggesting raw features already encode positional context
- Imperfect recall suggests non-statistical factors (fan voting, market size) also influence selection
