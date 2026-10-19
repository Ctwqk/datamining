### Classification: is the airline recommended? (held-out test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| tfidf + LR (raw) | 0.853 | 0.864 | 0.906 |
| tfidf + SVM (raw) | 0.853 | 0.864 | 0.906 |
| tfidf + LR (clean) | 0.838 | 0.850 | 0.899 |
| two-channel LSTM + GloVe | 0.833 | 0.845 | 0.908 |
| tfidf + NB (raw) | 0.828 | 0.842 | 0.881 |
| two-channel LSTM (random init) | 0.810 | 0.822 | 0.881 |
| vader + LR | 0.806 | 0.821 | 0.867 |
| DistilBERT SST-2 (zero-shot) | 0.790 | 0.777 | 0.892 |
| majority | 0.538 | 0.699 | 0.500 |

### Regression: overall rating 1-10 (held-out test set)

| Model | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|
| tfidf + Ridge (raw) | 1.334 | 1.712 | 0.752 | 0.847 |
| vader + Ridge | 1.690 | 2.160 | 0.604 | 0.764 |
| mean rating | 3.147 | 3.435 | -0.000 | - |
