# Master Results: Mean ± Sample Std (ddof=1) Over Seeds

| Dataset | Model | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) | AUC-PR (mean ± std) | Seeds | Test N | Test Positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMLSim | Logistic Regression | 0.1667 ± 0.0000 | 0.7500 ± 0.0000 | 0.2727 ± 0.0000 | 0.3623 ± 0.0000 | 5 | 290 | 16 |
| AMLSim | Random Forest | 0.9778 ± 0.0497 | 0.5375 ± 0.0342 | 0.6933 ± 0.0377 | 0.8214 ± 0.0126 | 5 | 290 | 16 |
| AMLSim | XGBoost | 0.9250 ± 0.1118 | 0.4500 ± 0.0685 | 0.6041 ± 0.0786 | 0.7186 ± 0.0347 | 5 | 290 | 16 |
| AMLSim | GraphSAGE | 0.2677 ± 0.0245 | 0.7125 ± 0.0713 | 0.3891 ± 0.0358 | 0.5442 ± 0.0515 | 5 | 290 | 16 |
| AMLSim | GAT | 0.1576 ± 0.0269 | 0.5875 ± 0.0559 | 0.2463 ± 0.0298 | 0.2701 ± 0.0531 | 5 | 290 | 16 |
| AMLSim | GAT (tuned) | 0.1742 ± 0.0588 | 0.6000 ± 0.0559 | 0.2661 ± 0.0738 | 0.2774 ± 0.1085 | 5 | 290 | 16 |
| Elliptic (random split) | Logistic Regression | 0.4430 ± 0.0000 | 0.9274 ± 0.0000 | 0.5996 ± 0.0000 | 0.7720 ± 0.0000 | 5 | 9313 | 909 |
| Elliptic (random split) | Random Forest | 0.9915 ± 0.0009 | 0.8946 ± 0.0009 | 0.9406 ± 0.0009 | 0.9805 ± 0.0003 | 5 | 9313 | 909 |
| Elliptic (random split) | XGBoost | 0.9829 ± 0.0043 | 0.9250 ± 0.0036 | 0.9531 ± 0.0032 | 0.9830 ± 0.0007 | 5 | 9313 | 909 |
| Elliptic (random split) | GraphSAGE | 0.7714 ± 0.0091 | 0.9039 ± 0.0048 | 0.8324 ± 0.0061 | 0.9361 ± 0.0017 | 5 | 9313 | 909 |
| Elliptic (random split) | GAT | 0.3815 ± 0.0077 | 0.9512 ± 0.0036 | 0.5445 ± 0.0075 | 0.8616 ± 0.0058 | 5 | 9313 | 909 |
| Elliptic (random split) | GAT (tuned) | 0.3893 ± 0.0163 | 0.9525 ± 0.0031 | 0.5525 ± 0.0163 | 0.8679 ± 0.0054 | 5 | 9313 | 909 |
| Elliptic (time-respecting split) | Logistic Regression | 0.1342 ± 0.0000 | 0.9566 ± 0.0000 | 0.2354 ± 0.0000 | 0.1984 ± 0.0000 | 5 | 16670 | 1083 |
| Elliptic (time-respecting split) | Random Forest | 0.9673 ± 0.0037 | 0.6419 ± 0.0221 | 0.7715 ± 0.0151 | 0.7831 ± 0.0008 | 5 | 16670 | 1083 |
| Elliptic (time-respecting split) | XGBoost | 0.8338 ± 0.0123 | 0.7097 ± 0.0051 | 0.7667 ± 0.0041 | 0.7805 ± 0.0027 | 5 | 16670 | 1083 |
| Elliptic (time-respecting split) | GraphSAGE | 0.3340 ± 0.0494 | 0.6842 ± 0.0242 | 0.4462 ± 0.0404 | 0.4573 ± 0.0486 | 5 | 16670 | 1083 |
| Elliptic (time-respecting split) | GAT | 0.1254 ± 0.0027 | 0.9435 ± 0.0052 | 0.2213 ± 0.0042 | 0.3287 ± 0.0108 | 5 | 16670 | 1083 |
