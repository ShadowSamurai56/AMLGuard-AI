# Typology Breakdown on Graph Test Split Across 5 Seeds (v2)

> **Note:** n per alert type is very small because the test set has 16 positives (4 cycle, 6 fan_in, 6 fan_out). Seeds vary only model initialisation, not the test set.

| Model | Alert Type | n | k per seed [0,1,42,123,2024] | Mean Recall | Min Recall | Max Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | cycle | 4 | [1, 1, 1, 1, 1] | 0.2500 | 0.2500 | 0.2500 |
| Random Forest | fan_in | 6 | [4, 4, 4, 4, 4] | 0.6667 | 0.6667 | 0.6667 |
| Random Forest | fan_out | 6 | [3, 4, 3, 4, 4] | 0.6000 | 0.5000 | 0.6667 |
| XGBoost | cycle | 4 | [2, 2, 2, 1, 1] | 0.4000 | 0.2500 | 0.5000 |
| XGBoost | fan_in | 6 | [3, 3, 2, 4, 4] | 0.5333 | 0.3333 | 0.6667 |
| XGBoost | fan_out | 6 | [2, 2, 2, 2, 4] | 0.4000 | 0.3333 | 0.6667 |
| GraphSAGE | cycle | 4 | [3, 3, 3, 4, 3] | 0.8000 | 0.7500 | 1.0000 |
| GraphSAGE | fan_in | 6 | [4, 4, 4, 4, 5] | 0.7000 | 0.6667 | 0.8333 |
| GraphSAGE | fan_out | 6 | [3, 4, 4, 5, 4] | 0.6667 | 0.5000 | 0.8333 |
