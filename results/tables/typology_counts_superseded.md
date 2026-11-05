# Typology Breakdown by Alert Type: GraphSAGE vs Random Forest

> **Note:** These counts are derived from a single evaluation run on AMLSim test positives; n per alert type is very small (n=4 for cycle, n=6 for fan_in, n=6 for fan_out; total n=16).

| Alert Type | GraphSAGE (k/n) | GraphSAGE Recall | Random Forest (k/n) | Random Forest Recall |
| --- | --- | --- | --- | --- |
| cycle | 2/4 | 0.5000 | 0.5 | 0.5000 |
| fan_in | 4/6 | 0.6667 | 0.6666666666666666 | 0.6667 |
| fan_out | 4/6 | 0.6667 | 0.8333333333333334 | 0.8333 |
