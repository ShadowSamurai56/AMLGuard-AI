# Project Report: AI-Based Anti-Money Laundering Transaction Monitoring System

**Student Name:** Raghav Krishan Aggarwal  
**Project Title:** AMLGuard AI  
**Course / Degree:** B.Tech / M.Sc Project  
**Academic Year:** 2025–2026  
**GitHub Repository:** [https://github.com/ShadowSamurai56/AMLGuard-AI](https://github.com/ShadowSamurai56/AMLGuard-AI)  

---

## 1. Introduction

### 1.1 Application Context
Anti-Money Laundering (AML) monitoring is something every regulated bank and fintech company has to do by law. When criminals generate illegal funds through drug trafficking, corruption, or cybercrime, they try to wash the dirty money through the banking system so it looks legitimate. Regulators like FinCEN in the US and the FCA in the UK force financial institutions to monitor every transaction and file Suspicious Activity Reports (SARs) whenever an account acts suspiciously.

In most real-world banks today, transaction monitoring is still done using old rule-based systems. These systems look at basic static thresholds. For example, they flag any transaction greater than $10,000, or alert when someone makes three withdrawals in a single day. Criminals already know these rules and easily get around them by splitting money into smaller chunks—a technique called structuring or "smurfing"—and passing funds through dozens of intermediate accounts. On top of that, these rigid rule engines produce huge false positive rates, often between 90% and 95%. Compliance analysts end up wasting most of their working hours clearing false alarms on innocent customers, while actual organized laundering networks slip right through.

### 1.2 Purpose of the Project
The main purpose of this project, **AMLGuard AI**, is to build and test a machine learning and graph-based monitoring system that looks at the whole transaction network instead of just isolated transfers. 

A lot of recent research papers claim that Graph Neural Networks (GNNs) are the ultimate solution for AML. But when you look closely at many published studies, they often evaluate models on random train/test splits that leak data, or they test on tiny, balanced subsets that don't look anything like real banking conditions. I wanted to build a realistic testbed to answer a straight question: **Do Graph Neural Networks (GraphSAGE and GAT) actually beat strong traditional machine learning models (Random Forest and XGBoost) when tested under realistic AML conditions?** 

To test this fairly, I set up a multi-seed evaluation across both a simulated multi-agent banking dataset (IBM AMLSim) and a real-world Bitcoin dataset (Elliptic), testing both random splits and time-respecting temporal splits.

### 1.3 Scope of the Work
The scope of this project covers:
1. Setting up a data pipeline that takes raw transaction logs from AMLSim, stores them in MySQL, and extracts 17 behavioral features per account without leaking future information or target labels.
2. Converting the transaction records into directed attributed graphs in PyTorch Geometric (PyG), with accounts as nodes and money transfers as directed edges.
3. Implementing five distinct models: Logistic Regression (linear baseline), Random Forest, XGBoost, GraphSAGE (inductive GNN), and Graph Attention Networks (GAT).
4. Running a 5-seed evaluation ($0, 1, 42, 123, 2024$) on an identical, leakage-free 60/20/20 train/validation/test split so trees and GNNs are tested on the exact same accounts.
5. Testing how models degrade when evaluated across time using the Elliptic dataset (training on time steps 1–34 and testing on steps 35–49).
6. Evaluating how well each model catches specific money laundering patterns like circular routing (cycles), gathering (fan-in), and scattering (fan-out).
7. Building an explainability layer using TreeSHAP for tree models and GNNExplainer for graph models, so compliance teams get clear evidence dossiers rather than black-box probabilities.
8. Building an interactive Streamlit workbench so an examiner or analyst can inspect accounts, view model consensus, and see the transaction graph live.

### 1.4 Concrete Objectives
- **Objective 1:** Ingest transaction logs into a local MySQL database and engineer 17 behavioral features per account (velocity, turnover ratio, inbound/outbound metrics, counterparty counts) while strictly preventing target label leakage.
- **Objective 2:** Construct PyG graph objects and establish identical train/val/test masks across tabular models and GNNs.
- **Objective 3:** Implement class-weighted loss functions to handle extreme class imbalance where less than 5% of accounts are illicit.
- **Objective 4:** Tune GAT hyperparameters via grid search across attention heads, hidden dimensions, and learning rates.
- **Objective 5:** Measure the performance drop when switching from random splits to temporal splits on the Elliptic Bitcoin dataset.
- **Objective 6:** Measure seed-by-seed recall across specific injected typologies (cycles, fan-in, and fan-out).
- **Objective 7:** Provide local SHAP feature attributions and GNNExplainer subgraph visualizations for case reviews.

---

## 2. Related Work

I reviewed 15 research papers relevant to graph neural networks, machine learning baselines, and explainability in financial crime and AML detection.

### 2.1 Critical Review of Selected Literature

**Paper 1: Finding Money Launderers Using Heterogeneous Graph Neural Networks (Johannessen & Jullum, 2025)**  
Most AML detection systems only look at who is sending money to whom. The authors point out that real-world financial crimes involve a lot more context, like shared company ownerships, overlapping addresses, or dual roles in different businesses. Traditional methods usually ignore this richer relational context. To capture these complex relationships, they built a heterogeneous graph with multiple node and edge types and processed it through a Message Passing Neural Network (MPNN) using real, proprietary data from DNB, Norway's biggest bank. Their multi-relational approach outperformed baseline models that only looked at basic transactions. However, because the data is proprietary, their results cannot be reproduced by other researchers. Also, a massive heterogeneous model is very complicated to explain to bank regulators. This paper validated that graph structures work for AML, but it helped clarify my own project's scope: how much performance do we really sacrifice by sticking to a simpler, more explainable single-relation GraphSAGE setup?

**Paper 2: Inductive Graph Representation Learning for Fraud Detection (Van Belle et al., 2022)**  
The authors point out a huge practical flaw in a lot of academic graph research: many models are "transductive" (like standard GCN), meaning they break if you try to feed them new, unseen nodes. Since real banks open new accounts and process new transactions every second, a transductive model isn't actually deployable in a live production environment. They tested inductive graph learning methods, specifically GraphSAGE, which learns functions to aggregate information from a node's local neighborhood rather than memorizing a static table of node embeddings. A major finding was that the graph structure holds unique predictive power, and inductive models let you score new accounts without retraining the entire network. This paper gave me my literature-backed justification for choosing GraphSAGE: inductive capability is a hard requirement for live deployment.

**Paper 3: Structural Entropy Minimization Combining Graph Representation for Money Laundering Identification (SEGE, 2024)**  
This study argues that money laundering is basically a team sport, not just an individual-account problem. Launderers operate in coordinated networks, but most detection algorithms just score accounts one by one, completely missing the broader collaborative structure. The researchers proposed an unsupervised method called SEGE (Structural Entropy Minimization Combined with Graph Embedding) that searches for tightly-knit subgroups within a network by minimizing structural entropy. They tested this on both private real-world data and a public synthetic dataset. The big result was that SEGE can identify laundering communities without needing labeled training data. But because it's unsupervised, it can't directly leverage confirmed SAR labels when you have them. Also, focusing purely on communities might miss a lone-wolf account acting suspiciously. This paper showed me that treating accounts in isolation is a blind spot, which led me to include typology-specific recall tests for circular laundering rings in my evaluation.

**Paper 4: Detecting Anomalous Cryptocurrency Transactions: An AML/CFT Application of Machine Learning-Based Forensics (Electronic Markets, 2023)**  
Applying AML techniques to cryptocurrency is a different challenge compared to traditional banking. Crypto transactions are pseudonymous and sit on a public ledger, meaning you can see the whole network structure, but you rarely have reliable ground-truth labels for which addresses belong to criminals. The authors modeled Bitcoin transaction flows as a directed graph and tested Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT). They proved that graph-based approaches are effective at finding anomalies in crypto. The biggest limitation is the domain gap: an anonymous crypto wallet is not the same thing as a bank account tied to a real person who passed KYC checks. You can't assume a model tuned for Bitcoin will work identically for retail banking. Including this paper gave me a solid basis to evaluate both fiat banking data (AMLSim) and cryptocurrency data (Elliptic) in my project.

**Paper 5: Intelligent Anti-Money Laundering Fraud Control Using Graph-Based Machine Learning Model for the Financial Domain (JCIT, 2023)**  
A lot of graph-based AML research looks great on small academic datasets, but falls apart when applied to massive banking networks. This paper tested a graph-based ML model on a large synthetic dataset with 100,000 nodes and 5.3 million edges to check scalability. They reported an overall accuracy of 77–79%. However, overall accuracy is a very misleading metric for AML because fraud datasets are heavily imbalanced. If only 2% of accounts are fraudulent, a dumb model that guesses "legitimate" every time gets 98% accuracy. The paper lacked a focus on minority-class metrics like precision and recall, and the authors admitted their input features were basic. This paper validated my decision to focus strictly on AUC-PR, Precision, Recall, and F1-score, and motivated the time I spent engineering 17 behavioral features.

**Paper 6: Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics (Weber et al., 2019)**  
Weber et al. introduced the Elliptic Bitcoin dataset (203,769 transactions and 234,355 directed payment edges across 49 time steps). They benchmarked Random Forest, standard GCN, and EvolveGCN. Their most important finding was that when they switched from a random train/test split to a realistic time-based split (training on past time steps and testing on future ones), model performance dropped sharply. This paper is the primary baseline for the Elliptic experiments in AMLGuard AI. It highlighted the risk of temporal data leakage in academic evaluations.

**Paper 7: Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)**  
Hamilton et al. introduced GraphSAGE, solving the limitation of older GNNs that could not generalize to unseen nodes. By sampling and aggregating feature information from local 1-hop and 2-hop neighborhoods using mean, pooling, or LSTM aggregators, GraphSAGE learns functions that can be applied to newly arrived accounts. In AMLGuard AI, I used GraphSAGE with a mean aggregator, dropout regularization, and class-weighted cross-entropy.

**Paper 8: Graph Attention Networks (Veličković et al., 2018)**  
Veličković et al. proposed Graph Attention Networks (GAT), introducing multi-head self-attention to graph convolution. Instead of treating all neighbors equally (like GCN) or sampling them uniformly (like GraphSAGE), GAT assigns dynamic attention weights $\alpha_{ij}$ to edges based on node feature differences. In theory, this allows the model to pay more attention to high-risk counterparties while ignoring routine transfers.

**Paper 9: EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs (Pareja et al., 2020)**  
Pareja et al. addressed the problem of graph structure changing over time by using an RNN to update the GCN weight matrices between consecutive time steps. Evaluating on the Elliptic dataset, they showed that dynamic models hold up better than static GCNs as criminal behavior shifts. This paper helped explain why static GNNs in my own experiments lost over 50% of their AUC-PR when evaluated on future time steps.

**Paper 10: AMLSim: A Synthetic Multi-Agent Financial Transaction Simulator (Altman et al. / IBM Research, 2019)**  
Because bank secrecy laws stop financial institutions from sharing real customer transaction data, IBM Research created AMLSim. AMLSim simulates banking environments with normal background commerce and injects known laundering topologies (gathering/fan-in, scattering/fan-out, and cycles). AMLGuard AI uses AMLSim to generate realistic synthetic banking logs with verifiable ground-truth SAR labels.

**Paper 11: A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)**  
Lundberg and Lee developed SHAP (SHapley Additive exPlanations), connecting game-theoretic Shapley values with local explanations. In financial compliance, black-box predictions cannot be used directly because analysts must write clear narratives when submitting SARs to regulators. I used TreeSHAP in AMLGuard AI to extract exact feature contributions for every alert generated by Random Forest and XGBoost.

**Paper 12: GNNExplainer: Generating Explanations for Graph Neural Networks (Ying et al., 2019)**  
While SHAP works well for tabular feature vectors, it cannot explain graph edge structures. Ying et al. developed GNNExplainer, an optimization method that identifies a compact subgraph and a small subset of node features that maximize mutual information with the GNN's prediction. In AMLGuard AI, I used GNNExplainer to identify which transaction edges contributed most to GraphSAGE predictions.

**Paper 13: Machine Learning Methods to Detect Money Laundering in Fully Attributed Transaction Networks (Lorenz et al., 2020)**  
Lorenz et al. evaluated tree-based algorithms and active learning strategies on cryptocurrency networks. They emphasized that tree ensembles (Random Forest and XGBoost) frequently match or beat complex deep learning models when rich local tabular features are properly extracted. Their findings match what I observed in my own experiments on AMLSim and Elliptic.

**Paper 14: Competence of Graph Convolutional Networks for Anti-Money Laundering in Bitcoin Blockchain (Alarab et al., 2020)**  
Alarab et al. benchmarked GCN variants against Random Forest on the Elliptic dataset. They confirmed that combining graph structural representations with supervised classifiers helps catch illicit transactions, but warned that stacking more than two or three GNN layers causes over-smoothing. This guided my decision to keep GraphSAGE and GAT architectures shallow (1 to 2 layers).

**Paper 15: Application of Machine Learning in Anti-Money Laundering: A Systematic Literature Review (Vassallo et al., 2021)**  
This systematic survey reviewed over 80 studies on machine learning in AML, identifying common problems in the field: extreme class imbalance, lack of standardized evaluation splits, over-reliance on inaccurate accuracy metrics, and an absence of explainability frameworks. This review served as a guide for my project to ensure our experimental design avoided these pitfalls.

---

### 2.2 Summary Table of Related Work

| # | Paper & Authors | Year | Core Contribution | Research Gap Left / Unaddressed |
|---|---|---|---|---|
| 1 | *Finding Money Launderers Using Heterogeneous Graph Neural Networks* (Johannessen & Jullum) | 2025 | Built multi-relational MPNN incorporating company ownership and shared addresses on real DNB banking data. | Evaluated only on proprietary private data; high model complexity makes regulatory audit and explanation difficult. |
| 2 | *Inductive Graph Representation Learning for Fraud Detection* (Van Belle et al.) | 2022 | Demonstrated inductive GraphSAGE models for fraud detection that generalize to newly created, unseen nodes without retraining. | Did not test on multi-seed evaluations and focused on card fraud rather than structured multi-hop AML typologies. |
| 3 | *Structural Entropy Minimization Combining Graph Representation (SEGE)* (Int. J. Mach. Learn. Cyber.) | 2024 | Proposed unsupervised structural entropy minimization to catch coordinated laundering syndicates without labeled training data. | Unsupervised clustering cannot exploit confirmed SAR supervision when available; misses lone-wolf illicit accounts. |
| 4 | *Detecting Anomalous Cryptocurrency Transactions* (Electronic Markets) | 2023 | Modeled Bitcoin transactions as directed graphs using GCN and GAT for anomaly forensics. | Domain gap: anonymous Bitcoin addresses lack KYC identity semantics, so findings cannot be directly assumed for retail banking. |
| 5 | *Intelligent AML Fraud Control Using Graph-Based ML* (JCIT) | 2023 | Tested scalability of graph ML models on a large synthetic network of 100K nodes and 5.3M edges. | Relied on misleading overall accuracy (77–79%) in an imbalanced setting; used basic feature engineering without deep metric analysis. |
| 6 | *Anti-Money Laundering in Bitcoin: GCNs for Financial Forensics* (Weber et al.) | 2019 | Released the benchmark Elliptic Bitcoin dataset and showed that temporal splits cause noticeable performance drops compared to random splits. | Did not provide post-hoc explainability tools for regulatory compliance; focused primarily on node-level GCNs. |
| 7 | *Inductive Representation Learning on Large Graphs (GraphSAGE)* (Hamilton et al.) | 2017 | Introduced inductive neighborhood aggregation functions (mean/pool/LSTM) for scalable node classification. | General graph theory paper; did not address extreme financial class imbalance or money laundering typologies. |
| 8 | *Graph Attention Networks (GAT)* (Veličković et al.) | 2018 | Introduced masked self-attention to dynamically weight neighbor importance during message passing. | Sensitive to hyperparameter choices; attention weights can collapse to uniform distributions on noisy, sparse transaction graphs. |
| 9 | *EvolveGCN: Evolving Graph Convolutional Networks* (Pareja et al.) | 2020 | Combined RNNs with GCNs to evolve model parameters across discrete time steps on dynamic graphs like Elliptic. | High computational training overhead; requires dense snapshot sequences that are often unavailable in real-time banking streams. |
| 10 | *AMLSim: Multi-Agent Transaction Simulator* (Altman et al. / IBM Research) | 2019 | Created a multi-agent transaction simulation engine generating realistic banking logs with injected laundering topologies. | Generates synthetic transactions; simulated behavioral rules may not capture novel, emergent real-world criminal strategies. |
| 11 | *A Unified Approach to Interpreting Model Predictions (SHAP)* (Lundberg & Lee) | 2017 | Formulated Shapley additive explanations for exact, game-theoretic local feature attributions in tree models. | Designed for tabular and image features; cannot explain edge connections or relational graph topologies directly. |
| 12 | *GNNExplainer: Explanations for Graph Neural Networks* (Ying et al.) | 2019 | Formulated mutual-information optimization to isolate explanatory subgraphs and node features for GNN predictions. | Explanations can be noisy on dense transaction graphs; requires careful thresholding to extract human-readable patterns. |
| 13 | *ML Methods to Detect Money Laundering in Fully Attributed Networks* (Lorenz et al.) | 2020 | Benchmarked tree ensembles with active learning on cryptocurrency networks, showing trees often beat deep learning on tabular features. | Did not evaluate deep graph attention networks or investigate specific multi-hop topological structures (cycles). |
| 14 | *Competence of GCNs for Anti-Money Laundering* (Alarab et al.) | 2020 | Evaluated GCN variations against Random Forest on Elliptic, finding that deeper GNN layers lead to over-smoothing. | Limited to 1-2 seeds; did not evaluate hyperparameter tuning for attention-based models or time-respecting drops. |
| 15 | *Application of ML in AML: Systematic Literature Review* (Vassallo et al.) | 2021 | Documented widespread methodological flaws in AML research: class imbalance mismanagement, lack of code reproducibility, and missing explainability. | Survey paper; did not conduct empirical comparative experiments across graph and tabular baselines. |

---

## 3. Methodology

### 3.1 Overall Pipeline
The end-to-end architecture of AMLGuard AI is organized into data ingestion, feature engineering, graph building, multi-seed training, explainability, and an interactive UI:

```
+-----------------------------------------------------------------------------------+
|                           AMLGuard AI System Pipeline                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Raw Data Sources]                                                               |
|  - IBM AMLSim (1,450 accounts, 10,000 transactions, 80 SAR positives)             |
|  - Elliptic Bitcoin (203,769 transactions, 234,355 payment flows, 49 steps)       |
|                                     |                                             |
|                                     v                                             |
|  [Storage & Relational Schema]                                                    |
|  - MySQL Database: raw_transactions, account_features, alert_accounts             |
|  - Offline Parquet Fallback Cache (for fast offline demo loading)                 |
|                                     |                                             |
|                                     v                                             |
|  [Feature Extraction Engine (Leakage-Free)]                                       |
|  - 17 behavioral metrics: count, volume, turnover ratio, active duration, peers  |
|  - Strictly excludes SAR labels and target IDs from feature matrices              |
|                                     |                                             |
|                                     v                                             |
|  [Graph Construction (PyG) & Unified 60/20/20 Masks]                              |
|  - Nodes = Accounts, Edges = Directed Money Flows                                 |
|  - Identical split masks: Train (870), Val (290), Test (290 / 16 positives)        |
|                                     |                                             |
|            +------------------------+------------------------+                    |
|            |                                                 |                    |
|            v                                                 v                    |
|  [Tabular Classifiers]                             [Graph Neural Networks]        |
|  - Logistic Regression (Scaled, Balanced)          - GraphSAGE (1-layer, 64-dim)  |
|  - Random Forest (100 trees, Balanced)             - GAT (Grid-search tuned)      |
|  - XGBoost (scale_pos_weight=17.1)                 - Class-weighted Cross-Entropy |
|            |                                                 |                    |
|            +------------------------+------------------------+                    |
|                                     |                                             |
|                                     v                                             |
|  [5-Seed Statistical Evaluation] (Seeds: 0, 1, 42, 123, 2024)                     |
|  - Metrics: AUC-PR, F1-Score, Precision, Recall                                   |
|  - Elliptic Temporal Drift Test (Steps 1-34 Train vs Steps 35-49 Test)            |
|  - Typology Breakdown (Recall on Cycles, Fan-In, Fan-Out)                         |
|                                     |                                             |
|                                     v                                             |
|  [Explainability & Compliance Layer]                                              |
|  - TreeSHAP: Local Shapley log-odds bar charts for top risk drivers               |
|  - GNNExplainer: Subgraph edge importance overlays                                |
|                                     |                                             |
|                                     v                                             |
|  [Interactive Streamlit Workbench] (http://localhost:8501)                       |
|  - Multi-model consensus cards, 1-hop radial ego graph, transaction audit ledger  |
+-----------------------------------------------------------------------------------+
```

### 3.2 Data Ingestion and Feature Extraction
The raw simulated logs were loaded into a local MySQL database (`amlguard`) with three tables:
- `raw_transactions`: stores individual transfers (`step`, `from_acc`, `to_acc`, `amount`, `is_sar`).
- `account_features`: stores engineered features per account.
- `alert_accounts`: stores the injected laundering typologies (`cycle`, `fan_in`, `fan_out`).

I engineered 17 behavioral features per account:
1. `out_tx_count`: count of outgoing transactions.
2. `in_tx_count`: count of incoming transactions.
3. `total_tx_count`: total transactions.
4. `out_total_amt`: total outgoing sum.
5. `in_total_amt`: total incoming sum.
6. `out_avg_amt`: average outgoing amount.
7. `in_avg_amt`: average incoming amount.
8. `out_std_amt`: standard deviation of outgoing amounts.
9. `in_std_amt`: standard deviation of incoming amounts.
10. `out_max_amt`: maximum outgoing transaction amount.
11. `in_max_amt`: maximum incoming transaction amount.
12. `out_unique_counterparties`: count of unique receiving accounts.
13. `in_unique_counterparties`: count of unique sending accounts.
14. `total_unique_counterparties`: total distinct accounts transacted with.
15. `turnover_ratio`: net passthrough flow ratio $\frac{|\text{in} - \text{out}|}{\text{in} + \text{out} + 1\text{e-}5}$.
16. `avg_hours_between_out_tx`: average time interval between outgoing transactions.
17. `net_flow`: net balance change ($\text{total\_in} - \text{total\_out}$).

> **Fixing Data Leakage:** In an earlier script, degree calculations counted transactions that included `is_sar` flags. I fixed this by ensuring feature extraction only uses raw transaction amounts and timestamps, completely excluding all target columns (`label_*`).

### 3.3 Graph Construction and Split Alignment
In PyTorch Geometric, accounts were mapped to nodes with feature vectors $x \in \mathbb{R}^{17}$, and transactions were mapped to directed edges.

**Fixing the Train/Test Split Mismatch:**  
When checking the baseline code, I caught a major split mismatch: `seeded_baseline_eval.py` originally used an 80/20 `train_test_split()`, while the GNNs used a fixed 60/20/20 mask stored in `models/aml_graph.pt`. This mismatch caused **9 out of the GNN's 16 test positives to leak into the tree models' training set**. 

I edited `load_amlsim()` so that both tree models and GNNs load their train, validation, and test accounts directly from `data.train_mask`, `data.val_mask`, and `data.test_mask` in `models/aml_graph.pt`. This guaranteed an honest, identical split:
- **Total accounts:** 1,450 (80 positives, 1,370 negatives, 5.52% positive rate).
- **Train set:** 870 accounts (48 positives, 822 negatives).
- **Validation set:** 290 accounts (16 positives, 274 negatives).
- **Test set:** 290 accounts (16 positives, 274 negatives).

### 3.4 Model Training Protocols
All models were evaluated across 5 seeds ($0, 1, 42, 123, 2024$):
- **Logistic Regression:** Scaled with `StandardScaler`, `class_weight='balanced'`, L2 regularization.
- **Random Forest:** 100 trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`.
- **XGBoost:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=17.1`.
- **GraphSAGE:** 1-layer GraphSAGE (`hidden_dim=64`, `dropout=0.3`), Adam optimizer (`lr=0.01`, `weight_decay=5e-4`), class-weighted cross-entropy, evaluated every 20 epochs with checkpointing on the best validation F1 score.
- **GAT:** 2-layer Graph Attention Network with grid search tuning over heads (2, 4, 8), hidden dimensions (16, 32, 64), and learning rates (0.001, 0.005, 0.01).

---

## 4. Results and Discussion

### 4.1 Master Benchmark Results
All metrics below report the sample mean $\pm$ sample standard deviation ($\text{ddof}=1$) over the 5 random seeds ($0, 1, 42, 123, 2024$). The primary metric is **AUC-PR** (Area Under the Precision-Recall Curve), which is standard for imbalanced fraud datasets.

| Dataset | Model | Precision | Recall | F1-Score | AUC-PR | Test Positives / Total N |
|---|---|---|---|---|---|---|
| **AMLSim** | Logistic Regression | $0.1667 \pm 0.0000$ | $0.7500 \pm 0.0000$ | $0.2727 \pm 0.0000$ | $0.3623 \pm 0.0000$ | 16 / 290 |
| **AMLSim** | **Random Forest** | $\mathbf{0.9778 \pm 0.0497}$ | $0.5375 \pm 0.0342$ | $\mathbf{0.6933 \pm 0.0377}$ | $\mathbf{0.8214 \pm 0.0126}$ | 16 / 290 |
| **AMLSim** | XGBoost | $0.9250 \pm 0.1118$ | $0.4500 \pm 0.0685$ | $0.6041 \pm 0.0786$ | $0.7186 \pm 0.0347$ | 16 / 290 |
| **AMLSim** | GraphSAGE | $0.2677 \pm 0.0245$ | $0.7125 \pm 0.0713$ | $0.3891 \pm 0.0358$ | $0.5442 \pm 0.0515$ | 16 / 290 |
| **AMLSim** | GAT (tuned) | $0.1742 \pm 0.0588$ | $0.6000 \pm 0.0559$ | $0.2661 \pm 0.0738$ | $0.2774 \pm 0.1085$ | 16 / 290 |
| **Elliptic (Random)** | Logistic Regression | $0.4430 \pm 0.0000$ | $0.9274 \pm 0.0000$ | $0.5996 \pm 0.0000$ | $0.7720 \pm 0.0000$ | 909 / 9,313 |
| **Elliptic (Random)** | Random Forest | $\mathbf{0.9915 \pm 0.0009}$ | $0.8946 \pm 0.0009$ | $0.9406 \pm 0.0009$ | $0.9805 \pm 0.0003$ | 909 / 9,313 |
| **Elliptic (Random)** | **XGBoost** | $0.9829 \pm 0.0043$ | $\mathbf{0.9250 \pm 0.0036}$ | $\mathbf{0.9531 \pm 0.0032}$ | $\mathbf{0.9830 \pm 0.0007}$ | 909 / 9,313 |
| **Elliptic (Random)** | GraphSAGE | $0.7714 \pm 0.0091$ | $0.9039 \pm 0.0048$ | $0.8324 \pm 0.0061$ | $0.9361 \pm 0.0017$ | 909 / 9,313 |
| **Elliptic (Random)** | GAT (tuned) | $0.3893 \pm 0.0163$ | $0.9525 \pm 0.0031$ | $0.5525 \pm 0.0163$ | $0.8679 \pm 0.0054$ | 909 / 9,313 |
| **Elliptic (Time)** | Logistic Regression | $0.1342 \pm 0.0000$ | $\mathbf{0.9566 \pm 0.0000}$ | $0.2354 \pm 0.0000$ | $0.1984 \pm 0.0000$ | 1,083 / 16,670 |
| **Elliptic (Time)** | **Random Forest** | $\mathbf{0.9673 \pm 0.0037}$ | $0.6419 \pm 0.0221$ | $\mathbf{0.7715 \pm 0.0151}$ | $\mathbf{0.7831 \pm 0.0008}$ | 1,083 / 16,670 |
| **Elliptic (Time)** | XGBoost | $0.8338 \pm 0.0123$ | $0.7097 \pm 0.0051$ | $0.7667 \pm 0.0041$ | $0.7805 \pm 0.0027$ | 1,083 / 16,670 |
| **Elliptic (Time)** | GraphSAGE | $0.3340 \pm 0.0494$ | $0.6842 \pm 0.0242$ | $0.4462 \pm 0.0404$ | $0.4573 \pm 0.0486$ | 1,083 / 16,670 |
| **Elliptic (Time)** | GAT | $0.1254 \pm 0.0027$ | $0.9435 \pm 0.0052$ | $0.2213 \pm 0.0042$ | $0.3287 \pm 0.0108$ | 1,083 / 16,670 |

---

### 4.2 Temporal Drift Analysis: Random Split vs Time Split
Evaluating models on random splits can be misleading because transactions from future time steps get mixed into the training set. When we tested on a strict time-respecting split on Elliptic (training on time steps 1–34 and testing on steps 35–49), every model dropped, but GNNs dropped much harder:

| Model | Random Split AUC-PR | Time Split AUC-PR | Absolute Drop | Relative Drop (%) |
|---|---|---|---|---|
| **Logistic Regression** | $0.7720 \pm 0.0000$ | $0.1984 \pm 0.0000$ | $-0.5736$ | **$-74.30\%$** |
| **Random Forest** | $0.9805 \pm 0.0003$ | $0.7831 \pm 0.0008$ | $-0.1973$ | **$-20.13\%$** |
| **XGBoost** | $0.9830 \pm 0.0007$ | $0.7805 \pm 0.0027$ | $-0.2025$ | **$-20.60\%$** |
| **GraphSAGE** | $0.9361 \pm 0.0017$ | $0.4573 \pm 0.0486$ | $-0.4788$ | **$-51.15\%$** |
| **GAT** | $0.8616 \pm 0.0058$ | $0.3287 \pm 0.0108$ | $-0.5329$ | **$-61.85\%$** |

**Why did this happen?**
Tree models rely on local transaction features (like outgoing volume vs incoming volume). These basic financial habits stay relatively consistent over time. Even when money launderers switch wallets, their basic cash-out volumes still stand out. 

Graph neural networks, on the other hand, rely on message passing across specific network connections. In Bitcoin, when illicit entities move to fresh clusters or change the number of hops they use, the old graph patterns become obsolete. The static GNN overfits to the early network topology and struggles when future time steps introduce different routing structures.

---

### 4.3 Typology Breakdown: The "Cycle Paradox"
Even though Random Forest had a higher global AUC-PR ($0.8214$ vs $0.5442$), evaluating models by specific money laundering typologies revealed where GNNs are uniquely valuable. 

I matched the 16 test positives against their injected laundering alert types:
- **Cycle ($n=4$):** Circular routing where money passes through 3 to 5 shell accounts and returns to the sender.
- **Fan-in ($n=6$):** Smurfing where multiple small deposits consolidate into one account.
- **Fan-out ($n=6$):** Scattering where a lump sum is dispersed into many recipient accounts.

The table below shows how many accounts each model caught across the 5 seeds:

| Model | Alert Type | $n$ | Flagged Count $k$ per Seed $[0, 1, 42, 123, 2024]$ | Mean Recall |
|---|---|---|---|---|
| **Random Forest** | cycle | 4 | $[1, 1, 1, 1, 1]$ | **25.0%** |
| Random Forest | fan_in | 6 | $[4, 4, 4, 4, 4]$ | **66.7%** |
| Random Forest | fan_out | 6 | $[3, 4, 3, 4, 4]$ | **60.0%** |
| **XGBoost** | cycle | 4 | $[2, 2, 2, 1, 1]$ | **40.0%** |
| XGBoost | fan_in | 6 | $[3, 3, 2, 4, 4]$ | **53.3%** |
| XGBoost | fan_out | 6 | $[2, 2, 2, 2, 4]$ | **40.0%** |
| **GraphSAGE** | **cycle** | 4 | $[3, 3, 3, 4, 3]$ | $\mathbf{80.0\%}$ |
| **GraphSAGE** | fan_in | 6 | $[4, 4, 4, 4, 5]$ | $\mathbf{70.0\%}$ |
| **GraphSAGE** | fan_out | 6 | $[3, 4, 4, 5, 4]$ | $\mathbf{66.7\%}$ |

**Why Random Forest missed cycles (25% recall):**
In a circular laundering ring, individual intermediary accounts pass money through in ordinary amounts. Their transaction counts and turnover ratios look completely normal when viewed one account at a time. Because Random Forest cannot see multi-hop connections, it only caught 1 out of the 4 cyclic accounts across all 5 seeds.

**Why GraphSAGE caught cycles (80% recall):**
GraphSAGE aggregates features from 1-hop and 2-hop neighbors. Through message passing, it detected that the money leaving an account was connected to accounts receiving funds from the exact same group, allowing it to flag 3 or 4 out of the 4 cyclic accounts in every single run.

**Practical Takeaway:** Relying only on tree models leaves a bank blind to circular money laundering. The best real-world architecture is a **two-tier hybrid system**: use Random Forest or XGBoost as a fast first-stage filter to catch volume anomalies, and run GraphSAGE as a second-stage graph scanner to detect complex multi-hop cycles.

---

### 4.4 Comparison with Published Work
- **Comparison with Weber et al. (2019):** Weber et al. reported an F1 of $0.72$ for Random Forest and $0.50$ for GCN on the Elliptic time split. In my experiments, Random Forest reached an F1 of $0.7715 \pm 0.0151$ (AUC-PR $0.7831$), while GraphSAGE reached an F1 of $0.4462 \pm 0.0404$ (AUC-PR $0.4573$). My results confirm their finding: static graph neural networks degrade significantly across time steps, while tree models stay much more stable.
- **Comparison with Lorenz et al. (2020):** Lorenz et al. argued that when solid domain features are extracted from transaction networks, tree ensembles consistently beat deep learning models while training much faster. My master results on both AMLSim and Elliptic match their conclusions.

---

## 5. References

1. **Johannessen, F., & Jullum, M. (2025).** Finding money launderers using heterogeneous graph neural networks. *Journal of Finance and Data Science*, 11, 100175. [https://doi.org/10.1016/j.jfds.2025.100175](https://doi.org/10.1016/j.jfds.2025.100175)
2. **Van Belle, R., Van Damme, C., Tytgat, H., & De Weerdt, J. (2022).** Inductive graph representation learning for fraud detection. *Expert Systems with Applications*, 193, 116463. [https://doi.org/10.1016/j.eswa.2021.116463](https://doi.org/10.1016/j.eswa.2021.116463)
3. **Zhang, X., Liu, C., & Wang, Y. (2024).** Structural entropy minimization combining graph representation for money laundering identification. *International Journal of Machine Learning and Cybernetics*, 15(4), 1421–1436. [https://doi.org/10.1007/s13042-024-02129-z](https://doi.org/10.1007/s13042-024-02129-z)
4. **Kamps, J., & Kleinberg, B. (2023).** Detecting anomalous cryptocurrency transactions: An AML/CFT application of machine learning-based forensics. *Electronic Markets*, 33(1), 32. [https://doi.org/10.1007/s12525-023-00654-3](https://doi.org/10.1007/s12525-023-00654-3)
5. **Sharma, A., & Kumar, P. (2023).** Intelligent anti-money laundering fraud control using graph-based machine learning model for the financial domain. *Journal of Cases on Information Technology (JCIT)*, 25(1), 1–18. [https://doi.org/10.4018/JCIT.316665](https://doi.org/10.4018/JCIT.316665)
6. **Weber, M., Chen, J., Toyoda, K., & Suzumura, T. (2019).** Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD Workshop on Applied Data Science for Healthcare and Financial Forensics*, arXiv:1908.02591. [https://arxiv.org/abs/1908.02591](https://arxiv.org/abs/1908.02591)
7. **Hamilton, W., Ying, Z., & Leskovec, J. (2017).** Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 1024–1034.
8. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** Graph attention networks. *International Conference on Learning Representations (ICLR 2018)*. [https://openreview.net/forum?id=rJXMpikCZ](https://openreview.net/forum?id=rJXMpikCZ)
9. **Pareja, A., Domeniconi, G., Chen, J., Ma, T., Toyoda, K., Lopez, A., & Suzumura, T. (2020).** EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), 5363–5370. [https://doi.org/10.1609/aaai.v34i04.6000](https://doi.org/10.1609/aaai.v34i04.6000)
10. **Altman, E., Suzumura, T., et al. / IBM Research. (2019).** AMLSim: A multi-agent synthetic financial transaction simulator for anti-money laundering research. *ACM International Conference on AI in Finance (ICAIF)* / GitHub: [https://github.com/IBM/AMLSim](https://github.com/IBM/AMLSim).
11. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774.
12. **Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).** GNNExplainer: Generating explanations for graph neural networks. *Advances in Neural Information Processing Systems (NeurIPS 2019)*, 32, 9240–9251.
13. **Lorenz, J., Silva, M. I., Aparício, D., Ascensao, J. T., & Bizarro, P. (2020).** Machine learning methods to detect money laundering in fully attributed transaction networks. *ECML-PKDD 2020 Workshop on Applied Data Science for Banking and Finance*, arXiv:2009.00350. [https://arxiv.org/abs/2009.00350](https://arxiv.org/abs/2009.00350)
14. **Alarab, I., Prakoonwit, S., & Nishioka, M. (2020).** Competence of graph convolutional networks for anti-money laundering in Bitcoin blockchain. *Proceedings of the 21st International Conference on Computer Systems and Technologies (CompSysTech '20)*, 169–175. [https://doi.org/10.1145/3400899.3400932](https://doi.org/10.1145/3400899.3400932)
15. **Vassallo, M., Adams, R., & Borg, J. (2021).** Application of machine learning in anti-money laundering: A systematic literature review. *SN Computer Science*, 2(6), 464. [https://doi.org/10.1007/s42979-021-00853-x](https://doi.org/10.1007/s42979-021-00853-x)

---

## 6. Code and Dataset Links

### 6.1 GitHub Repository
* **Project Repository:** [https://github.com/ShadowSamurai56/AMLGuard-AI](https://github.com/ShadowSamurai56/AMLGuard-AI)
* **Application Entrypoint:** `app.py` (Streamlit Transaction Monitoring Workbench)

### 6.2 Dataset Sources & Access Links
1. **IBM AMLSim Synthetic Banking Dataset:**
   * **Source Link:** [https://github.com/IBM/AMLSim](https://github.com/IBM/AMLSim)
   * **Description:** Multi-agent synthetic transaction simulation with 1,450 accounts, 10,000 transactions, and 80 confirmed SAR positive accounts. Injected typologies include cycles, fan-in, and fan-out patterns.
   * **Location in Project:** Stored in `data/` and loaded into MySQL database `amlguard`.
2. **Elliptic Bitcoin Transaction Dataset:**
   * **Source Link:** [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
   * **Description:** Public benchmark dataset mapping 203,769 Bitcoin transactions and 234,355 directed payment flows across 49 discrete time steps.
   * **Location in Project:** Extracted in `elliptic_data/` and processed into PyG graph objects in `models/`.

### 6.3 Instructions to Run the Project
To run the evaluation scripts and launch the Streamlit demo:
```bash
# 1. Activate environment and set runtime flags
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# 2. Run multi-seed baseline models (AMLSim & Elliptic)
python -u src/seeded_baseline_eval.py --only all

# 3. Run GAT hyperparameter tuning
python -u src/gat_tuning.py

# 4. Run typology-specific evaluation (cycles, fan-in, fan-out)
python -u src/typology_eval.py

# 5. Launch the interactive Streamlit workbench
streamlit run app.py
```
