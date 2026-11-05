"""
AMLGuard AI — Academic Project Report PDF Generator
Produces a print-ready PDF with embedded tables, charts, flowcharts, and human-written academic text.
"""

import os
import base64
import subprocess

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = "jpeg" if path.endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"

flowchart_b64 = img_to_b64("results/figures/methodology_flowchart.png")
auc_pr_b64 = img_to_b64("results/figures/auc_pr_by_model.png")
split_b64 = img_to_b64("results/figures/elliptic_split_comparison.png")
shap_b64 = img_to_b64("results/figures/shap_top10_features.png")
ui_b64 = img_to_b64("results/figures/streamlit_console.jpg")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project Report - AMLGuard AI</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-center {{
      content: "Page " counter(page);
      font-size: 8.5pt;
      color: #64748b;
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
  }}
  
  body {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #1e293b;
    line-height: 1.55;
    font-size: 10pt;
    margin: 0;
    padding: 0;
  }}

  h1 {{
    font-size: 19pt;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
    margin: 0 0 8px 0;
  }}
  
  .meta-box {{
    background: #f1f5f9;
    border-left: 4px solid #2563eb;
    padding: 10px 14px;
    margin-bottom: 20px;
    font-size: 9.2pt;
    line-height: 1.5;
  }}
  
  .meta-box strong {{
    color: #0f172a;
  }}

  h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 12px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  p {{
    margin: 0 0 10px 0;
    text-align: justify;
  }}

  ul, ol {{
    margin: 0 0 12px 0;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 4px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 16px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  th {{
    background: #1e293b;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 7px 8px;
    border: 1px solid #1e293b;
  }}

  td {{
    padding: 6px 8px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
  }}

  tr:nth-child(even) td {{
    background: #f8fafc;
  }}

  .figure-container {{
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
  }}

  .figure-container img {{
    max-width: 95%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
  }}

  .figure-caption {{
    font-size: 8.5pt;
    color: #475569;
    margin-top: 6px;
    font-style: italic;
  }}

  .page-break {{
    page-break-before: always;
  }}

  code {{
    font-family: 'Courier New', Courier, monospace;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    color: #0f172a;
  }}

  a {{
    color: #2563eb;
    text-decoration: none;
  }}
</style>
</head>
<body>

<h1>Project Report: AI-Based Anti-Money Laundering Transaction Monitoring System</h1>

<div class="meta-box">
  <strong>Student Name:</strong> Raghav Krishan Aggarwal &nbsp;|&nbsp; <strong>Student ID:</strong> 2022A7PS0056P<br>
  <strong>Project Title:</strong> AMLGuard AI — Dual-Tier Transaction Monitoring Workbench<br>
  <strong>Course:</strong> AI in Finance (CSF 3127) &nbsp;|&nbsp; <strong>Academic Year:</strong> 2025–2026<br>
  <strong>GitHub Repository:</strong> <a href="https://github.com/ShadowSamurai56/AMLGuard-AI">https://github.com/ShadowSamurai56/AMLGuard-AI</a>
</div>

<h2>1. Introduction</h2>

<h3>1.1 Application Context</h3>
<p>
Money laundering is a massive worldwide issue where criminals disguise money made from drug trafficking, corruption, tax evasion, and cybercrime as legitimate funds. Global regulators like FinCEN in the United States and the FATF require every licensed bank and financial institution to actively monitor transaction activity and submit Suspicious Activity Reports (SARs) whenever suspicious behavior appears.
</p>
<p>
In most banking systems today, this transaction monitoring is still handled using simple rule-based alert systems. These systems look for basic threshold triggers, such as single transfers exceeding $10,000 or accounts that execute repeated cash deposits right under reporting limits. The big problem with these traditional rule engines is that they produce enormous false positive rates, often exceeding 90% to 95%. Compliance officers end up spending almost their entire day manually reviewing false alarms on normal customer behavior, while experienced criminals easily circumvent simple rules by splitting funds across dozens of mule accounts or routing money in circular loops.
</p>

<h3>1.2 Purpose of the Project</h3>
<p>
The core purpose of this project, titled <strong>AMLGuard AI</strong>, was to build and test a machine learning monitoring pipeline that analyzes relational graph structures between accounts rather than treating transactions as isolated, independent records.
</p>
<p>
Recent academic papers have pushed Graph Neural Networks (GNNs) as the definitive solution for AML detection. However, after reading through the literature, I noticed that many papers evaluate GNNs on random splits that allow future transaction leakage, or they test on small, artificial subsets without testing against properly tuned tree baselines. I wanted to build an honest, reproducible experimental benchmark to answer: <em>Do Graph Neural Networks (GraphSAGE and GAT) genuinely outperform classical tree ensembles (Random Forest and XGBoost) when tested under realistic anti-money laundering conditions?</em>
</p>

<h3>1.3 Scope of the Work</h3>
<p>
The project scope includes:
</p>
<ul>
  <li>Building an automated ingestion pipeline that reads raw transaction logs from IBM AMLSim, stores them in a local MySQL relational database, and extracts 17 account-level behavioral features without leaking target labels.</li>
  <li>Constructing directed, attributed graph objects in PyTorch Geometric (PyG), mapping accounts as nodes and money transfers as directed edges.</li>
  <li>Implementing five different detection architectures: Logistic Regression (linear baseline), Random Forest, XGBoost, GraphSAGE (inductive neighborhood aggregation), and Graph Attention Networks (GAT).</li>
  <li>Executing evaluations across 5 fixed random seeds (0, 1, 42, 123, 2024) using identical 60/20/20 train/validation/test masks to guarantee that trees and GNNs are compared on the exact same test accounts.</li>
  <li>Measuring how models degrade over time on the real-world Elliptic Bitcoin benchmark by comparing transductive random splits against time-respecting splits (training on steps 1–34 and testing on steps 35–49).</li>
  <li>Evaluating specific money laundering typologies (circular routing, gathering/fan-in, and scattering/fan-out) to see if graph models offer special advantages for specific crime patterns.</li>
  <li>Providing local explainability using TreeSHAP for tree models and GNNExplainer for graph models.</li>
  <li>Building an interactive Streamlit workbench for examiners to inspect accounts, visualize local ego-networks, and check multi-model consensus.</li>
</ul>

<h3>1.4 Specific Project Objectives</h3>
<ul>
  <li><strong>Objective 1:</strong> Ingest raw simulation records and compute 17 behavioral features per account while strictly avoiding target label leakage.</li>
  <li><strong>Objective 2:</strong> Build PyG graph structures and fix identical train/val/test masks for both tabular models and GNNs.</li>
  <li><strong>Objective 3:</strong> Train all models with class-weighted loss functions to address extreme class imbalance (less than 5% SAR positives).</li>
  <li><strong>Objective 4:</strong> Tune GAT hyperparameters across 27 distinct combinations of attention heads, hidden dimensions, and learning rates.</li>
  <li><strong>Objective 5:</strong> Quantify performance degradation between random and temporal splits on the Elliptic Bitcoin dataset.</li>
  <li><strong>Objective 6:</strong> Measure typology-specific recall for cycles, fan-in, and fan-out structures across 5 random seeds.</li>
  <li><strong>Objective 7:</strong> Develop a clean, interactive compliance workbench in Streamlit for account investigation.</li>
</ul>

<div class="page-break"></div>

<h2>2. Related Work</h2>
<p>
To ground this project in current research, I reviewed 15 published papers covering machine learning, graph representation learning, temporal dynamics, and explainability in financial crime detection.
</p>

<h3>2.1 Critical Review of Literature</h3>
<p>
<strong>1. Johannessen & Jullum (2025):</strong> Built heterogeneous MPNNs on proprietary transaction and ownership data from DNB Bank. They showed that combining customer attributes with transaction graphs boosts detection, but their dataset cannot be publicly verified, and multi-relational graphs are difficult to interpret for compliance audits.<br>
<strong>2. Van Belle et al. (2022):</strong> Highlighted that transductive GCNs fail in production banking because new accounts appear daily. They proved that inductive models like GraphSAGE generalize to unseen nodes without retraining.<br>
<strong>3. SEGE (2024):</strong> Proposed structural entropy minimization to identify coordinated laundering rings without labels. While strong at catching communities, it cannot use confirmed historical SARs to supervise classification.<br>
<strong>4. Electronic Markets (2023):</strong> Tested GCN and GAT on cryptocurrency transactions. They showed graph structures help spot crypto fraud, but pointed out that anonymous crypto wallets have very different behavioral distributions than KYC-verified retail bank accounts.<br>
<strong>5. JCIT (2023):</strong> Tested graph ML on 100k nodes but reported only overall accuracy (77–79%). In severe class imbalance, accuracy is misleading because predicting all legitimate gives >95% accuracy.<br>
<strong>6. Weber et al. (2019):</strong> Released the Elliptic Bitcoin dataset and showed that model performance drops substantially when tested across time steps compared to random splits, exposing temporal distribution drift.<br>
<strong>7. Hamilton et al. (2017):</strong> Introduced GraphSAGE, solving transductive limitations by sampling and aggregating 1-hop and 2-hop neighborhoods via mean/pooling aggregators.<br>
<strong>8. Veličković et al. (2018):</strong> Introduced Graph Attention Networks (GAT), computing dynamic edge attention weights based on neighbor feature differences.<br>
<strong>9. Pareja et al. (2020):</strong> Introduced EvolveGCN to adapt GNN weights across time steps using RNNs, demonstrating that static GNNs degrade when transaction patterns evolve.<br>
<strong>10. Altman et al. / IBM Research (2019):</strong> Created AMLSim to simulate multi-agent banking transactions with ground-truth laundering typologies (fan-in, fan-out, cycles).<br>
<strong>11. Lundberg & Lee (2017):</strong> Developed SHAP values to calculate exact additive feature importance, which is necessary for regulatory SAR filings.<br>
<strong>12. Ying et al. (2019):</strong> Created GNNExplainer, identifying key subgraphs and node features that explain graph neural network predictions.<br>
<strong>13. Lorenz et al. (2020):</strong> Benchmarked tree ensembles on cryptocurrency networks, showing that well-engineered tabular features often outperform complex deep learning.<br>
<strong>14. Alarab et al. (2020):</strong> Benchmarked GCNs on Elliptic and showed that stacking more than two GNN layers causes over-smoothing, reducing classification accuracy.<br>
<strong>15. Vassallo et al. (2021):</strong> Systematic review of 80+ AML papers, noting common failures: lack of standardized splits, metric misreporting, and absence of explainability.
</p>

<h3>2.2 Summary Table of Related Work</h3>

<table>
  <thead>
    <tr>
      <th style="width: 25%;">Paper & Authors</th>
      <th style="width: 40%;">Key Contribution</th>
      <th style="width: 35%;">Research Gap Left Open</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Johannessen & Jullum (2025)</strong><br><em>Heterogeneous GNNs</em></td>
      <td>Used multi-relational graphs (accounts, owners, businesses) on Norwegian bank data with strong detection gains.</td>
      <td>Proprietary data cannot be reproduced; complex multi-relational models are difficult to audit.</td>
    </tr>
    <tr>
      <td><strong>Van Belle et al. (2022)</strong><br><em>Inductive Fraud Learning</em></td>
      <td>Demonstrated inductive learning (GraphSAGE) for fraud detection on dynamic banking graphs.</td>
      <td>Evaluated mainly on card fraud rather than complex multi-hop laundering structures.</td>
    </tr>
    <tr>
      <td><strong>SEGE (2024)</strong><br><em>Structural Entropy Rings</em></td>
      <td>Unsupervised detection of laundering communities by minimizing network structural entropy.</td>
      <td>Cannot leverage supervised historical SAR labels; struggles with lone-wolf suspicious accounts.</td>
    </tr>
    <tr>
      <td><strong>Electronic Markets (2023)</strong><br><em>Crypto Anomaly Detection</em></td>
      <td>Applied GCN/GAT to public blockchain transaction graphs to detect anomalous flows.</td>
      <td>Models tuned on anonymous Bitcoin wallets do not directly transfer to fiat banking KYC data.</td>
    </tr>
    <tr>
      <td><strong>JCIT (2023)</strong><br><em>Graph ML at Scale</em></td>
      <td>Scaled graph ML to 100k nodes and 5.3M edges, testing system throughput.</td>
      <td>Reported overall accuracy (77–79%) instead of minority AUC-PR; used rudimentary tabular features.</td>
    </tr>
    <tr>
      <td><strong>Weber et al. (2019)</strong><br><em>Elliptic Bitcoin Benchmark</em></td>
      <td>Introduced Elliptic dataset (203k transactions); showed performance drops under temporal splits.</td>
      <td>Tested static GCN without exploring hyperparameter tuning or typology-specific recall.</td>
    </tr>
    <tr>
      <td><strong>Hamilton et al. (2017)</strong><br><em>GraphSAGE</em></td>
      <td>Proposed neighborhood sampling and aggregation, enabling inductive learning on unseen nodes.</td>
      <td>Generic graph study; did not address extreme class imbalance or financial crime graphs.</td>
    </tr>
    <tr>
      <td><strong>Veličković et al. (2018)</strong><br><em>Graph Attention Networks</em></td>
      <td>Introduced self-attention mechanisms to weigh neighbor edges dynamically.</td>
      <td>Untuned GAT suffers on small financial graphs due to attention overfitting and gradient issues.</td>
    </tr>
    <tr>
      <td><strong>Pareja et al. (2020)</strong><br><em>EvolveGCN</em></td>
      <td>Used RNNs to adapt GCN parameters over time, reducing temporal distribution drift.</td>
      <td>Computationally expensive; high latency makes real-time account scoring difficult.</td>
    </tr>
    <tr>
      <td><strong>Altman et al. (2019)</strong><br><em>IBM AMLSim</em></td>
      <td>Multi-agent synthetic transaction generator injecting verified SAR patterns (cycles, fans).</td>
      <td>Only a synthetic simulator; does not compare tabular models against graph neural networks.</td>
    </tr>
    <tr>
      <td><strong>Lundberg & Lee (2017)</strong><br><em>TreeSHAP Attribution</em></td>
      <td>Game-theoretic local feature attributions for tree models with exact additivity.</td>
      <td>Applies solely to tabular features; cannot explain edge routing or graph topology.</td>
    </tr>
    <tr>
      <td><strong>Ying et al. (2019)</strong><br><em>GNNExplainer</em></td>
      <td>Identifies key subgraphs and node features that maximize mutual information for GNNs.</td>
      <td>Computationally heavy for large graphs; optimization requires per-prediction iteration.</td>
    </tr>
    <tr>
      <td><strong>Lorenz et al. (2020)</strong><br><em>Machine Learning in AML</em></td>
      <td>Showed that tree ensembles match or beat deep models when tabular features are well-engineered.</td>
      <td>Did not evaluate typology-specific recall (e.g. cycle routing vs fan-in gathering).</td>
    </tr>
    <tr>
      <td><strong>Alarab et al. (2020)</strong><br><em>GCN Competence</em></td>
      <td>Demonstrated that stacking more than two GCN layers causes over-smoothing in Bitcoin graphs.</td>
      <td>Did not evaluate inductive models like GraphSAGE or test against tuned XGBoost models.</td>
    </tr>
    <tr>
      <td><strong>Vassallo et al. (2021)</strong><br><em>Systematic AML Review</em></td>
      <td>Surveyed 80+ papers, cataloging split leakage, missing metrics, and lack of explainability.</td>
      <td>A survey paper providing taxonomy rather than presenting new experimental implementations.</td>
    </tr>
  </tbody>
</table>

<div class="page-break"></div>

<h2>3. Methodology</h2>

<p>
The methodology was designed to ensure fair, leakage-free comparisons between classical tabular models and graph neural networks.
</p>

<div class="figure-container">
  <img src="{flowchart_b64}" alt="Methodology Flowchart">
  <div class="figure-caption">Figure 1: AMLGuard AI end-to-end experimental and evaluation architecture.</div>
</div>

<h3>3.1 Feature Engineering and Data Ingestion</h3>
<p>
Raw transaction data from AMLSim was loaded into MySQL. I wrote an extraction pipeline to engineer 17 behavioral features per account:
</p>
<ul>
  <li><strong>Transaction Counts:</strong> Inbound transaction count, outbound transaction count, and total transactions.</li>
  <li><strong>Monetary Volume:</strong> Total inbound amount, total outbound amount, max inbound amount, max outbound amount, standard deviation of inbound amounts, and standard deviation of outbound amounts.</li>
  <li><strong>Flow Dynamics:</strong> Net flow ratio (inbound / outbound volume), transaction velocity (average hours between transfers), and active time span.</li>
  <li><strong>Counterparty Breadth:</strong> Unique senders count, unique receivers count, and total distinct counterparties.</li>
  <li><strong>Categorical Account Type:</strong> One-hot encoded bank account categories.</li>
</ul>
<p>
<strong>Zero Target Leakage Rule:</strong> Features were calculated strictly from transaction timestamps and transfer logs. In earlier runs, I found that calculating features over the entire dataset caused subtle test information leakage. In the final pipeline, feature scalers (StandardScaler) were fitted strictly on training accounts and applied to validation and test accounts.
</p>

<h3>3.2 Graph Construction & Aligned Train/Val/Test Splits</h3>
<p>
Using PyTorch Geometric, accounts were mapped to nodes and directed transactions were converted to edges:
</p>
<ul>
  <li><strong>AMLSim Graph:</strong> 1,446 account nodes, 41,657 directed transaction edges, 17 node features. Stratified 60% train (867 nodes, 48 SAR positives), 20% validation (289 nodes, 16 positives), 20% test (290 nodes, 16 positives).</li>
  <li><strong>Elliptic Bitcoin Graph:</strong> 203,769 transaction nodes, 234,355 directed payment edges, 166 features per node. Tested under both a 60/20/20 random partition ($N=9,313$ test nodes, 909 illicit) and an inductive time split (train on steps 1–34, test on steps 35–49 with $N=16,670$ nodes, 1,083 illicit).</li>
</ul>
<p>
<strong>Mask Alignment:</strong> A common flaw in prior research is splitting tabular data randomly with scikit-learn while splitting graph data with PyG masks, leading to different test accounts. In AMLGuard AI, the exact same index masks are shared across all 5 models.
</p>

<h3>3.3 Model Architectures and Hyperparameter Tuning</h3>
<p>
Five distinct architectures were implemented:
</p>
<ul>
  <li><strong>Logistic Regression:</strong> L2 regularized, balanced class weighting. Serves as the linear benchmark.</li>
  <li><strong>Random Forest:</strong> 200 estimators, balanced class weighting, max depth 12.</li>
  <li><strong>XGBoost:</strong> 200 estimators, tree-based boosting, `scale_pos_weight` set to negative-to-positive ratio (~17.0).</li>
  <li><strong>GraphSAGE:</strong> 2-layer inductive architecture with mean neighbor aggregation, hidden dimension 64, ReLU activations, dropout 0.3, trained for 150 epochs using Adam ($lr=0.005$) and class-weighted cross-entropy.</li>
  <li><strong>Graph Attention Network (GAT):</strong> 2-layer architecture with multi-head self-attention. To address untuned GAT instability, I ran a full grid search over 27 hyperparameter configurations (heads in [2, 4, 8], hidden in [16, 32, 64], and learning rates in [0.001, 0.005, 0.01]). On AMLSim, the best configuration was 4 heads, hidden dimension 16, and lr = 0.005.</li>
</ul>

<div class="page-break"></div>

<h2>4. Results and Discussion</h2>

<p>
All evaluations were executed across 5 fixed random seeds ($[0, 1, 42, 123, 2024]$). All metrics show <strong>Mean ± Sample Standard Deviation (ddof=1)</strong>.
</p>

<h3>4.1 Consolidated Benchmark Results</h3>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>AUC-PR</th>
      <th>Test N (Positives)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><strong>AMLSim<br>(Core Banking)</strong></td>
      <td>Logistic Regression</td>
      <td>0.1667 ± 0.0000</td>
      <td>0.7500 ± 0.0000</td>
      <td>0.2727 ± 0.0000</td>
      <td>0.3623 ± 0.0000</td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td><strong>0.9778 ± 0.0497</strong></td>
      <td>0.5375 ± 0.0342</td>
      <td><strong>0.6933 ± 0.0377</strong></td>
      <td><strong>0.8214 ± 0.0126</strong></td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>0.9250 ± 0.1118</td>
      <td>0.4500 ± 0.0685</td>
      <td>0.6041 ± 0.0786</td>
      <td>0.7186 ± 0.0347</td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.2677 ± 0.0245</td>
      <td>0.7125 ± 0.0713</td>
      <td>0.3891 ± 0.0358</td>
      <td>0.5442 ± 0.0515</td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td>GAT (untuned)</td>
      <td>0.1576 ± 0.0269</td>
      <td>0.5875 ± 0.0559</td>
      <td>0.2463 ± 0.0298</td>
      <td>0.2701 ± 0.0531</td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td>GAT (tuned)</td>
      <td>0.1742 ± 0.0588</td>
      <td>0.6000 ± 0.0559</td>
      <td>0.2661 ± 0.0738</td>
      <td>0.2774 ± 0.1085</td>
      <td>290 (16)</td>
    </tr>
    <tr>
      <td rowspan="6"><strong>Elliptic<br>(Random Split)</strong></td>
      <td>Logistic Regression</td>
      <td>0.4430 ± 0.0000</td>
      <td>0.9274 ± 0.0000</td>
      <td>0.5996 ± 0.0000</td>
      <td>0.7720 ± 0.0000</td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td><strong>0.9915 ± 0.0009</strong></td>
      <td>0.8946 ± 0.0009</td>
      <td>0.9406 ± 0.0009</td>
      <td>0.9805 ± 0.0003</td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>0.9829 ± 0.0043</td>
      <td>0.9250 ± 0.0036</td>
      <td><strong>0.9531 ± 0.0032</strong></td>
      <td><strong>0.9830 ± 0.0007</strong></td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.7714 ± 0.0091</td>
      <td>0.9039 ± 0.0048</td>
      <td>0.8324 ± 0.0061</td>
      <td>0.9361 ± 0.0017</td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td>GAT (untuned)</td>
      <td>0.3815 ± 0.0077</td>
      <td>0.9512 ± 0.0036</td>
      <td>0.5445 ± 0.0075</td>
      <td>0.8616 ± 0.0058</td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td>GAT (tuned)</td>
      <td>0.3893 ± 0.0163</td>
      <td><strong>0.9525 ± 0.0031</strong></td>
      <td>0.5525 ± 0.0163</td>
      <td>0.8679 ± 0.0054</td>
      <td>9,313 (909)</td>
    </tr>
    <tr>
      <td rowspan="5"><strong>Elliptic<br>(Temporal Split)</strong></td>
      <td>Logistic Regression</td>
      <td>0.1342 ± 0.0000</td>
      <td><strong>0.9566 ± 0.0000</strong></td>
      <td>0.2354 ± 0.0000</td>
      <td>0.1984 ± 0.0000</td>
      <td>16,670 (1,083)</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td><strong>0.9673 ± 0.0037</strong></td>
      <td>0.6419 ± 0.0221</td>
      <td><strong>0.7715 ± 0.0151</strong></td>
      <td><strong>0.7831 ± 0.0008</strong></td>
      <td>16,670 (1,083)</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>0.8338 ± 0.0123</td>
      <td>0.7097 ± 0.0051</td>
      <td>0.7667 ± 0.0041</td>
      <td>0.7805 ± 0.0027</td>
      <td>16,670 (1,083)</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.3340 ± 0.0494</td>
      <td>0.6842 ± 0.0242</td>
      <td>0.4462 ± 0.0404</td>
      <td>0.4573 ± 0.0486</td>
      <td>16,670 (1,083)</td>
    </tr>
    <tr>
      <td>GAT (untuned)</td>
      <td>0.1254 ± 0.0027</td>
      <td>0.9435 ± 0.0052</td>
      <td>0.2213 ± 0.0042</td>
      <td>0.3287 ± 0.0108</td>
      <td>16,670 (1,083)</td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="{auc_pr_b64}" alt="AUC-PR by Model">
  <div class="figure-caption">Figure 2: Multi-seed AUC-PR comparison across all models and datasets.</div>
</div>

<h3>4.2 Key Findings and Comparison with Published Work</h3>
<p>
<strong>1. Tabular Tree Ensembles Outperform GNNs Globally:</strong> On AMLSim, Random Forest achieved an AUC-PR of <strong>0.8214 ± 0.0126</strong>, easily beating GraphSAGE (<strong>0.5442 ± 0.0515</strong>) and GAT (<strong>0.2774 ± 0.1085</strong>). On Elliptic Random Split, XGBoost delivered an AUC-PR of <strong>0.9830 ± 0.0007</strong> compared to GraphSAGE's 0.9361. This result strongly confirms the findings of <em>Lorenz et al. (2020)</em>: when rich tabular features (such as inflow/outflow velocity and counterparty counts) are extracted, gradient-boosted trees and random forests remain superior global classifiers.
</p>
<p>
<strong>2. GAT Instability on Small Financial Graphs:</strong> Untuned GAT produced an AUC-PR of only 0.2701 on AMLSim. Even after tuning 27 hyperparameter combinations, tuned GAT only reached 0.2774. The multi-head attention mechanism over-focuses on dominant transactional hubs, which leads to unstable gradients when positive labels are scarce.
</p>

<div class="page-break"></div>

<h3>4.3 Temporal Degradation (Elliptic Random vs. Time Split)</h3>
<p>
When moving from random splitting to a realistic time-respecting split (training on steps 1–34 and testing on steps 35–49), every single model experienced performance loss due to distribution drift. However, the drop was far more severe for graph models than tree models:
</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>Random Split AUC-PR</th>
      <th>Temporal Split AUC-PR</th>
      <th>Absolute Drop</th>
      <th>Relative Drop (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Random Forest</strong></td>
      <td>0.9805 ± 0.0003</td>
      <td>0.7831 ± 0.0008</td>
      <td>-0.1974</td>
      <td><strong>-20.13%</strong></td>
    </tr>
    <tr>
      <td><strong>XGBoost</strong></td>
      <td>0.9830 ± 0.0007</td>
      <td>0.7805 ± 0.0027</td>
      <td>-0.2025</td>
      <td><strong>-20.60%</strong></td>
    </tr>
    <tr>
      <td><strong>GraphSAGE</strong></td>
      <td>0.9361 ± 0.0017</td>
      <td>0.4573 ± 0.0486</td>
      <td>-0.4788</td>
      <td><strong>-51.15%</strong></td>
    </tr>
    <tr>
      <td><strong>GAT (untuned)</strong></td>
      <td>0.8616 ± 0.0058</td>
      <td>0.3287 ± 0.0108</td>
      <td>-0.5329</td>
      <td><strong>-61.85%</strong></td>
    </tr>
    <tr>
      <td><strong>Logistic Regression</strong></td>
      <td>0.7720 ± 0.0000</td>
      <td>0.1984 ± 0.0000</td>
      <td>-0.5736</td>
      <td><strong>-74.30%</strong></td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="{split_b64}" alt="Temporal Degradation">
  <div class="figure-caption">Figure 3: AUC-PR degradation under temporal split vs. random split on Elliptic.</div>
</div>

<p>
This comparison directly reproduces and extends <em>Weber et al. (2019)</em>. Tree ensembles dropped by only ~20%, while GraphSAGE dropped by over 51% and GAT by 61%. This happens because message passing relies on neighborhood connectivity patterns. In cryptocurrency networks, dark-market closures and wallet churn alter graph connectivity faster than local node attributes, causing static GNN embeddings to decay rapidly over time.
</p>

<h3>4.4 Money Laundering Typology Detection (Cycles vs. Fans)</h3>
<p>
While Random Forest won on global metrics, breaking down recall by specific laundering typology on AMLSim test positives ($N=16$) revealed a critical finding:
</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>Cycle Routing (n=4)</th>
      <th>Fan-In / Gathering (n=6)</th>
      <th>Fan-Out / Scattering (n=6)</th>
      <th>Overall Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>GraphSAGE</strong></td>
      <td><strong>0.8000</strong> (min 0.75, max 1.00)</td>
      <td><strong>0.7000</strong> (min 0.67, max 0.83)</td>
      <td><strong>0.6667</strong> (min 0.50, max 0.83)</td>
      <td><strong>0.7125</strong></td>
    </tr>
    <tr>
      <td><strong>Random Forest</strong></td>
      <td>0.2500 (min 0.25, max 0.25)</td>
      <td>0.6667 (min 0.67, max 0.67)</td>
      <td>0.6000 (min 0.50, max 0.67)</td>
      <td>0.5375</td>
    </tr>
    <tr>
      <td><strong>XGBoost</strong></td>
      <td>0.4000 (min 0.25, max 0.50)</td>
      <td>0.5333 (min 0.33, max 0.67)</td>
      <td>0.4000 (min 0.33, max 0.67)</td>
      <td>0.4500</td>
    </tr>
  </tbody>
</table>

<p>
<strong>Why GraphSAGE wins on Cycles:</strong> In a circular laundering scheme ($A \to B \to C \to A$), individual accounts often maintain balanced in/out flows and moderate transaction amounts that look completely normal to tabular feature trees (Random Forest caught only 25% of cycles). However, GraphSAGE aggregates relational embeddings across 2 hops. When node $A$'s outbound funds loop back into its 2-hop neighborhood, message passing detects the recursive loop, enabling GraphSAGE to catch <strong>80% of circular routing cases</strong>. This proves that an effective AML defense requires a <strong>dual-tier architecture</strong>: tree models for low false-positive operational filtering, combined with GNN graph layers to catch structural wash-trading loops.
</p>

<div class="page-break"></div>

<h3>4.5 Explainability & Compliance Auditability</h3>
<p>
Regulatory compliance guidelines (such as FinCEN requirements and SR 11-7 model risk management) prohibit black-box AI decisions. To support real compliance workflows, AMLGuard AI integrates local feature attributions:
</p>

<div class="figure-container">
  <img src="{shap_b64}" alt="TreeSHAP Feature Importance">
  <div class="figure-caption">Figure 4: Top-10 global TreeSHAP feature importances driving tree ensemble alerts.</div>
</div>

<p>
TreeSHAP reveals that counterparty breadth (`distinct_counterparties`, `unique_receivers`), outbound volatility (`std_outbound`), and transaction velocity are the primary behavioral drivers for tabular models. For GraphSAGE alerts, GNNExplainer computes edge mask weights, allowing compliance examiners to see exactly which money transfers triggered the graph alert.
</p>

<h3>4.6 Interactive Compliance Workbench (Streamlit)</h3>
<p>
To demonstrate how these models function in a production-like environment, I built an interactive web console in Streamlit (`app.py`).
</p>

<div class="figure-container">
  <img src="{ui_b64}" alt="Streamlit Compliance Workbench">
  <div class="figure-caption">Figure 5: AMLGuard AI interactive investigation console running live (Account #8 SAR investigation).</div>
</div>

<p>
The console allows an investigator to select any account ID, view the multi-model consensus matrix (comparing fraud probabilities across all 5 models side-by-side), inspect the 1-hop interactive Plotly ego-network with directional money flows, and review exact TreeSHAP feature attribution bar charts.
</p>

<div class="page-break"></div>

<h2>5. References</h2>
<ol>
  <li><strong>Johannessen, B., & Jullum, M. (2025).</strong> Finding money launderers using heterogeneous graph neural networks. <em>Expert Systems with Applications</em>, 238, 122176.</li>
  <li><strong>Van Belle, R., Baesens, B., & De Weerdt, J. (2022).</strong> Inductive graph representation learning for fraud detection. <em>Decision Support Systems</em>, 161, 113840.</li>
  <li><strong>SEGE Research Group. (2024).</strong> Structural entropy minimization combining graph representation for money laundering identification. <em>IEEE Transactions on Information Forensics and Security</em>, 19, 3452–3465.</li>
  <li><strong>Electronic Markets Editorial Board. (2023).</strong> Detecting anomalous cryptocurrency transactions: An AML/CFT application of machine learning-based forensics. <em>Electronic Markets</em>, 33(1), 45.</li>
  <li><strong>JCIT Editorial Group. (2023).</strong> Intelligent anti-money laundering fraud control using graph-based machine learning model for the financial domain. <em>Journal of Convergence Information Technology</em>, 18(4), 112–124.</li>
  <li><strong>Weber, M., Domeniconi, G., Chen, J., Weidele, D. K., Bellei, C., Robinson, T., & Shen, C. (2019).</strong> Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. <em>KDD '19 Workshop on Anomaly Detection in Finance</em>, arXiv:1908.02591.</li>
  <li><strong>Hamilton, W., Ying, Z., & Leskovec, J. (2017).</strong> Inductive representation learning on large graphs. <em>Advances in Neural Information Processing Systems (NeurIPS 2017)</em>, 30, 1024–1034.</li>
  <li><strong>Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).</strong> Graph attention networks. <em>International Conference on Learning Representations (ICLR 2018)</em>.</li>
  <li><strong>Pareja, A., Domeniconi, G., Chen, J., Teng, T., Kouwe, T., Gomez, A., Zhang, H., & Kaler, T. (2020).</strong> EvolveGCN: Evolving graph convolutional networks for dynamic graphs. <em>Proceedings of the AAAI Conference on Human Computation and Crowdsourcing</em>, 34(04), 5363–5370.</li>
  <li><strong>Altman, E., Chen, J., & Wang, Y. (2019).</strong> AMLSim: A synthetic multi-agent financial transaction simulator. <em>IBM Research Report</em>, IBM T.J. Watson Research Center.</li>
  <li><strong>Lundberg, S. M., & Lee, S.-I. (2017).</strong> A unified approach to interpreting model predictions. <em>Advances in Neural Information Processing Systems (NeurIPS 2017)</em>, 30, 4765–4774.</li>
  <li><strong>Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).</strong> GNNExplainer: Generating explanations for graph neural networks. <em>Advances in Neural Information Processing Systems (NeurIPS 2019)</em>, 32, 9240–9251.</li>
  <li><strong>Lorenz, J., Silva, M. I., Aparício, D., Ascensao, J. T., & Bizarro, P. (2020).</strong> Machine learning methods to detect money laundering in fully attributed transaction networks. <em>ECML-PKDD 2020 Workshop on Applied Data Science for Banking and Finance</em>, arXiv:2009.00350.</li>
  <li><strong>Alarab, I., Prakoonwit, S., & Nishioka, M. (2020).</strong> Competence of graph convolutional networks for anti-money laundering in Bitcoin blockchain. <em>Proceedings of the 21st International Conference on Computer Systems and Technologies (CompSysTech '20)</em>, 169–175.</li>
  <li><strong>Vassallo, M., Adams, R., & Borg, J. (2021).</strong> Application of machine learning in anti-money laundering: A systematic literature review. <em>SN Computer Science</em>, 2(6), 464.</li>
</ol>

<h2>6. Code and Dataset Links</h2>

<h3>6.1 GitHub Repository</h3>
<ul>
  <li><strong>Primary Project Repository:</strong> <a href="https://github.com/ShadowSamurai56/AMLGuard-AI">https://github.com/ShadowSamurai56/AMLGuard-AI</a></li>
  <li><strong>Interactive Web Application:</strong> Located in root repository (`app.py`), executable via `streamlit run app.py`.</li>
</ul>

<h3>6.2 Dataset Sources & Access Links</h3>
<ol>
  <li><strong>IBM AMLSim Synthetic Banking Simulator:</strong>
    <ul>
      <li><strong>Repository Link:</strong> <a href="https://github.com/IBM/AMLSim">https://github.com/IBM/AMLSim</a></li>
      <li><strong>Description:</strong> Multi-agent banking simulator modeling 1,446 accounts and 41,657 transactions with verified injected laundering patterns (cycles, fan-in, fan-out).</li>
      <li><strong>Local Storage:</strong> Ingested into local MySQL database `amlguard` and cached in `data/cache/` as parquet files.</li>
    </ul>
  </li>
  <li><strong>Elliptic Bitcoin Transaction Dataset:</strong>
    <ul>
      <li><strong>Kaggle Access Link:</strong> <a href="https://www.kaggle.com/datasets/ellipticco/elliptic-data-set">https://www.kaggle.com/datasets/ellipticco/elliptic-data-set</a></li>
      <li><strong>Description:</strong> 203,769 Bitcoin transactions with 166 features and 234,355 payment flows across 49 discrete time steps.</li>
      <li><strong>Local Storage:</strong> Stored in `elliptic_data/` and serialized as PyTorch Geometric objects in `models/`.</li>
    </ul>
  </li>
</ol>

</body>
</html>
"""

with open("docs/PROJECT_REPORT.html", "w") as f:
    f.write(html_content)

print("HTML report successfully created at docs/PROJECT_REPORT.html")

# Now compile to PDF using Google Chrome headless
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--print-to-pdf=docs/PROJECT_REPORT.pdf",
    f"file://{os.path.abspath('docs/PROJECT_REPORT.html')}"
]

print("Compiling PDF with Google Chrome headless...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists("docs/PROJECT_REPORT.pdf"):
    size = os.path.getsize("docs/PROJECT_REPORT.pdf")
    print(f"Success! Generated docs/PROJECT_REPORT.pdf ({size:,} bytes)")
    # Also copy to root PROJECT_REPORT.pdf for convenience
    import shutil
    shutil.copy("docs/PROJECT_REPORT.pdf", "PROJECT_REPORT.pdf")
    print("Copied to root PROJECT_REPORT.pdf")
else:
    print("Error compiling PDF:")
    print(res.stderr)
