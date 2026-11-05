"""
Generate a professional, ATS-friendly single-page resume PDF for Raghav Krishan Aggarwal
incorporating the AMLGuard AI project with quantifiable metrics and live deployment links.
"""

import os
import subprocess

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Resume - Raghav Krishan Aggarwal</title>
<style>
  @page {
    size: A4;
    margin: 9mm 13mm 9mm 13mm;
  }

  * {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827;
    line-height: 1.32;
    font-size: 8.8pt;
    margin: 0;
    padding: 0;
  }

  /* Header */
  .header {
    text-align: center;
    border-bottom: 1.5px solid #1e3a8a;
    padding-bottom: 5px;
    margin-bottom: 7px;
  }

  .name {
    font-size: 16pt;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #0f172a;
    margin-bottom: 3px;
    text-transform: uppercase;
  }

  .contact-bar {
    font-size: 8.5pt;
    color: #334155;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
  }

  .contact-bar a {
    color: #1d4ed8;
    text-decoration: none;
    font-weight: 500;
  }

  /* Section Headings */
  .section-title {
    font-size: 9.8pt;
    font-weight: 700;
    color: #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 6px;
    margin-bottom: 4px;
  }

  /* Items */
  .item {
    margin-bottom: 5px;
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 8.8pt;
  }

  .item-title {
    font-weight: 700;
    color: #0f172a;
  }

  .item-subtitle {
    font-style: italic;
    color: #475569;
    font-size: 8.5pt;
  }

  .item-date {
    font-size: 8.2pt;
    color: #64748b;
    font-weight: 600;
    white-space: nowrap;
  }

  /* Bullets */
  ul {
    margin: 2px 0 3px 0;
    padding-left: 15px;
  }

  li {
    margin-bottom: 2px;
    color: #1e293b;
    text-align: justify;
  }

  li strong {
    color: #0f172a;
  }

  /* Academics Table */
  table.edu-table {
    width: 100%;
    border-collapse: collapse;
    margin: 3px 0 5px 0;
    font-size: 8.2pt;
  }

  table.edu-table th {
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 3px 6px;
    border: 1px solid #cbd5e1;
  }

  table.edu-table td {
    padding: 3px 6px;
    border: 1px solid #e2e8f0;
    color: #1e293b;
  }

  /* Skills */
  .skill-group {
    margin-bottom: 2.5px;
    font-size: 8.5pt;
    line-height: 1.35;
  }

  .skill-label {
    font-weight: 700;
    color: #0f172a;
    display: inline-block;
    min-width: 130px;
  }

  .project-tagline {
    font-size: 8.2pt;
    color: #2563eb;
    font-weight: 600;
    margin-bottom: 2px;
  }
</style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="name">Raghav Krishan Aggarwal</div>
    <div class="contact-bar">
      <span>📞 +91-9456054000</span>
      <span>•</span>
      <span>✉️ <a href="mailto:raghavag112@gmail.com">raghavag112@gmail.com</a></span>
      <span>•</span>
      <span>🔗 <a href="https://linkedin.com/in/raghavkaggarwal">linkedin.com/in/raghavkaggarwal</a></span>
      <span>•</span>
      <span>💻 <a href="https://github.com/ShadowSamurai56">github.com/ShadowSamurai56</a></span>
    </div>
  </div>

  <!-- Education -->
  <div class="section-title">Education</div>
  <table class="edu-table">
    <thead>
      <tr>
        <th style="width: 25%;">Degree / Qualification</th>
        <th style="width: 38%;">Institute</th>
        <th style="width: 20%;">Board / University</th>
        <th style="width: 10%;">Score</th>
        <th style="width: 7%;">Year</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>B.Tech (CSFT)</strong></td>
        <td>Manipal Institute of Technology, Manipal</td>
        <td>MAHE</td>
        <td><strong>7.22 CGPA</strong></td>
        <td>2024–28</td>
      </tr>
      <tr>
        <td>Class XII (CBSE)</td>
        <td>Sacred Heart Convent School, Ludhiana</td>
        <td>CBSE Board</td>
        <td><strong>92.4%</strong></td>
        <td>2024</td>
      </tr>
      <tr>
        <td>Class X (CBSE)</td>
        <td>The Lawrence School, Sanawar</td>
        <td>CBSE Board</td>
        <td><strong>95.6%</strong></td>
        <td>2022</td>
      </tr>
    </tbody>
  </table>

  <!-- Featured Projects -->
  <div class="section-title">Technical Projects</div>
  <div class="item">
    <div class="item-header">
      <div>
        <span class="item-title">AMLGuard AI — Relational Anti-Money Laundering Monitoring Platform</span>
        <span class="item-subtitle"> | Python, PyTorch Geometric, XGBoost, Streamlit, NetworkX</span>
      </div>
      <span class="item-date">
        <a href="https://github.com/ShadowSamurai56/AMLGuard-AI" style="color: #1d4ed8; text-decoration: none;">[GitHub]</a>
      </span>
    </div>
    <ul>
      <li>Architected an end-to-end AML transaction surveillance engine evaluating <strong>Graph Neural Networks (GraphSAGE, GAT)</strong> vs. <strong>classical tree ensembles (Random Forest, XGBoost)</strong> across 5 random seeds on <strong>AMLSim</strong> and <strong>Elliptic Bitcoin (203k txns)</strong>.</li>
      <li>Discovered that 2-hop message-passing enables <strong>GraphSAGE to achieve 80% recall on complex circular routing rings</strong> (16/20 detections across 5 seeds, vs. 25% for Random Forest), while tree models delivered superior global precision and <strong>AUC-PR (0.8214 vs. 0.5442)</strong> on tabular account behaviors.</li>
      <li>Formulated a zero-leakage temporal split pipeline (steps 1–34 train, 35–49 test); analyzed why GNNs degraded by <strong>51.2%</strong> under cryptocurrency concept drift compared to only <strong>20.1%</strong> degradation in gradient-boosted trees.</li>
      <li>Engineered an interactive <strong>Streamlit workbench</strong> rendering 1-hop Plotly ego-networks, real-time multi-model consensus, and <strong>TreeSHAP / GNNExplainer</strong> feature attributions fulfilling regulatory model governance standards (SR 11-7).</li>
    </ul>
  </div>

  <!-- Internships -->
  <div class="section-title">Internship Experience</div>

  <div class="item">
    <div class="item-header">
      <div>
        <span class="item-title">Apollo Pipes Ltd.</span>
        <span class="item-subtitle"> — Information Technology Analyst Intern</span>
      </div>
      <span class="item-date">Jun '26 – Jul '26 (2 Months)</span>
    </div>
    <ul>
      <li>Engineered and optimized internal business web applications using <strong>Laravel, PHP, and MySQL</strong>, reducing page load latencies.</li>
      <li>Architected normalized relational database schemas (ER diagrams) ensuring data integrity across supply-chain and inventory modules.</li>
      <li>Configured and maintained local testing environments with <strong>XAMPP</strong>, conducting API debugging and systematic defect resolution.</li>
    </ul>
  </div>

  <div class="item">
    <div class="item-header">
      <div>
        <span class="item-title">Krishna Honda</span>
        <span class="item-subtitle"> — Marketing & Customer Relations Intern</span>
      </div>
      <span class="item-date">Dec '25 – Jan '26 (1 Month)</span>
    </div>
    <ul>
      <li>Interfaced directly with retail clients to explain automobile technical specifications, loan financing schemes, and insurance structures.</li>
      <li>Maintained relational client payment databases, automated payment follow-up schedules, and supported promotional drives.</li>
    </ul>
  </div>

  <div class="item">
    <div class="item-header">
      <div>
        <span class="item-title">Krishna Steel Rolling Mill</span>
        <span class="item-subtitle"> — Product Management Intern</span>
      </div>
      <span class="item-date">Jun '25 – Jul '25 (1.5 Months)</span>
    </div>
    <ul>
      <li>Structured operational tracking systems to monitor raw material throughput and finished inventory across multi-stage steel rolling operations.</li>
      <li>Assisted in daily transaction billing audits and streamlined operational reporting logs for management accounting.</li>
    </ul>
  </div>

  <!-- Technical Skills -->
  <div class="section-title">Technical Skills</div>
  <div class="skill-group">
    <span class="skill-label">Programming Languages:</span>
    <span>Python, Java, PHP, SQL, C/C++</span>
  </div>
  <div class="skill-group">
    <span class="skill-label">AI & Graph Analytics:</span>
    <span>PyTorch Geometric (PyG), scikit-learn, XGBoost, NetworkX, TreeSHAP, GNNExplainer, Pandas, NumPy</span>
  </div>
  <div class="skill-group">
    <span class="skill-label">Frameworks & Tools:</span>
    <span>Streamlit, Plotly, Spring Boot, Laravel, MySQL, Postman, Git & GitHub, XAMPP</span>
  </div>
  <div class="skill-group">
    <span class="skill-label">Core Concepts:</span>
    <span>Data Structures & Algorithms, OOP, RESTful API Architecture, Database Normalization & ERD, Model Risk Governance (SR 11-7)</span>
  </div>
  <div class="skill-group">
    <span class="skill-label">Soft Skills:</span>
    <span>Technical Problem Solving, Team Leadership, Cross-Functional Communication, Pipeline Tracking</span>
  </div>

  <!-- Positions of Responsibility -->
  <div class="section-title">Leadership & Extra-Curriculars</div>
  <div class="item">
    <div class="item-header">
      <div>
        <span class="item-title">AIESEC in M.A.H.E.</span>
        <span class="item-subtitle"> — Senior Manager (Global Volunteer Operations)</span>
      </div>
      <span class="item-date">Manipal, India</span>
    </div>
    <ul>
      <li>Led a student team driving youth engagement, screening candidates, and coordinating end-to-end logistics for international exchange delegates.</li>
      <li>Streamlined pipeline conversion tracking, enhancing stakeholder communication and team operational productivity.</li>
    </ul>
  </div>

</body>
</html>
"""

html_path = "Raghav_Krishan_Aggarwal_Resume.html"
pdf_path = "Raghav_Krishan_Aggarwal_Resume.pdf"

with open(html_path, "w") as f:
    f.write(html_content)

print(f"HTML generated at {html_path}")

chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--user-data-dir=/tmp/chrome_resume_profile",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}",
    f"file://{os.path.abspath(html_path)}"
]

print("Compiling PDF with Google Chrome headless...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists(pdf_path):
    size = os.path.getsize(pdf_path)
    print(f"Success! Generated {pdf_path} ({size:,} bytes)")
else:
    print("Error compiling PDF:")
    print(res.stderr)

