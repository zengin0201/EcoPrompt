## 📌 Problem Statement & Green AI
The explosive growth of LLMs has led to massive energy consumption in global data centers. Standard AI architectures route all user inputs to state-of-the-art models (e.g., 70B+ parameter giants), even for trivial tasks (greeting, classification, basic Q&A). This approach wastes GPU compute, increases latency, and inflates cloud API costs.

**EcoPrompt** solves this by implementing an intelligent middleware router that classifies prompt complexity before calling any API, optimizing the trade-off between output accuracy and computational cost.

---

## ⚙️ System Architecture

EcoPrompt evaluates prompt complexity using high-dimensional TF-IDF vectorization and a Random Forest Classifier trained on task-representative datasets.

```text
       [User Input Query]
               │
               ▼
   [EcoPrompt Router (ML Engine)]  ◄── (Sub-millisecond inference)
               │
       ┌───────┴───────┐
(Complexity = 0)  (Complexity = 1)
       │               │
       ▼               ▼
 [Llama-3.1-8B]  [Llama-3.3-70B]
 (Fast & Cheap)  (Heavy Reasoning)
```

*   **Lightweight Tier:** `llama-3.1-8b-instant` — optimized for fast, low-cost execution of factual retrieval, translation, and classification.
*   **Heavy Tier:** `llama-3.3-70b-versatile` — reserved for advanced synthesis, brainstorming, summarization, and code generation.

---

## 📊 Empirical Results
The framework was evaluated on a subset of the `Databricks-dolly-15k` dataset. 

| Evaluation Metric | Baseline (100% Llama-70B) | EcoPrompt Framework | Net Improvement |
| :--- | :--- | :--- | :--- |
| **Routing Decision Latency** | N/A | < 1.0 ms | **Negligible Overhead** |
| **Simple Task Cost (Avg)** | $0.0001395 | $0.0000147 | **🌿 90.2% Cost Savings** |
| **End-to-End Latency** | 1.63s | 0.58s | **⚡ 64.4% Latency Reduction** |

---

## 🚀 Local Installation & Usage

### Prerequisites
* Python 3.11 or higher
* Groq API Key (Free tier from [Groq Console](https://console.groq.com/))

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EcoPrompt.git
   cd EcoPrompt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_api_key_here
   ```

4. Run the command-line benchmark:
   ```bash
   python main.py
   ```

5. Run the interactive Streamlit web application:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Cloud Deployment (Streamlit Share)

You can deploy this interactive dashboard to the web for free using Streamlit Cloud.

### Step 1: Push Code to GitHub
Ensure your repository is public and contains the following files:
* `main.py`
* `app.py`
* `requirements.txt`
* *(Do NOT push your `.env` file to GitHub to protect your API key).*

### Step 2: Configure Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New App"**, then select your `EcoPrompt` repository, branch, and set `app.py` as the main file path.
3. Click **"Advanced Settings"** before deploying.
4. In the **"Secrets"** text area, paste your Groq API key securely:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_api_key_here"
   ```
5. Click **"Save"** and then **"Deploy"**. Your live application will be online in under 2 minutes!

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Machine Learning:** Scikit-Learn (TF-IDF, Random Forest)
* **Data Handling:** Pandas, NumPy, HuggingFace Datasets
* **Inference Platform:** Groq Cloud API SDK
* **Frontend UI:** Streamlit Web Framework

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
```
