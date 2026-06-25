# 🌿 EcoPrompt: Intelligent Semantic Routing for Energy-Efficient LLM Inference

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)

EcoPrompt is an open-source, lightweight Machine Learning framework designed to reduce the operational expenditures (OpEx) and computational carbon footprint of Large Language Model (LLM) applications. 

By analyzing the semantic and syntactic complexity of user prompts in sub-milliseconds, EcoPrompt dynamically orchestrates inference between small, eco-friendly models and large, high-reasoning models.

---

## 📌 Problem Statement & Green AI
The explosive growth of LLMs has led to massive energy consumption in global data centers. Standard AI architectures route all user inputs to state-of-the-art models (e.g., 70B+ parameter giants), even for trivial tasks (greeting, classification, basic Q&A). This approach wastes GPU compute, increases latency, and inflates cloud API costs.

**EcoPrompt** solves this by implementing an intelligent middleware router that classifies prompt complexity before calling any API, optimizing the trade-off between output accuracy and computational cost.

---

## ⚙️ System Architecture

EcoPrompt evaluates prompt complexity using **all-MiniLM-L6-v2 semantic embeddings** and a balanced Random Forest Classifier trained on task-representative datasets.

```text
       [User Input Query]
               │
               ▼
   [EcoPrompt Router (ML Engine)]  ◄── (Sub-millisecond inference via MiniLM)
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
The framework was evaluated on the `Databricks-dolly-15k` dataset. 

| Evaluation Metric | Baseline (100% Llama-70B) | EcoPrompt Framework | Net Improvement |
| :--- | :--- | :--- | :--- |
| **Routing Decision Latency** | N/A | < 1.0 ms | **Negligible Overhead** |
| **Simulated Cost (600 Queries)** | $0.1035 | $0.0392 | **🌿 62.1% Cost Optimization** |

*(Note: The model is specifically balanced to ensure complex tasks are properly routed to the 70B parameter model, preserving high output accuracy while still achieving over 60% compute savings).*

---

## 🧩 How to Integrate into Your Own Project

If you want to use the EcoPrompt router inside your own custom backend or chatbot, you can import the `EcoRouter` class directly:

```python
from router import EcoRouter

# 1. Initialize and load the pre-trained router
router = EcoRouter()
router.load("router_model.pkl")

# 2. Use the router to dynamically orchestrate inference in your app
user_query = "Write a convolutional neural network in Python from scratch"
decision = router.predict(user_query)

if decision == 1:
    print("Routing to heavy model (Llama-70B)...")
    # Call your heavy API here
else:
    print("Routing to light model (Llama-8B)...")
    # Call your light API here
```

---

## 🚀 Local Installation & Usage

### Prerequisites
* Python 3.11 or higher
* Groq API Key (Free tier from [Groq Console](https://console.groq.com/))

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/zengin0201/EcoPrompt.git
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

4. Run the training script to generate the model weights (`router_model.pkl`) and academic plots:
   ```bash
   python train_benchmark.py
   ```

5. Run the interactive Streamlit web application:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Cloud Deployment (Streamlit Share)

You can deploy this interactive dashboard to the web for free using Streamlit Cloud. Ensure your repository contains:
* `app.py`
* `router.py`
* `router_model.pkl`
* `requirements.txt`
*(Do NOT push your `.env` file to GitHub).*

Set up your secrets in Streamlit Cloud Settings:
```toml
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

---

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn (Random Forest), Sentence-Transformers (MiniLM)
* **Data Handling:** Pandas, NumPy, HuggingFace Datasets
* **Inference Platform:** Groq Cloud API SDK
* **Frontend UI:** Streamlit Web Framework

## 📄 License
This project is licensed under the MIT License.
```
