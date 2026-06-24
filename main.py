import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from groq import Groq

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Error: GROQ_API_KEY is missing in the .env file.")

groq_client = Groq(api_key=API_KEY)

# === ML ROUTER MODULE ===
class EcoRouter:
    def __init__(self):
        # TF-IDF + Random Forest setup for lightweight, low-compute classification
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, df):
        print("Training the EcoRouter model...")
        
        # Convert text prompts to numerical features
        X_text = self.vectorizer.fit_transform(df['prompt']).toarray()
        
        # Add prompt length as an additional feature
        lengths = np.array([len(str(t).split()) for t in df['prompt']]).reshape(-1, 1)
        X_final = np.hstack((X_text, lengths))
        
        self.classifier.fit(X_final, df['label'])
        self.is_trained = True
        print("Model training completed.")

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Router must be trained before inference.")
            
        vec = self.vectorizer.transform([text]).toarray()
        length = np.array([[len(text.split())]])
        X_new = np.hstack((vec, length))
        
        return self.classifier.predict(X_new)[0]

# === LLM API INTEGRATION WITH DYNAMIC COST CALCULATION ===
def call_heavy_model(prompt):
    """
    Calls Llama-3.3-70B.
    Pricing: Input = $0.59 / 1M tokens, Output = $0.79 / 1M tokens.
    """
    start_time = time.time()
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt[:500]}],
            max_tokens=150
        )
        latency = time.time() - start_time
        
        # Calculate dynamic cost based on actual token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = (prompt_tokens * 0.00000059) + (completion_tokens * 0.00000079)
        
        return response.choices[0].message.content, latency, cost
    except Exception as e:
        print(f"Heavy model API failed: {e}")
        return "Error calling Heavy Model", 0, 0

def call_light_model(prompt):
    """
    Calls Llama-3.1-8B.
    Pricing: Input = $0.05 / 1M tokens, Output = $0.08 / 1M tokens.
    Also returns the equivalent cost of running this request on the heavy model.
    """
    start_time = time.time()
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt[:500]}],
            max_tokens=150
        )
        latency = time.time() - start_time
        
        # Calculate dynamic cost based on actual token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = (prompt_tokens * 0.00000005) + (completion_tokens * 0.00000008)
        
        # Calculate equivalent cost for Llama-70B (for savings metrics)
        heavy_equivalent_cost = (prompt_tokens * 0.00000059) + (completion_tokens * 0.00000079)
        
        return response.choices[0].message.content, latency, cost, heavy_equivalent_cost
    except Exception as e:
        print(f"Light model API failed: {e}")
        return "Error calling Light Model", 0, 0, 0

# === BENCHMARK EXPERIMENT ===
def run_benchmark(prompts, router):
    print("\n" + "="*50)
    print("Starting API Benchmark Experiment...")
    print("="*50)
    
    results = {
        "baseline": {"latency": 0, "cost": 0},
        "eco_routed": {"latency": 0, "cost": 0}
    }

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}...", end="\r")
        
        # 1. Baseline approach: Route everything to Llama-3.3-70B
        _, t_heavy, c_heavy = call_heavy_model(prompt)
        results["baseline"]["latency"] += t_heavy
        results["baseline"]["cost"] += c_heavy

        # 2. EcoPrompt approach: Smart routing
        route_class = router.predict(prompt)
        if route_class == 1:
            _, t_route, c_route = call_heavy_model(prompt)
        else:
            _, t_route, c_route, _ = call_light_model(prompt)
            
        results["eco_routed"]["latency"] += t_route
        results["eco_routed"]["cost"] += c_route
        
        # Sleep to avoid rate limits on the free API tier
        time.sleep(1)

    print("\n\nExperiment finished successfully.\n")

    # Output results for the research paper
    print("--- BENCHMARK RESULTS ---")
    print("1. Baseline (100% Llama-70B):")
    print(f"   Total Latency: {results['baseline']['latency']:.2f} sec")
    print(f"   Compute Cost:  ${results['baseline']['cost']:.6f}")
    
    print("\n2. EcoPrompt Framework (Dynamic Routing):")
    print(f"   Total Latency: {results['eco_routed']['latency']:.2f} sec")
    print(f"   Compute Cost:  ${results['eco_routed']['cost']:.6f}")
    print("-" * 50)
    
    if results['baseline']['cost'] > 0:
        cost_savings = (1 - results['eco_routed']['cost'] / results['baseline']['cost']) * 100
        time_savings = (1 - results['eco_routed']['latency'] / results['baseline']['latency']) * 100
        
        print("KEY FINDINGS:")
        print(f"-> Compute/Energy Savings: {cost_savings:.1f}%")
        print(f"-> Latency Reduction:      {time_savings:.1f}%")
        print("="*50)


if __name__ == "__main__":
    print("Fetching Databricks Dolly 15k dataset...")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = dataset.to_pandas()
    
    # Labeling logic
    complex_categories = ['brainstorming', 'generation', 'summarization', 'information_extraction']
    df['prompt'] = df.apply(
        lambda row: f"{row['instruction']}\n{row['context']}" if pd.notna(row['context']) and str(row['context']).strip() != "" else str(row['instruction']), 
        axis=1
    )
    df['label'] = df['category'].apply(lambda x: 1 if x in complex_categories else 0)

    # Train model
    train_df = df.sample(n=1000, random_state=42)
    router = EcoRouter()
    router.train(train_df)

    # Run Benchmark
    test_df = df.drop(train_df.index).sample(n=25, random_state=777)
    test_prompts = test_df['prompt'].tolist()
    run_benchmark(test_prompts, router)


    
