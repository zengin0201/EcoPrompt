import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from router import EcoRouter

# Academic styling for the plots
sns.set_theme(style="whitegrid", palette="muted")

def prepare_data():
    print("Loading Databricks Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = dataset.to_pandas()
    
    # Map categories to binary complexity labels
    complex_tasks = ['brainstorming', 'generation', 'summarization', 'information_extraction']
    df['prompt'] = df.apply(
        lambda row: f"{row['instruction']}\n{row['context']}" if pd.notna(row['context']) and str(row['context']).strip() != "" else str(row['instruction']), 
        axis=1
    )
    df['label'] = df['category'].apply(lambda x: 1 if x in complex_tasks else 0)
    return df

def generate_plots(y_test, y_pred, df_test):
    print("Generating academic plots...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Simple (8B)', 'Complex (70B)'], 
                yticklabels=['Simple (8B)', 'Complex (70B)'])
    plt.title('EcoPrompt Routing Accuracy (Confusion Matrix)')
    plt.ylabel('True Task Complexity')
    plt.xlabel('Predicted Routing Decision')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    
    # 2. Cost Optimization Bar Chart (Simulated for 500 queries to save API limits)
    # Llama 8B: ~$0.06/1M tokens. Llama 70B: ~$0.69/1M tokens.
    tokens_per_prompt = 250
    cost_70b = 0.69 / 1_000_000
    cost_8b = 0.06 / 1_000_000
    
    baseline_cost = len(df_test) * tokens_per_prompt * cost_70b
    
    eco_routed_cost = sum([tokens_per_prompt * cost_70b if pred == 1 else tokens_per_prompt * cost_8b for pred in y_pred])

    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Baseline\n(100% 70B)', 'EcoPrompt\n(Dynamic Routing)'], 
                   [baseline_cost, eco_routed_cost], 
                   color=['#e74c3c', '#2ecc71'])
    
    plt.title(f'OpEx Optimization for {len(df_test)} Queries')
    plt.ylabel('Total API Cost ($)')
    plt.bar_label(bars, fmt='$%.4f')
    plt.tight_layout()
    plt.savefig('cost_optimization.png', dpi=300)

if __name__ == "__main__":
    df = prepare_data()
    
    # Sample 3000 rows for a solid benchmark
    df = df.sample(n=3000, random_state=42) 
    
    X_train, X_test, y_train, y_test = train_test_split(df[['prompt']], df['label'], test_size=0.2, random_state=42)
    
    router = EcoRouter()
    
    # Train and save the model
    train_df = pd.DataFrame({'prompt': X_train['prompt'], 'label': y_train})
    router.train(train_df)
    router.save("router_model.pkl")
    
    # Evaluate performance
    print("\nEvaluating the Router on the test set...")
    y_pred = [router.predict(text) for text in X_test['prompt']]
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Simple (8B)', 'Complex (70B)']))
    
    # Generate charts
    test_df = pd.DataFrame({'prompt': X_test['prompt'], 'label': y_test})
    generate_plots(y_test, y_pred, test_df)
    
    print("\nDone. Model weights saved to 'router_model.pkl'.")
    print("Check your folder for 'confusion_matrix.png' and 'cost_optimization.png'.")