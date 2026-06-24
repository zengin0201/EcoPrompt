import streamlit as st
import pandas as pd
from main import EcoRouter, call_heavy_model, call_light_model
from datasets import load_dataset

# Page Configuration
st.set_page_config(page_title="EcoPrompt Router", page_icon="🌿", layout="centered")

st.title("🌿 EcoPrompt Intelligent Router")
st.write("This AI system automatically routes your prompt to the most efficient model, saving cost and energy.")

# Cache the dataset and router training to prevent re-training on page reload
@st.cache_resource
def get_trained_router():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = dataset.to_pandas()
    complex_tasks = ['brainstorming', 'generation', 'summarization', 'information_extraction']
    df['prompt'] = df.apply(
        lambda row: str(row['instruction']) + "\n" + str(row['context']) if pd.notna(row['context']) and row['context'] != "" else str(row['instruction']), 
        axis=1
    )
    df['label'] = df['category'].apply(lambda x: 1 if x in complex_tasks else 0)
    train_data = df.sample(n=1000, random_state=42)
    
    router = EcoRouter()
    router.train(train_data)
    return router

with st.spinner("Initializing and training the EcoRouter model (takes 10-15s)..."):
    router = get_trained_router()
st.success("EcoRouter is online and ready!")

# User input text area
user_prompt = st.text_area("Enter your prompt for AI:", placeholder="Type something here...")

if st.button("Submit to AI"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Analyzing prompt complexity..."):
            route_class = router.predict(user_prompt)
        
        st.write("---")
        
        # Routing execution logic
        if route_class == 1:
            st.info("🧠 **EcoRouter Decision:** This is a **COMPLEX** task. Routing to **Llama-3.3-70B**.")
            with st.spinner("Waiting for Llama-3.3-70B..."):
                response_text, latency, cost = call_heavy_model(user_prompt)
            
            st.subheader("AI Response:")
            st.write(response_text)
            st.metric(label="Inference Cost (Dynamic)", value=f"${cost:.7f}", delta="Premium Processing", delta_color="inverse")
        else:
            st.success("⚡ **EcoRouter Decision:** This is a **SIMPLE** task. Routing to **Llama-3.1-8B**.")
            with st.spinner("Waiting for Llama-3.1-8B..."):
                response_text, latency, cost, heavy_equivalent_cost = call_light_model(user_prompt)
            
            st.subheader("AI Response:")
            st.write(response_text)
            
            # Calculate dynamic savings percentage
            if heavy_equivalent_cost > 0:
                savings = (1 - cost / heavy_equivalent_cost) * 100
            else:
                savings = 0
                
            st.metric(
                label="Inference Cost (Dynamic)", 
                value=f"${cost:.7f}", 
                delta=f"🌿 Saved {savings:.1f}% budget!", 
                delta_color="normal"
            )
            
        st.write(f"⏱️ **Response Time (Latency):** {latency:.2f} seconds")