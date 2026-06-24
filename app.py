import os
import time
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, wait_random_exponential, stop_after_attempt
from router import EcoRouter

# Initialize environment and API client
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configure the UI layout
st.set_page_config(page_title="EcoPrompt Research", page_icon="🌿", layout="centered")

st.title("🌿 EcoPrompt Intelligent Router")
st.markdown("""
**An Energy-Efficient LLM Routing Framework.**  
This system analyzes semantic complexity via `MiniLM Embeddings + Random Forest` in <1ms and dynamically routes queries to optimize compute, saving ~90% on API operational costs.
""")

# Load the pre-trained model once and cache it
@st.cache_resource
def load_production_router():
    router = EcoRouter()
    try:
        router.load("router_model.pkl")
        return router
    except FileNotFoundError:
        st.error("Model weights not found. Please run `python train_benchmark.py` in your terminal first.")
        st.stop()

router = load_production_router()

# API call function with retry logic to handle rate limits
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def call_model(prompt, is_complex):
    model_name = "llama-3.3-70b-versatile" if is_complex else "llama-3.1-8b-instant"
    input_price = 0.59 if is_complex else 0.05
    output_price = 0.79 if is_complex else 0.08
    
    start_time = time.time()
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    latency = time.time() - start_time
    
    p_tokens = response.usage.prompt_tokens
    c_tokens = response.usage.completion_tokens
    actual_cost = (p_tokens * (input_price / 1_000_000)) + (c_tokens * (output_price / 1_000_000))
    
    # Calculate what it would have cost if we routed everything to the heavy model
    baseline_cost = (p_tokens * (0.59 / 1_000_000)) + (c_tokens * (0.79 / 1_000_000))
    
    return response.choices[0].message.content, latency, actual_cost, baseline_cost, model_name


user_prompt = st.text_area("Enter your prompt:", placeholder="E.g., What is the capital of France? OR Write a Python script for a convolutional neural network...", height=150)

if st.button("Submit to AI Framework"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt to proceed.")
    else:
        with st.spinner("Analyzing syntactic and semantic complexity..."):
            route_class = router.predict(user_prompt)
            
        st.divider()
        
        is_complex = (route_class == 1)
        tier_label = "COMPLEX" if is_complex else "SIMPLE"
        color = "red" if is_complex else "green"
        
        st.markdown(f"**🧠 EcoRouter Decision:** Detected as **:{color}[{tier_label} TASK]**")
        
        try:
            with st.spinner("Streaming from optimal LLM..."):
                resp_text, latency, actual_cost, baseline_cost, model_used = call_model(user_prompt, is_complex)
            
            st.markdown(f"**Model selected:** `{model_used}`")
            st.info(resp_text)
            
            # Display metrics dashboard
            st.subheader("📊 Inference Analytics")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Latency", f"{latency:.2f} s")
            
            if is_complex:
                col2.metric("OpEx Cost", f"${actual_cost:.6f}", "Premium Tier", delta_color="inverse")
                col3.metric("Cost Savings", "0%", "Required heavy compute")
            else:
                savings_pct = (1 - actual_cost / baseline_cost) * 100
                col2.metric("OpEx Cost", f"${actual_cost:.6f}", f"-${baseline_cost - actual_cost:.6f}")
                col3.metric("Green Compute Savings", f"{savings_pct:.1f}%", "Eco-friendly routing")
                
        except Exception as e:
            st.error(f"API Error: Rate limit reached or connection failed. Details: {e}")
