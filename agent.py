import os
import streamlit as st
from transformers import pipeline

# --- Streamlit page setup (must be first Streamlit command) ---
st.set_page_config(page_title="Qurat's Smart Agent", page_icon="🤖")

# --- Load the local AI model (DistilGPT-2) ---
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

# --- Define the AI Agent class ---
class SmartAgent:
    def __init__(self, name): 
        self.name = name 

    def decide(self, user_input):
        # Generate a response using DistilGPT-2
        result = generator(
            user_input,
            max_length=80,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        # Extract only the generated text after user's input
        text = result[0]['generated_text']
        reply = text[len(user_input):].strip()
        return reply or "I'm thinking..."

# --- Streamlit UI ---
st.title("Qurat's Agent")
st.write("Chat freely!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create agent instance
agent = SmartAgent("Qurat's Agent")

# Input box
user_input = st.chat_input("Say something...")

# Process input
if user_input:
    response = agent.decide(user_input)
    # Save conversation
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)
