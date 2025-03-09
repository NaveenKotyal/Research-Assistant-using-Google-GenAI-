from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

os.environ["GOOGLE_API_KEY"] = "AIzaSyDiwGYIM-SNR3hWTqJbe6Qomvo-iiHwSzA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

st.title("Reasearch Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

languages = st.selectbox( "Select the language of Explantation",["English","Hindi"])


template = load_prompt("template.json")

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
    'languages': languages

})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)




