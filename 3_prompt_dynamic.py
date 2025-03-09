from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "AIzaSyDiwGYIM-SNR3hWTqJbe6Qomvo-iiHwSzA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

st.title("Reasearch Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

languages = st.selectbox( "Select the language of Explantation",["English","Hindi"])


template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}
Language: {languages}
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input','length_input','languages'],
validate_template=True
)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
    'languages': languages

})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)




