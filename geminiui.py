import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import gdown

def generate_reply(ticket, department):
    model_gemini = genai.GenerativeModel("gemini-flash-latest")

    prompt = f"""
    You are a professional customer support agent.

    Customer ticket: "{ticket}"
    Department: {department}

    Write a polite, helpful, and professional reply.
    Give me the response in 100 to 200 words 
    """

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ Unable to generate response. Please try again."
        
@st.cache_data
def download_model():
    file_id = "1BJRvJlehCErRzS-nZvA814UaIxeyW9bl"

    if not os.path.exists("model1.pkl"):
        gdown.download(
            id=file_id,
            output="model1.pkl",
            quiet=False
        )

download_model()  

with open("model1.pkl","rb") as file:
    model, vectorizer=pickle.load(file)

API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)


# This for identifying the available free models in Gemini

# models = genai.list_models()

# for m in models:
#     print(m.name, "->", m.supported_generation_methods)


st.sidebar.title("Navigation")
page=st.sidebar.radio('Go To',['Project Introduction','Chatbot','Creator Info'])


if page=='Project Introduction':
    st.title("🤖 AI-Powered Customer Support Automation")
    st.subheader("📩 Intelligent Ticket Classification & Automated Response System")

    st.write("""
This application is designed to automate the process of handling customer support tickets
using **Natural Language Processing (NLP)** and **Generative AI**. It helps organizations
efficiently manage large volumes of customer queries by automatically classifying tickets
and generating instant responses.

The system analyzes the text of a customer support ticket and predicts the appropriate
department (queue) such as **billing, technical support, or account management**.
After classification, it generates a **polite acknowledgment response** using Generative AI,
ensuring quick and professional communication with customers.

The project integrates machine learning, deep learning, and AI-driven text generation
to build a scalable and intelligent support system.

**Key Capabilities:**
- Automatically classify customer support tickets into the correct **support queue**.
- Generate **instant polite responses** using Generative AI (Gemini API).
- Improve **ticket routing accuracy** and reduce manual effort.
- Detect important signals like **keywords, urgency, and sentiment**.
- Provide a **scalable solution** for handling large volumes of support requests.

**Technologies Used:**
- Python
- Natural Language Processing (NLP)
- Machine Learning (Logistic Regression, Naïve Bayes, SVM)
- Deep Learning (RNN / LSTM)
- Transformers (BERT / RoBERTa - optional)
- Generative AI (Gemini API)
- Data Processing (Pandas, NumPy)
- Streamlit for the web interface

**Dataset Used:** `Tobi-Bueck/customer-support-tickets (Hugging Face)`

This tool helps businesses enhance customer experience by providing **faster responses,
accurate ticket routing, and reduced operational costs**, making customer support systems
more efficient and intelligent.
""")

elif page == 'Chatbot':

    st.title("🤖 AI Customer Support Automation")

    input=st.text_area("Enter your text here")

    if st.button("Submit"):
        
         if input.strip() == "":
            st.warning("Please enter a reply")
         else:
            
            vector_input = vectorizer.transform([input])
            prediction = model.predict(vector_input)[0]

            reply = generate_reply(input, prediction)

            st.subheader("📌 Predicted Category")
            st.success(prediction)

            st.subheader("💬 AI Response")
            st.write(reply)

elif page=='Creator Info':
    st.title("👩‍💻 Creator of this Project")

    st.write("""
**Developed by:** Vasantha Leela MB  

🎓 **Education:** B.E. Computer Science and Engineering  
🏫 **Institution:** Karpagam Academy of Higher Education  

💡 **Skills:**
- Python
- Machine Learning & Deep Learning
- Natural Language Processing (NLP)
- Generative AI (Gemini API)
- Data Analysis (Pandas, NumPy)
- Streamlit & Web Development  

🚀 **Project Focus:**
- Customer Support Automation  
- Ticket Classification using ML & LSTM  
- AI-based Response Generation  

📌 Passionate about building intelligent, scalable, and user-friendly AI applications.
""")
