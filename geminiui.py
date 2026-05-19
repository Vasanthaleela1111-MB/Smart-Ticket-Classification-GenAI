import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import google.generativeai as genai
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    

@st.cache_resource
def load_model():

    file_id = "1ZuhG7PZ7ie_m39tc1Z0OD78PNyQyJOjv"

    if not os.path.exists("model1.pkl"):

        gdown.download(
            id=file_id,
            output="model1.pkl",
            quiet=False
        )

    model, vectorizer = joblib.load("model1.pkl")

    return model, vectorizer

model, vectorizer = load_model()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# This for identifying the available free models in Gemini

# models = genai.list_models()

# for m in models:
#     print(m.name, "->", m.supported_generation_methods)


with st.sidebar:

    st.title("🤖 Support AI")

    st.caption("Smart Customer Ticket Automation")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📘 Project Introduction",
            "🎫 Ticket Classification",
            "👩‍💻 Creator Info"
        ]
    )

    st.divider()

    st.info(
        "Built using NLP, Machine Learning, and Gemini AI."
    )


if page == '📘 Project Introduction':

    st.title("🤖 AI-Powered Customer Support Automation")

    st.caption(
        "Intelligent ticket classification and AI-generated customer support responses."
    )

    st.write("")

    # Hero Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("⚡ Instant Ticket Analysis")

    with col2:
        st.info("🧠 NLP + Gemini AI")

    with col3:
        st.info("🎯 Automated Routing")

    st.write("")

    # About Project
    with st.container(border=True):

        st.subheader("📘 About the Project")

        st.write("""
        This application automates customer support ticket handling using
        **Natural Language Processing (NLP)** and **Generative AI**.

        The system analyzes customer queries, predicts the appropriate
        support department, and generates intelligent AI-powered responses
        to improve customer support efficiency.

        It helps organizations reduce manual effort, improve response speed,
        and automate ticket routing effectively.
        """)

    st.write("")

    # Features
    with st.container(border=True):

        st.subheader("✨ Key Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ✅ Smart Ticket Classification  
            ✅ AI Generated Responses  
            ✅ Real-Time Prediction  
            ✅ NLP-Based Processing  
            """)

        with col2:
            st.markdown("""
            ✅ Automated Ticket Routing  
            ✅ Reduced Manual Effort  
            ✅ Sentiment Understanding  
            ✅ Scalable Support Solution  
            """)

    st.write("")

    # Tabs Section
    tab1, tab2, tab3 = st.tabs(
        ["🧠 Technologies", "📊 ML Models", "💬 AI Features"]
    )

    with tab1:

        st.markdown("""
        ### Technologies Used

        - Python
        - Streamlit
        - Pandas & NumPy
        - NLP Techniques
        - Generative AI (Gemini API)
        """)

    with tab2:

        st.markdown("""
        ### Machine Learning Models

        - Logistic Regression
        - Naïve Bayes
        - Support Vector Machine (SVM)
        - RNN / LSTM
        - Transformers (Optional)
        """)

    with tab3:

        st.markdown("""
        ### AI Capabilities

        - Intelligent Response Generation
        - Context-Aware Conversations
        - Automated Customer Assistance
        - Professional AI Replies
        """)

    st.write("")

    # Dataset
    with st.expander("📂 Dataset Information"):

        st.write("""
        **Dataset Used:**  
        `Tobi-Bueck/customer-support-tickets (Hugging Face)`

        The dataset contains customer support queries categorized into
        multiple support departments for training and evaluation.
        """)

    st.write("")

    # Benefits
    with st.container(border=True):

        st.subheader("🚀 Business Benefits")

        st.success("""
        ✔ Faster customer response times  
        ✔ Improved ticket routing accuracy  
        ✔ Reduced operational workload  
        ✔ Better customer satisfaction  
        ✔ Scalable AI-powered support system
        """)

elif page == '🎫 Ticket Classification':

    st.markdown("# 🎫 Smart Ticket Classification")

    st.caption(
        "Enter a customer issue and get AI-powered classification and response."
    )

    st.write("")

    with st.container(border=True):

        user_input = st.text_area(
            "Customer Query",
            placeholder="Example: My order was damaged and I need a refund...",
            height=180
        )

        col1, col2, col3 = st.columns([1,1,4])

        with col1:
            submit = st.button("Analyze")

    if submit:

        if user_input.strip() == "":

            st.warning("⚠️ Please enter a customer query.")

        else:

            with st.spinner("Analyzing customer ticket..."):

                vector_input = vectorizer.transform([user_input])

                prediction = model.predict(vector_input)[0]

                reply = generate_reply(user_input, prediction)

            st.write("")

            with st.container(border=True):

                st.subheader("📌 Predicted Category")

                st.success(prediction)

            st.write("")

            with st.container(border=True):

                st.subheader("💬 Support Assistant Response")

                st.write(reply)

            st.write("")

            # st.toast("Analysis Completed Successfully ✅")

elif page == '👩‍💻 Creator Info':

    st.markdown("# 👩‍💻 Creator")

    with st.container(border=True):

        st.markdown("""
        ### Vasantha Leela MB

        🎓 B.E Computer Science and Engineering  
        🏫 Karpagam Academy of Higher Education  

        ### 🚀 Skills

        - Python
        - Machine Learning
        - Deep Learning
        - NLP
        - Generative AI
        - Streamlit Development

        ### 💡 Project Focus

        - AI Customer Support
        - Ticket Classification
        - Intelligent Response Systems
        """)

    st.write("")

    st.success(
        "Passionate about building intelligent and user-friendly AI applications."
    )
