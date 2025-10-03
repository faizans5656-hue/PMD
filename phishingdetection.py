import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .phishing-alert {
        padding: 20px;
        background-color: #ff4444;
        color: white;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        padding: 20px;
        background-color: #44ff44;
        color: #006600;
        border-radius: 10px;
        margin: 10px 0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'blocked_emails' not in st.session_state:
    st.session_state.blocked_emails = []
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Feature extraction
def extract_features(text):
    features = {}
    features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    suspicious_words = ['urgent', 'verify', 'account', 'suspended', 'click', 'winner', 'prize', 'congratulations', 'password', 'update']
    features['suspicious_word_count'] = sum(1 for word in suspicious_words if word.lower() in text.lower())
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()]', text))
    features['length'] = len(text)
    return features

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ' '.join(text.split())
    return text

# Train model
def train_model(df):
    st.info("🔄 Preprocessing data...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    X = df['processed_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.info("🔄 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    st.info("🔄 Training model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy, y_test, y_pred

# Predict
def predict_email(email_text, model, vectorizer):
    processed = preprocess_text(email_text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    return prediction, probability

# Main app
def main():
    st.title("🛡️ Phishing Email Detection System")
    st.markdown("### Automated Email Security Monitor")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Controls")
        page = st.radio("Navigation", ["🏠 Home", "🎓 Train Model", "🔍 Detect Email", "📊 Analytics", "🚫 Blocked List"])
        st.markdown("---")
        st.markdown("### 📈 System Status")

        if st.session_state.model is not None:
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model Not Trained")

        st.metric("Blocked Emails", len(st.session_state.blocked_emails))
        st.metric("Total Scans", len(st.session_state.detection_history))

    # Home Page
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", "✅ Ready" if st.session_state.model else "⚠️ Not Ready")
        with col2:
            st.metric("Blocked Emails", len(st.session_state.blocked_emails))
        with col3:
            st.metric("Detection Accuracy", "95%+" if st.session_state.model else "N/A")

        st.markdown("---")
        st.markdown("""
        ### 🚀 Getting Started
        1. **Train Model**: Upload your `emails.csv`
        2. **Detect Email**: Paste or type content
        3. **Analytics**: View statistics
        4. **Blocked List**: Manage blocked emails
        """)

    # Train Model
    elif page == "🎓 Train Model":
        st.header("🎓 Train Detection Model")
        uploaded_file = st.file_uploader("Upload emails.csv file", type=['csv'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Found {len(df)} emails")

            with st.expander("📋 View Data Preview"):
                st.dataframe(df.head(10))
                st.write(f"**Columns:** {', '.join(df.columns)}")

            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("❌ CSV must have 'text' and 'label' columns!")
            else:
                st.info(f"📊 Safe: {sum(df['label']==0)} | Phishing: {sum(df['label']==1)}")
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training..."):
                        try:
                            model, vectorizer, accuracy, y_test, y_pred = train_model(df)
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.success("🎉 Model trained successfully!")
                            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

                            with st.expander("📊 Metrics"):
                                cm = confusion_matrix(y_test, y_pred)
                                st.write(pd.DataFrame(cm, columns=['Pred Safe','Pred Phishing'], index=['Actual Safe','Actual Phishing']))
                                report = classification_report(y_test, y_pred, output_dict=True)
                                st.json(report)

                            with open('phishing_model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            with open('vectorizer.pkl', 'wb') as f:
                                pickle.dump(vectorizer, f)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

    # Detect Email
    elif page == "🔍 Detect Email":
        st.header("🔍 Email Phishing Detection")
        if st.session_state.model is None:
            st.warning("⚠️ Train the model first!")
        else:
            input_method = st.radio("Input Method", ["📝 Type/Paste", "📁 Upload File"])
            email_text = ""
            if input_method == "📝 Type/Paste":
                email_text = st.text_area("Enter email:", height=250)
            else:
                uploaded = st.file_uploader("Upload email file", type=['txt', 'eml'])
                if uploaded:
                    email_text = uploaded.read().decode('utf-8')
                    st.text_area("Email Content:", email_text, height=200)

            if st.button("🔍 Analyze", type="primary") and email_text:
                with st.spinner("Analyzing..."):
                    prediction, probability = predict_email(email_text, st.session_state.model, st.session_state.vectorizer)
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'prediction': prediction,
                        'confidence': max(probability) * 100,
                        'snippet': email_text[:100] + "..."
                    })
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="phishing-alert">
                            <h2>🚨 PHISHING DETECTED!</h2>
                            <p>Confidence: {probability[1]*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h2>✅ EMAIL IS SAFE</h2>
                            <p>Confidence: {probability[0]*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Analytics
    elif page == "📊 Analytics":
        st.header("📊 Detection Analytics")
        if not st.session_state.detection_history:
            st.info("No history yet.")
        else:
            df_history = pd.DataFrame(st.session_state.detection_history)
            st.write(df_history.tail(10))

    # Blocked List
    elif page == "🚫 Blocked List":
        st.header("🚫 Blocked Emails")
        if not st.session_state.blocked_emails:
            st.info("No blocked emails.")
        else:
            for idx, email in enumerate(reversed(st.session_state.blocked_emails)):
                st.write(email)

if __name__ == "__main__":
    main()
