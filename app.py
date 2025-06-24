import streamlit as st
import pandas as pd
import joblib # For loading pre-trained model and vectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load pre-trained components
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset for EDA
@st.cache_data
def load_data():
    return pd.read_csv("train.txt", sep=";", names=["text", "emotion"])

df = load_data()

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocessing
df['clean_text'] = df['text'].apply(clean_text)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📘 Overview", "📊 Visualization & Exploration", "🤖 Predict Emotion"])

# -----------------------------------------------
# 📘 Overview Page
if page == "📘 Overview":
    st.title("Text Emotion Classification Web App")

    st.markdown("""
    ### Welcome to the Emotion Detection App!

    This web application allows you to analyze and classify human emotions based on text input. 
    It was built using:

    - **Python** 🐍
    - **Streamlit** for the web interface
    - **scikit-learn** for machine learning
    - **TF-IDF** vectorization for feature extraction
    - **Logistic Regression** as the classifier

    ### Features:
    - Full **EDA and visualization** of training and test data
    - Real-time **emotion prediction** from custom sentences
    - Visual performance tracking using **confusion matrices**

    Navigate through the sidebar to explore insights and interact with the app!
    """)

# -----------------------------------------------
# 📊 Visualization & Exploration Page
elif page == "📊 Visualization & Exploration":
    st.title("📊 Visualization & Exploration")

    st.subheader("1. Sample Data")
    st.write(df.sample(10))

    st.subheader("2. Emotion Class Distribution")
    fig1, ax1 = plt.subplots()
    df['emotion'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)
    st.markdown("Most frequent emotions in the training data are **joy** and **sadness**, while **surprise** appears the least.")

    st.subheader("3. Word Count Statistics")
    st.write(df['word_count'].describe())
    st.markdown("On average, each sentence contains about **19 words**, with some going as high as 66. This helps understand sentence length trends.")

    st.subheader("4. Histogram of Word Counts")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['word_count'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)
    st.markdown("The distribution is **right-skewed**, meaning most sentences are short, but a few are long.")

    st.subheader("5. Box Plot of Word Counts by Emotion")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='emotion', y='word_count', data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.markdown("Emotions like **joy** and **love** tend to have longer text inputs, while **surprise** is often short.")

    try:
        test_df = pd.read_csv("test.txt", sep=";", names=["text", "emotion"])
        test_df['clean_text'] = test_df['text'].apply(clean_text)
        test_df['word_count'] = test_df['clean_text'].apply(lambda x: len(x.split()))

        st.subheader("6. Final Test Set Emotion Distribution")
        fig4, ax4 = plt.subplots()
        test_df['emotion'].value_counts().plot(kind='bar', ax=ax4, color='orange')
        st.pyplot(fig4)
        st.markdown("The emotion distribution in the **final test set** is also dominated by **joy** and **sadness**.")

    except:
        st.info("Final test set not found or failed to load.")

    try:
        from sklearn.model_selection import train_test_split

        df['label'] = le.transform(df['emotion'])
        X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

        X_train_vec = tfidf.transform(X_train)
        X_test_vec = tfidf.transform(X_test)
        y_pred = model.predict(X_test_vec)

        st.subheader("7. Confusion Matrix on Internal Test Set")
        cm = confusion_matrix(y_test, y_pred)
        fig7, ax7 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax7)
        st.pyplot(fig7)
        st.markdown("This matrix shows how well the model performs on **20% split from training data**. Most predictions align closely with actual labels.")

        if 'test_df' in locals():
            test_vec = tfidf.transform(test_df['clean_text'])
            test_labels = le.transform(test_df['emotion'])
            final_preds = model.predict(test_vec)

            st.subheader("8. Confusion Matrix on Final Test Set")
            cm_final = confusion_matrix(test_labels, final_preds)
            fig8, ax8 = plt.subplots()
            sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax8)
            st.pyplot(fig8)
            st.markdown("Final test set matrix shows **real-world generalization** performance. Slightly lower performance may occur if class balance differs.")

    except Exception as e:
        st.error(f"Error generating confusion matrices: {e}")

# -----------------------------------------------
# 🤖 Predict Emotion Page
elif page == "🤖 Predict Emotion":
    st.title("🤖 Predict the Emotion")

    st.write("Type a sentence below and click Predict:")

    user_input = st.text_area("Enter your sentence here")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a valid sentence.")
        else:
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            emotion = le.inverse_transform([prediction])[0]

            st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")