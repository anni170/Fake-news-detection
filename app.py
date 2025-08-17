import streamlit as st
import joblib
import pandas as pd

from model import wordopt

LR = joblib.load('LR.pkl')
vectorization = joblib.load('tfidf_vectorizer.pkl')

def output_label(n):
    if n==0:
        return "It is a Fake news"
    elif n==1:
        return "It is a True news"

wordopt = {
    'Fake': 0,
    'True': 1
}

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = LR.predict(new_xv_test)

    return "\n\nLR Prediction:{} ".format(
        output_label(pred_lr[0])
    )


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below and check if it's real or fake.")

input_text = st.text_area("Enter News Article")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed = vectorization.transform([input_text])
        prediction = LR.predict(transformed)[0]
        label = "🚫 FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
        st.subheader(label)