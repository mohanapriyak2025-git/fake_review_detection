import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('fake_review_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")

st.title("🛡️Fake Review Detection System")
st.write("Detect Fake or Genuine reviews using Machine Learning")

# ---------------- MULTI-LINE INPUT ----------------
st.subheader("✍️ Paste Reviews (One per line)")

reviews_text = st.text_area("Enter  reviews")

if st.button("Predict Pasted Reviews"):

    if reviews_text.strip() == "":
        st.warning("Please enter reviews")
    else:
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]

        predictions = model.predict(tfidf.transform(reviews))

        results = []
        for r, p in zip(reviews, predictions):
            results.append({
                "Review": r,
                "Prediction": "Fake Review" if p == 0 else "Genuine Review"
            })

        st.dataframe(results, use_container_width=True)


st.divider()

# ---------------- CSV UPLOAD ----------------
st.subheader("📂 Upload CSV File")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()

    # detect review column
    review_col = None
    for col in df.columns:
        if col in ['review', 'text', 'content', 'comment', 'review_text', 'sentence']:
            review_col = col
            break

    if review_col is None:
        review_col = df.columns[0]

    reviews = df[review_col].astype(str)

    predictions = model.predict(tfidf.transform(reviews))

    df['prediction'] = predictions

    df['result'] = df['prediction'].apply(
        lambda x: "Fake Review" if x == 0 else "Genuine Review"
    )

    st.dataframe(df, use_container_width=True)


# ---------------- BACKGROUND ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #ffd6e0, #ff4d88);
    }
    </style>
    """,
    unsafe_allow_html=True
)