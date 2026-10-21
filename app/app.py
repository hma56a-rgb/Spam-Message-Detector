import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="Spam Message Detector", page_icon="🛡️")


@st.cache_resource
def load_artifacts():
    model = joblib.load(APP_DIR / "spam_model.pkl")
    config = json.loads((APP_DIR / "config.json").read_text())
    return model, config


try:
    model, config = load_artifacts()
except Exception as exc:
    st.error(
        "Could not load the model. Re-run notebooks/02_model_training.ipynb and "
        "notebooks/03_model_interpretation.ipynb to regenerate it for your environment."
    )
    st.exception(exc)
    st.stop()

union = model.named_steps["features"]
coefs = model.named_steps["clf"].coef_[0]
names = union.get_feature_names_out()
feature_type = np.array([n.split("__")[0] for n in names])
feature_term = np.array([n.split("__", 1)[1] for n in names])

EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim.",
    "Free offer": "Free entry in a weekly competition to win FA Cup tickets! Text FA to 87121 to receive entry questions.",
    "Casual chat": "Hey, are we still meeting at 7 tonight? I'll bring the notes from class.",
    "Order update": "Your order has been shipped and should arrive on Thursday. Thanks for shopping with us!",
    "Known miss: shortcode spam": "ringtoneking 84484",
    "Known false alarm: short ham": "Glad to see your reply.",
}


def spam_probability(texts):
    return model.predict_proba(texts)[:, 1]


def explain(message, top_n=6):
    row = union.transform([message]).tocoo()
    if row.nnz == 0:
        return pd.DataFrame(columns=["type", "term", "contribution", "pushes toward"])
    out = pd.DataFrame({
        "type": feature_type[row.col],
        "term": feature_term[row.col],
        "contribution": row.data * coefs[row.col],
    })
    words = out[out["type"] == "word"]
    top = words.loc[words["contribution"].abs().sort_values(ascending=False).index].head(top_n)
    chars = pd.DataFrame([{
        "type": "char",
        "term": "(all character patterns)",
        "contribution": out.loc[out["type"] == "char", "contribution"].sum(),
    }])
    table = pd.concat([top, chars], ignore_index=True)
    table["contribution"] = table["contribution"].round(3)
    table["pushes toward"] = np.where(table["contribution"] > 0, "spam", "ham")
    return table


with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider("Flag as spam above", 0.05, 0.99, float(config["threshold"]), 0.01)
    st.caption(f"Tuned value: {config['threshold']:.2f} (>= 99% precision on training folds).")
    st.subheader("Held-out test set at the tuned value")
    st.write(f"Precision: {config['test_precision']:.1%}")
    st.write(f"Recall: {config['test_recall']:.1%}")
    st.write(f"F1: {config['test_f1']:.3f}")
    st.caption("English, SMS-style text only.")

st.title("🛡️ Spam Message Detector")
st.caption("Word + character TF-IDF with Logistic Regression, trained on the SMS Spam Collection.")

single_tab, batch_tab = st.tabs(["Single message", "Batch"])

with single_tab:
    example = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
    message = st.text_area("Message", value=EXAMPLES.get(example, ""), height=120)

    if st.button("Check message", type="primary"):
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            prob = float(spam_probability([message])[0])
            is_spam = prob >= threshold
            col1, col2 = st.columns(2)
            col1.metric("Spam probability", f"{prob:.1%}")
            col2.metric("Verdict", "Spam" if is_spam else "Not spam")
            st.progress(min(int(prob * 100), 100))
            if is_spam:
                st.error("🚫 Likely spam.")
            else:
                st.success("✅ Looks legitimate.")
            st.subheader("What drove the score")
            st.caption("Top words, plus the combined effect of character patterns (digits, symbols, writing style).")
            st.dataframe(explain(message), hide_index=True)

with batch_tab:
    st.write("Paste one message per line to triage a whole inbox.")
    batch_text = st.text_area("Messages", height=200, key="batch_input")

    if st.button("Classify all"):
        lines = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Paste at least one message.")
        else:
            probs = spam_probability(lines)
            st.session_state["batch_result"] = pd.DataFrame({
                "message": lines,
                "spam_probability": probs.round(3),
                "verdict": np.where(probs >= threshold, "spam", "ok"),
            })

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        st.write(f"{(result['verdict'] == 'spam').sum()} of {len(result)} flagged as spam")
        st.dataframe(result, hide_index=True)
        st.download_button("Download results (CSV)", result.to_csv(index=False), "triage_results.csv", "text/csv")
