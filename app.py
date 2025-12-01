import streamlit as st
from transformers import pipeline
import textstat

# ----------------------------
# Load both models
# ----------------------------
@st.cache_resource
def load_models():
    # Smaller, faster models for Streamlit Cloud
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    rewriter = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    return summarizer, rewriter


summarizer, rewriter = load_models()

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="AI Text Simplifier Prototype", layout="centered")

st.title("AI Text Simplifier Prototype")

st.write(
    "This working prototype rewrites complex academic text into plain English using a two-step process:"
)

st.markdown(
    """
    <h4>How it Works</h4>
    1. <strong>Summarize</strong> dense academic text into a shorter description.<br>
    2. <strong>Rewrite</strong> the summary in clear, plain English for a general audience.<br><br>
    The prototype also reports <strong>Flesch Reading Ease</strong> scores before and after simplification to show changes in readability.
    """,
    unsafe_allow_html=True
)

# ----------------------------
# User Input
# ----------------------------
st.markdown("<h4>Enter Academic Text</h4>", unsafe_allow_html=True)
text_input = st.text_area("", height=180)

# ----------------------------
# Two-Step Simplification Function
# ----------------------------
def simplify_text(text):
    if not text.strip():
        return "Please enter some text to simplify.", 0, 0, ""

    original_score = textstat.flesch_reading_ease(text)

    # Step 1 – Summarize
    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    # Step 2 – Rewrite summary into plain English
    instruction = (
        "Rewrite the following summary in clear, plain English for a general audience. "
        "Use short sentences, simple vocabulary, and make it easy to understand."
    )
    prompt = f"{instruction}\n\n{summary}"
    simplified_output = rewriter(
        prompt,
        max_length=256,
        do_sample=False
    )[0]["generated_text"].strip()

    simplified_score = textstat.flesch_reading_ease(simplified_output)

    return simplified_output, original_score, simplified_score, summary

# ----------------------------
# Run Simplification
# ----------------------------
if st.button("Simplify"):
    simplified_text, original_score, simplified_score, summary = simplify_text(text_input)

    st.markdown("<h4>Simplified Output</h4>", unsafe_allow_html=True)
    st.write(simplified_text)

    with st.expander("View Intermediate Summary (Step 1)"):
        st.write(summary)

    st.markdown(f"**Original Flesch Reading Ease:** {original_score:.2f}")
    st.markdown(f"**Simplified Flesch Reading Ease:** {simplified_score:.2f}")

    # ----------------------------
    # Readability Feedback Boxes (with icons)
    # ----------------------------
    if simplified_score > original_score + 10:
        st.success("✅ Readability improved significantly!")
    elif simplified_score > original_score:
        st.info("ℹ️ Readability improved slightly. Try a longer or more complex input for stronger results.")
    else:
        st.warning("⚠️ The simplified version may still be complex. Try another phrasing or shorter sections.")

    # ----------------------------
    # Download Results Button
    # ----------------------------
    results_text = (
        f"Simplified Output:\n{simplified_text}\n\n"
        f"Intermediate Summary:\n{summary}\n\n"
        f"Original Flesch Reading Ease: {original_score:.2f}\n"
        f"Simplified Flesch Reading Ease: {simplified_score:.2f}"
    )

    st.download_button(
        label="Download results as .txt",
        data=results_text,
        file_name="simplified_results.txt",
        mime="text/plain"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Developed by Shatondra Asor-Sallaah, Tamika Mosley, and Aprylee Brown | "
    "LSIS 5460 AI Capstone | Accessibility Prototype v3.0"
)

