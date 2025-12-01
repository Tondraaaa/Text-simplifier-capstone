import streamlit as st
from transformers import pipeline
import textstat

# ----------------------------
# Load smaller, cloud-friendly models
# ----------------------------
@st.cache_resource
def load_models():
    # DistilBART is a smaller version of BART CNN
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    # FLAN-T5 base is much smaller than flan-t5-large
    rewriter = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    return summarizer, rewriter


# ----------------------------
# Streamlit UI setup
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
# User input
# ----------------------------
st.markdown("<h4>Enter Academic Text</h4>", unsafe_allow_html=True)
text_input = st.text_area(
    "Enter academic text here",
    height=180,
    label_visibility="visible"
)


# ----------------------------
# Two-step simplification function
# ----------------------------
def simplify_text(text: str):
    text = text.strip()
    if not text:
        return "Please enter some text to simplify.", 0.0, 0.0, ""

    # Original readability
    original_score = textstat.flesch_reading_ease(text)

    # Step 1 – summarize
    summary_result = st.session_state["summarizer"](
        text,
        max_length=120,
        min_length=30,
        do_sample=False,
        truncation=True,
    )
    summary = summary_result[0]["summary_text"]

    # Step 2 – rewrite summary in plain English
    instruction = (
        "Rewrite the following summary in clear, plain English for a general audience. "
        "Use short sentences, simple vocabulary, and make it easy to understand."
    )
    prompt = f"{instruction}\n\n{summary}"

    rewrite_result = st.session_state["rewriter"](
        prompt,
        max_length=256,
        do_sample=False,
    )
    simplified_output = rewrite_result[0]["generated_text"].strip()

    simplified_score = textstat.flesch_reading_ease(simplified_output)

    return simplified_output, original_score, simplified_score, summary


# ----------------------------
# Run simplification
# ----------------------------
if st.button("Simplify"):
    # Lazy-load models only when needed
    if "summarizer" not in st.session_state or "rewriter" not in st.session_state:
        with st.spinner(
            "Loading language models. This may take a moment the first time..."
        ):
            summarizer, rewriter = load_models()
            st.session_state["summarizer"] = summarizer
            st.session_state["rewriter"] = rewriter

    simplified_text, original_score, simplified_score, summary = simplify_text(
        text_input
    )

    st.markdown("<h4>Simplified Output</h4>", unsafe_allow_html=True)
    st.write(simplified_text)

    if summary:
        with st.expander("View Intermediate Summary (Step 1)"):
            st.write(summary)

    st.markdown(f"**Original Flesch Reading Ease:** {original_score:.2f}")
    st.markdown(f"**Simplified Flesch Reading Ease:** {simplified_score:.2f}")

    # Readability feedback
    if simplified_score > original_score + 10:
        st.success("✅ Readability improved significantly!")
    elif simplified_score > original_score:
        st.info(
            "ℹ️ Readability improved slightly. Try a longer or more complex input for stronger results."
        )
    else:
        st.warning(
            "⚠️ The simplified version may still be complex. Try another phrasing or shorter sections."
        )

    # Download button
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
        mime="text/plain",
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Developed by Shatondra Asor-Sallaah, Tamika Mosley, and Aprylee Brown | "
    "LSIS 5460 AI Capstone | Accessibility Prototype v3.0"
)
