import streamlit as st
from transformers import pipeline

def main():
    # Set Streamlit app title
    st.title("Text Summarization App")

    # Create a text area for user input
    user_input = st.text_area("Enter text to summarize:")

    # Check if user input is not empty
    if user_input:
        # Create a button to trigger text summarization
        if st.button("Summarize"):
            # Perform text summarization using the pipeline
            summarizer = pipeline("summarization", model="Falconsai/text_summarization")
            summary = summarizer(user_input, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

            # Display the summarized text
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])

if __name__ == "__main__":
    main()
