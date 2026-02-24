import re
import streamlit as st

def highlight_text(text, keywords, summary_sentences):
    # Highlight summary sentences first (blue)
    for sent in summary_sentences:
        pattern = re.escape(sent.strip())
        text = re.sub(
            pattern,
            f"<mark style='background-color:#90caf9; color:black'>{sent}</mark>",
            text,
            flags=re.IGNORECASE
        )

    # Highlight keywords (yellow)
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(
            pattern,
            f"<span style='background-color:#ffd54f; color:black'>{word}</span>",
            text,
            flags=re.IGNORECASE
        )

    return text

@st.dialog("Document View", width="large")
def show_document_modal(doc_name, raw_text, cleaned_text, keywords, summary_sentences):
    st.markdown(
        f"### {doc_name}\n\n"
        "**Legend:** \n"
        "<span style='background-color:#90caf9; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Summary Sentence</span> \n"
        "&nbsp;&nbsp;\n"
        "<span style='background-color:#ffd54f; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Keyword</span>\n"
        "<hr style='margin-top: 10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True
    )
    
    tab1, tab2 = st.tabs(["Cleaned Text (Highlighted)", "Raw Text"])
    
    with tab1:
        highlighted = highlight_text(cleaned_text, keywords, summary_sentences)
        st.markdown(
            f"""
            <div style="
                max-height: 500px;
                overflow-y: auto;
                padding: 1rem;
                border-radius: 8px;
                background-color: #111;
                line-height: 1.6;
                font-size: 16px;
                white-space: pre-wrap;
            ">
            {highlighted}
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with tab2:
        st.markdown(
            f"""
            <div style="
                max-height: 500px;
                overflow-y: auto;
                padding: 1rem;
                border-radius: 8px;
                background-color: #111;
                line-height: 1.6;
                font-size: 16px;
                white-space: pre-wrap;
            ">
            {raw_text}
            </div>
            """,
            unsafe_allow_html=True
        )
