import streamlit as st
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")

# Task Keywords
TASK_KEYWORDS = {
    "buy", "schedule", "write", "send", "fix", "clean", "attend", "call", "finalize", "complete", "organize",
    "submit", "review", "prepare", "update", "arrange", "resolve", "plan", "book", "report", "approve", "notify",
    "research", "deliver", "respond", "implement", "build", "create", "design", "analyze", "manage", "coordinate",
    "deploy", "evaluate", "investigate", "present", "edit", "draft", "install", "launch", "assign", "execute",
    "document", "conduct", "schedule", "order", "process"
}

# Time Keywords
TIME_KEYWORDS = {
    "today", "tomorrow", "morning", "afternoon", "evening", "night", "midnight", "noon", "weekend",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
    "tonight", "AM", "PM", "next", "this", "coming", "before", "by", "after"
}

# Function to extract tasks
def extract_tasks(text):
    processed_sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
    extracted_tasks = []
    doc = nlp(text)

    for sentence in processed_sentences:
        task = []
        who = "Unknown"
        when = "No deadline mentioned"
        task_found = False

        # Extract subject using SpaCy NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":  
                who = ent.text  

        for i, (word, pos) in enumerate(sentence):
            word_lower = word.lower()

            # Identify subject using POS tagging
            if who == "Unknown" and pos in {"NNP", "NN"}:
                who = word.capitalize()

            # Identify actionable verbs
            if word_lower in TASK_KEYWORDS:
                if task:
                    extracted_tasks.append({"task": " ".join(task), "who": who, "when": when})
                task = [word]
                task_found = True

            # Capture valid nouns after verbs
            elif task_found and (pos.startswith('NN') or pos.startswith('VB')) and word_lower not in TIME_KEYWORDS:
                task.append(word)

            # Identify time expressions
            if word in TIME_KEYWORDS or pos == "CD":
                when = sentence[i][0]
                if i + 1 < len(sentence) and sentence[i+1][0] in {"AM", "PM"}:
                    when += " " + sentence[i+1][0]

        if task:
            extracted_tasks.append({"task": " ".join(task), "who": who, "when": when})

    return extracted_tasks

# Streamlit UI
st.title("📌 Task Extraction App")
st.markdown("Enter a paragraph below, and the app will extract structured tasks.")

# User Input
user_input = st.text_area("Enter a paragraph:")

if st.button("Extract Tasks"):
    if user_input:
        extracted_tasks = extract_tasks(user_input)
        if extracted_tasks:
            st.subheader("✅ Extracted Tasks:")
            for task in extracted_tasks:
                st.write(f"📌 **Task:** {task['task']}")
                st.write(f"👤 **Who:** {task['who']}")
                st.write(f"🕒 **When:** {task['when']}")
                st.write("---")
        else:
            st.warning("No tasks found in the provided text.")
    else:
        st.error("Please enter a paragraph before extracting tasks.")
