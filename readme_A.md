# Task Extraction and Categorization Pipeline

## **Introduction**
This document outlines the implementation of an NLP pipeline to extract and categorize tasks from unstructured text without using LLMs. The pipeline is designed to identify task-related phrases, responsible entities (who), and deadlines (when). The implementation is modular, ensuring flexibility and scalability.

## **Implementation Steps**

### **1. Preprocessing**
Preprocessing involves cleaning the text, tokenizing it, and tagging parts of speech to aid in structured extraction.

#### **Preprocessing Function**
```python
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")
```

### **2. Task Extraction**
The core functionality of the pipeline is to extract actionable tasks, their subjects, and deadlines using heuristic-based keyword matching and Named Entity Recognition (NER).

#### **Task Extraction Function**
```python
# Expanded Task Keywords
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
```

### **3. Task Categorization**
Extracted tasks are categorized into meaningful groups to improve usability.

#### **Categorization Function**
```python
# Categorization Rules
CATEGORY_MAP = {
    "Shopping & Planning": {"buy", "order", "book"},
    "Household & Maintenance": {"clean", "fix", "repair"},
    "Work & Emails": {"send", "report", "write", "submit"},
    "Meetings & Coordination": {"schedule", "attend", "arrange", "call", "plan"},
    "General Task": {"review", "prepare", "update", "research", "analyze", "execute"}
}

def categorize_task(task):
    for category, keywords in CATEGORY_MAP.items():
        if any(word in task.lower() for word in keywords):
            return category
    return "General Task"

def categorize_extracted_tasks(extracted_tasks):
    for task in extracted_tasks:
        task["category"] = categorize_task(task["task"])
    return extracted_tasks
```

### **4. Validation with a Sample Paragraph**
```python
sample_text = "John needs to submit his project by Friday evening. Meanwhile, Priya is planning to book flight tickets for her vacation next Monday. Alex has to review the financial report before the meeting on Wednesday morning. Sneha must call the technician to fix the air conditioner tomorrow. In addition, Rahul is organizing a team lunch this weekend."

extracted_tasks = extract_tasks(sample_text)
categorized_tasks = categorize_extracted_tasks(extracted_tasks)

for task in categorized_tasks:
    print(task)
```

## **Challenges Faced & Insights**
### **Challenges:**
1. **Ambiguous Subjects** - Identifying who is responsible when a task is written in passive voice.
2. **Time Extraction** - Extracting date/time phrases correctly without hardcoded rules.
3. **Contextual Understanding** - Some tasks have multiple possible interpretations based on surrounding text.

### **Insights:**
- **SpaCy's Named Entity Recognition (NER)** significantly improved the identification of responsible persons.
- **Keyword-based heuristics** proved to be an effective alternative to dependency parsing for task extraction.
- **Categorization** was refined through iterative testing with multiple text samples.

## **Conclusion**
This NLP pipeline effectively extracts structured tasks from free text without using LLMs. It combines SpaCy's NER and heuristic-based rule sets for improved performance. Future improvements could include:
- **Dependency parsing** for more context-aware extractions.
- **Machine learning-based categorization** for improved classification accuracy.
- **UI Implementation (e.g., Streamlit)** for ease of use in real-world applications.

---
📌 **End of Part A Documentation** ✅

