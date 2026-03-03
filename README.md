ARIL: Law Assistant
ARIL (Artificial Intelligence in Law Assistance) is a specialized legal reference tool designed to provide instant, accurate access to legal codes, statutes, and judicial definitions. Built for students, researchers, and legal enthusiasts, it acts as a digital bridge between complex legal texts and the end-user.

Project Objective:
The primary goal of this project is to eliminate "hallucinations" commonly found in general-purpose LLMs when citing specific laws. By using a structured retrieval approach, ARIL ensures that when a user provides a law code, they receive the exact verbatim text from the official gazette.

Scope of Data
This assistant is mapped to provide information on:

1. The New Criminal Laws (2024): Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA).

2. Historical Frameworks: Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and the Indian Evidence Act.

3. Civil & Land Litigation: Key sections related to property disputes, contract breaches, and land acquisition.

How it Works

1. Direct Code Lookup: Enter a specific section (e.g., "BNS Section 103" or "IPC 302") to retrieve the full legal definition, punishment criteria, and nature of the offense (Bailable/Non-Bailable).

2. Semantic Context Search: If the section number is unknown, users can describe a situation (e.g., "What is the law for defamation?"), and the system uses NLP to find the most relevant legal provision.

3. Cross-Reference Mapping: A unique feature that maps old IPC sections to their new corresponding BNS sections for comparative study.

Technical Architecture

1. Language: Python 3.10+

2. Data Structure: Structured JSON/CSV indexing for 100% accuracy.

3. Search Engine: TF-IDF or SBERT (Sentence-BERT) for semantic understanding of legal queries.

4. Interface: Streamlit / CLI for a lightweight, fast user experience.
