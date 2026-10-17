# Scheme Finder

Explainable, multilingual matching of citizens to Indian government schemes. Built for IBM Z Datathon 2026 (Wildcard: Beyond the Obvious).

## Problem
Millions of eligible Indians miss benefits because schemes are scattered, complex and mostly in English.

## How it works
1. User describes their situation in plain English or Hindi.
2. A profile (age, income, state, occupation, category, flags) is extracted.
3. A multilingual sentence-transformer (paraphrase-multilingual-MiniLM-L12-v2) embeds the text and schemes; cosine similarity ranks them.
4. A rule engine removes schemes the person is clearly ineligible for (age, income, state, gender, category).
5. Each result shows why it matches, what to confirm, and the documents needed.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## IBM Z / LinuxONE
Prototype runs on standard Linux. Planned: containerise the matcher (s390x) on LinuxONE for high-throughput, secure, low-latency service with pervasive encryption and on-chip AI acceleration.

## Data and open-source
Schemes curated from https://www.myscheme.gov.in and official scheme portals (verify before real use). Libraries: sentence-transformers, scikit-learn, pandas, Streamlit.

## Privacy
No user input is stored.
