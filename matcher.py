"""Scheme Finder - matching engine.
Pipeline: user text -> profile extraction -> embedding similarity -> eligibility rules -> explanation.
"""
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "data/schemes.csv"

# ---------- Profile extraction (English + Hindi keywords) ----------
OCCUPATIONS = {
    "farmer": ["farmer", "farming", "agricultur", "kisan", "किसान", "खेती", "ರೈತ"],
    "student": ["student", "college", "school", "studying", "scholarship", "छात्र", "विद्यार्थी", "पढ़ाई", "ವಿದ್ಯಾರ್ಥಿ"],
    "artisan": ["artisan", "carpenter", "potter", "tailor", "blacksmith", "weaver", "goldsmith", "cobbler", "mason", "barber", "कारीगर", "बढ़ई"],
    "street vendor": ["street vendor", "hawker", "vendor", "thela", "ठेला", "रेहड़ी"],
    "entrepreneur": ["small business", "shop", "start a business", "own business", "self employed", "self-employed", "entrepreneur", "दुकान", "व्यापार"],
}
STATES = ["karnataka", "kerala", "tamil nadu", "maharashtra", "uttar pradesh", "bihar", "gujarat",
          "rajasthan", "punjab", "haryana", "telangana", "andhra pradesh", "west bengal", "odisha",
          "madhya pradesh", "assam"]
STATE_HI = {"कर्नाटक": "karnataka", "ಕರ್ನಾಟಕ": "karnataka", "बिहार": "bihar", "उत्तर प्रदेश": "uttar pradesh"}


def extract_profile(text: str) -> dict:
    t = text.lower()
    p = {"age": None, "income": None, "state": None, "gender": None,
         "occupation": None, "category": None, "flags": set()}

    m = re.search(r"(\d{1,3})\s*(?:years?|yrs?|yr|साल|वर्ष|वर्षीय|ವರ್ಷ)", t) or \
        re.search(r"\bage\s*(?:is|of|:)?\s*(\d{1,3})", t) or \
        re.search(r"\b(?:i am|i'm|im)\s*(\d{1,3})\b", t)
    if m:
        p["age"] = int(m.group(1))
    elif re.search(r"senior citizen|old age|elderly|बुजुर्ग|वृद्ध", t):
        p["age"] = 65

    # income: "income 1 lakh", "2 lakh", "income 120000", "income Rs 50,000"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|लाख)", t)
    if m:
        p["income"] = int(float(m.group(1)) * 100000)
    else:
        m = re.search(r"income[^\d]{0,15}(\d[\d,]*)", t) or re.search(r"(?:rs\.?|₹)\s*(\d[\d,]*)", t)
        if m:
            p["income"] = int(m.group(1).replace(",", ""))

    for s in STATES:
        if s in t:
            p["state"] = s
    for hi, en in STATE_HI.items():
        if hi in t:
            p["state"] = en

    if re.search(r"\b(woman|women|female|girl|mother|widow|wife|daughter|lady)\b|महिला|औरत|विधवा|ಮಹಿಳೆ", t):
        p["gender"] = "female"
    elif re.search(r"\b(man|male|boy|father|husband|son)\b|पुरुष", t):
        p["gender"] = "male"

    for occ, words in OCCUPATIONS.items():
        if any(w in t for w in words):
            p["occupation"] = occ
            break

    if re.search(r"\bsc\b|scheduled caste|dalit|अनुसूचित जाति", t):
        p["category"] = "sc"
    elif re.search(r"\bst\b|scheduled tribe|आदिवासी", t):
        p["category"] = "st"
    elif re.search(r"\bobc\b", t):
        p["category"] = "obc"

    if re.search(r"\bbpl\b|below poverty|poor|ration card|गरीब|गरीबी", t):
        p["flags"].add("bpl")
    if re.search(r"widow|विधवा", t):
        p["flags"].update(["widow", "bpl"] if "poor" in t or "bpl" in t else ["widow"])
        p["gender"] = "female"
    if re.search(r"disab|handicap|divyang|विकलांग|दिव्यांग|ಅಂಗವಿಕಲ", t):
        p["flags"].add("disabled")
    if re.search(r"pregnan|expecting|गर्भवती", t):
        p["flags"].add("pregnant")
        p["gender"] = "female"
    if re.search(r"no house|homeless|kutcha|kachcha|mud house|without a house|बेघर|कच्चा", t):
        p["flags"].add("homeless")
    if re.search(r"village|rural|gram|गाँव|गांव|ಹಳ್ಳಿ", t) or p["occupation"] == "farmer":
        p["flags"].add("rural")
    return p


# ---------- Embeddings (with TF-IDF fallback so the demo never breaks) ----------
class Matcher:
    def __init__(self, csv_path=CSV_PATH):
        self.df = pd.read_csv(csv_path).fillna("")
        self.df["text"] = (self.df["name"] + ". " + self.df["description"] + ". " +
                           "Benefit: " + self.df["benefit"] + ". " + self.df["occupation"].astype(str) + " " +
                           self.df["special"].astype(str))
        self.mode = "embeddings"
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.scheme_vecs = self.model.encode(self.df["text"].tolist(), normalize_embeddings=True)
        except Exception as e:  # offline / model not downloaded
            print("Embedding model unavailable, using TF-IDF fallback:", e)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.mode = "tfidf"
            self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.scheme_vecs = self.vec.fit_transform(self.df["text"])

    def similarity(self, query: str):
        if self.mode == "embeddings":
            q = self.model.encode([query], normalize_embeddings=True)
        else:
            q = self.vec.transform([query])
        return cosine_similarity(q, self.scheme_vecs)[0]

    # ---------- Eligibility rules + explanation ----------
    def check(self, row, p):
        """Returns (status, reasons, warnings). status: 'eligible' | 'check' | 'no'"""
        reasons, warnings = [], []

        if row["state"].lower() != "all":
            if p["state"] and p["state"] != row["state"].lower():
                return "no", [], []
            if not p["state"]:
                warnings.append(f"Only for residents of {row['state'].title()}; please confirm your state")
            else:
                reasons.append(f"you live in {p['state'].title()}")

        if p["age"] is not None:
            if not (row["min_age"] <= p["age"] <= row["max_age"]):
                return "no", [], []
            if row["min_age"] > 18 or row["max_age"] < 100:
                reasons.append(f"age {p['age']} fits the {row['min_age']}-{row['max_age']} range")
        elif row["min_age"] >= 18 and row["max_age"] < 100:
            warnings.append(f"Age limit {row['min_age']}-{row['max_age']}; your age was not mentioned")

        if str(row["gender"]).lower() != "all":
            if p["gender"] and p["gender"] != row["gender"].lower():
                return "no", [], []
            if p["gender"]:
                reasons.append("this scheme is for women")
            else:
                warnings.append("For women only; gender not mentioned")

        occ = str(row["occupation"]).lower()
        if occ != "all":
            if p["occupation"] and p["occupation"] != occ:
                return "no", [], []
            if p["occupation"] == occ:
                reasons.append(f"you are a {occ}")
            else:
                warnings.append(f"For {occ}s; occupation not mentioned")

        cat = str(row["category"]).lower()
        if cat != "all":
            if p["category"] and p["category"] != cat:
                return "no", [], []
            if p["category"] == cat:
                reasons.append(f"you belong to {cat.upper()} category")
            else:
                warnings.append(f"For {cat.upper()} category; category not mentioned")

        if row["max_income"] and row["max_income"] > 0:
            if p["income"] is not None:
                if p["income"] > row["max_income"]:
                    return "no", [], []
                reasons.append(f"income Rs {p['income']:,} is under the limit of Rs {int(row['max_income']):,}")
            else:
                warnings.append(f"Family income must be under Rs {int(row['max_income']):,}; income not mentioned")

        special = [s for s in str(row["special"]).lower().split(";") if s and s != "all"]
        for s in special:
            if s in p["flags"]:
                reasons.append({"bpl": "you mentioned low income / BPL", "widow": "you are a widow",
                                "disabled": "you mentioned a disability", "pregnant": "you are pregnant",
                                "homeless": "you do not have a pucca house", "rural": "you live in a rural area"}.get(s, s))
            elif s != "rural":
                warnings.append({"bpl": "Needs BPL / low-income status", "widow": "For widows",
                                 "disabled": "Needs a disability certificate", "pregnant": "For pregnant women",
                                 "homeless": "For families without a pucca house"}.get(s, s))

        status = "eligible" if not warnings and reasons else ("check" if not warnings else "check")
        if not reasons and not warnings:
            reasons.append("open to all citizens")
            status = "eligible"
        return status, reasons, warnings

    def recommend(self, query: str, top_k=5):
        profile = extract_profile(query)
        sims = self.similarity(query)
        results = []
        for i, row in self.df.iterrows():
            status, reasons, warnings = self.check(row, profile)
            if status == "no":
                continue
            # Score: semantic similarity + bonus for confirmed rule matches - penalty for unknowns
            score = float(sims[i]) + 0.12 * len(reasons) - 0.05 * len(warnings)
            results.append({
                "name": row["name"], "benefit": row["benefit"], "status": status,
                "reasons": reasons, "warnings": warnings,
                "documents": [d.strip() for d in row["documents"].split(";") if d.strip()],
                "link": row["link"], "score": round(score, 3), "similarity": round(float(sims[i]), 3),
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return profile, results[:top_k]


if __name__ == "__main__":
    m = Matcher()
    tests = [
        "I am a farmer in Karnataka with 2 acres, age 45, income 1 lakh",
        "I am a 68 year old poor widow living in a village",
        "मैं एक किसान हूँ और मेरे पास कच्चा घर है",
        "I am an SC student, 20 years old, family income 2 lakh",
    ]
    for t in tests:
        prof, res = m.recommend(t, 3)
        print("\n>>", t, "\n   profile:", prof)
        for r in res:
            print("  -", r["name"], r["status"], r["score"], "|", "; ".join(r["reasons"]))
