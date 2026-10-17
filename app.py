import streamlit as st
from matcher import Matcher

st.set_page_config(page_title="Scheme Finder", page_icon="🇮🇳", layout="centered")


@st.cache_resource
def load():
    return Matcher()


matcher = load()

st.title("🇮🇳 Scheme Finder")
st.caption("Tell us about yourself in English or Hindi. We'll find government schemes you may qualify for, "
           "explain why, and list the documents you need. Nothing you type is stored.")

examples = {
    "Farmer (Karnataka)": "I am a farmer in Karnataka, age 45, family income 1 lakh, I have 2 acres of land",
    "Widow (village)": "I am a 68 year old poor widow living in a village",
    "SC student": "I am an SC student, 20 years old, family income 2 lakh, studying in college",
    "Hindi": "मैं एक किसान हूँ और मेरे पास कच्चा घर है",
    "Pregnant woman": "I am a 24 year old pregnant woman from a poor family",
}

choice = st.selectbox("Try an example (or type your own below)", ["-- type my own --"] + list(examples))
default = examples.get(choice, "")
query = st.text_area("Describe your situation", value=default, height=110,
                     placeholder="e.g. I am a woman farmer in Karnataka, age 40, income 1 lakh...")

if st.button("Find my schemes", type="primary") and query.strip():
    profile, results = matcher.recommend(query, top_k=5)

    with st.expander("What we understood from your text"):
        shown = {k: (sorted(v) if isinstance(v, set) else v) for k, v in profile.items() if v}
        st.json(shown if shown else {"note": "Nothing specific detected, add age, income, state, occupation."})

    if not results:
        st.warning("No matching schemes found. Try adding your age, income, state and occupation.")

    for r in results:
        with st.container(border=True):
            badge = "✅ Likely eligible" if r["status"] == "eligible" else "🟡 Check details"
            st.subheader(r["name"])
            st.markdown(f"**{badge}** · Match score: `{r['score']}`")
            st.markdown(f"**Benefit:** {r['benefit']}")
            if r["reasons"]:
                st.markdown("**Why it matches you:** " + "; ".join(r["reasons"]).capitalize() + ".")
            if r["warnings"]:
                st.markdown("**To confirm:**")
                for w in r["warnings"]:
                    st.markdown(f"- {w}")
            st.markdown("**Documents you will need:**")
            for d in r["documents"]:
                st.markdown(f"- ☐ {d}")
            st.markdown(f"[Official page / apply]({r['link']})")

    st.info("Results are guidance, not a guarantee. Always confirm eligibility on the official website.")
