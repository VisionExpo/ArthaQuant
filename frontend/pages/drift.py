import streamlit as st
import random
from frontend.app import api_post

st.header("🚨 Drift Detection")

if st.button("Check Drift"):
    reference = [random.random() for _ in range(100)]
    current = [random.random() for _ in range(100)]

    out = api_post("/drift", {
        "reference": reference,
        "current": current,
    })

    st.json(out)
