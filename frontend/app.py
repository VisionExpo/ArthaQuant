import streamlit as st
import requests
from datetime import datetime

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="ArthaQuant", layout="wide")

st.title("📈 ArthaQuant — Paper Trading Dashboard")

user_id = st.sidebar.number_input("User ID", min_value=1, value=1)

page = st.sidebar.selectbox(
    "Navigate",
    ["Predict", "Paper Trade", "Portfolio", "Analytics", "Drift"],
)

def api_get(path):
    return requests.get(f"{API_BASE}{path}").json()

def api_post(path, payload):
    return requests.post(f"{API_BASE}{path}", json=payload).json()
