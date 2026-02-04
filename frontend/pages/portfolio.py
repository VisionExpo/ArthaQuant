import streamlit as st
from frontend.app import api_get

st.header("📊 Portfolio")

data = api_get("/portfolio")

st.subheader("Cash")
st.write(data["cash"])

st.subheader("Positions")
st.json(data["positions"])

st.subheader("Trades")
st.json(data["trades"])
