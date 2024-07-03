import streamlit as st
import pandas as pd


st.title("Hello world")

st.write("This is a demo app.")

df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

df

df = pd.DataFrame([
    {"a": 1, "b": 5},
    {"a": 2, "b": 6},
    {"a": 3, "b": 7},
    {"a": 4, "b": 8},
])

df

df = pd.read_csv("streamlit-app/data/test_data.csv")

df
