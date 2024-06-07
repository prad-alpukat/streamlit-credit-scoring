import streamlit as st

test = "testing"

test2 = False

def print():
    st.write(test)
    global test2
    st.chat_input("test")
    test2 = True

st.write(test2)

st.button("Print", on_click=print)