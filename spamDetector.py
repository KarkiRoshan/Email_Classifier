import pickle
import streamlit as st


model = pickle.load(open("spam.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))
def main():
    st.title("Email Spam Classification App")
    msg = st.text_input("Enter a text")
    if st.button("Predict"):
        data = [msg]
        
        prediction = model.predict(cv.transform(data).toarray())
        result =prediction[0]
        if result==1:
            st.error("This is a spam email")
        else:
            st.success("This is not a spam email")


main()