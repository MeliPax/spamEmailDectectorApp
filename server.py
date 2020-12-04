import streamlit as st
import pandas as pd
import numpy as np



def main():
    # st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">SpamEmail Classification App</h2>
    </div>
    <br><br>
    <div>
        <p>Enter email text:</p>
        <p><textarea  name = "email" rows="4" cols="80"/></p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.button('Classify')
   
    # if st.button('Classify'):
    #     if option=='Linear Regression':
    #         st.success(classify(lin_model.predict(inputs)))
    #     elif option=='Logistic Regression':
    #         st.success(classify(log_model.predict(inputs)))
    #     else:
    #        st.success(classify(svm.predict(inputs)))

# def classify(num):
#     if num < 0.5:
#         return 'Setosa'
#     elif num < 1.5:
#         return 'Versicolor'
#     else:
#         return 'Virginica'
if __name__ == '__main__':
    main()


