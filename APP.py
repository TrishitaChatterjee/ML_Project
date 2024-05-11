import streamlit as st
import pickle
st.title('MPG ML Project')

displacement=st.number_input('Displacement',value=300, placeholder="enter any number for the displacement")

horsepower = st.number_input('Horsepower', value = 130, placeholder='Enter value for Horsepower')

weight = st.number_input('Weight', value = 3000, placeholder='Enter value for Weight')

acceleration = st.number_input('Acceleration', value = 12, placeholder='Enter value for Acceleration')

loaded_model = pickle.load(open('mpg_regression.sav','rb'))
prediction = loaded_model.predict([[displacement,horsepower,weight,acceleration]])
st.subheader(f'predicted mpg value for above parameter is {prediction[0]}')
st.write(prediction)
