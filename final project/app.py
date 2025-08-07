import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time
#title
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('california housing price prediction')
st.image('https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg')
st.header('model of housing prices to predict median house values in California',divider=True)

st.sidebar.title('select your house feature 🏠')
st.sidebar.image('https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg')
st.sidebar.image('https://www.opalexteriors.com/wp-content/uploads/2016/09/Anatomy-of-a-house-exterior-infographic-opal-web2-e1499442846661.jpg')
temp_df = pd.read_csv('california.csv')
random.seed(12)
all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)
    
ss = StandardScaler()
ss.fit(temp_df[col])
final_value= ss.transform([all_values])
with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)
price = chatgpt.predict(final_value)[0]

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar=st.progress(0)
placeholder=st.empty()
placeholder.subheader('predicting price')
place=st.empty()
place.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS02aleLkDuc1l0VaLGRMC7s7F5Wek15T0WhQ&s',width=300)
if price>0:
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body= f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()

    st.success(body)
else:
    body = 'invalid house feature values'
    st.warning(body)
    
        