# streamlit run your_script.py
import streamlit as st
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

np.random.seed(seed=10)

st.set_page_config(layout='wide')

path = 'filtered_dataset.csv'
data = pd.read_csv(path)


data['sent_date'] = pd.to_datetime(data['sent_date'])
data['click_date'] = pd.to_datetime(data['click_date'])
data_filtered = data[data['click_date'].notna()]
data_filtered = data_filtered[data_filtered['transaction_amount'].notna()]
data_filtered['time_to_open']=(data_filtered['click_date'] - data_filtered['sent_date']).apply(lambda x:x.total_seconds() / 3600)

limit = st.sidebar.number_input(label='Limit to consider fast send', value=1.0)


data_filtered['fast_open'] = data_filtered['time_to_open'] < limit
data_filtered['sent_date_hour'] = data_filtered['sent_date'].apply(lambda x:x.hour)
data_filtered['sent_date_wday'] = data_filtered['sent_date'].apply(lambda x:x.day_of_week)

def random_day(x):
    return np.random.randint(0,7)
def random_hour(x):
    return np.random.randint(0,24)

data_filtered['sent_date_hour'] = data_filtered['sent_date_hour'].apply(random_hour)
data_filtered['sent_date_wday'] = data_filtered['sent_date_wday'].apply(random_day)

encoder = OneHotEncoder(sparse=False)
# encoder = OneHotEncoder()
# categorical_features = ["subject", "content", "sender", "recipient", "location", "day_of_week", "hour_of_day"]
# categorical_features = ['email_name','transaction_amount','sent_date_hour','sent_date_wday']
categorical_features = ['email_name','sent_date_hour','sent_date_wday']
encoded_features = encoder.fit_transform(data_filtered[categorical_features])

columns_filtered = np.append(encoder.get_feature_names_out(),'transaction_amount')
data_filtered_X_np = np.concatenate([encoded_features,data_filtered['transaction_amount'].to_numpy().reshape(-1,1)],axis=1)

feature_names = columns_filtered
encoded_data = pd.DataFrame(data_filtered_X_np, columns=feature_names)

# Split data_filtered into train and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_data, data_filtered["fast_open"], test_size=0.2, random_state=42)

# Train a logistic regression model
# model = LogisticRegression(max_iter=1000)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict the optimal time for a new email
# Assume we have a new email with some features given
# new_email = {"subject": "New offer", "content": "Buy one get one free", "sender": "sales@company.com", "recipient": "customer@gmail.com", "location": "São Paulo"}

# Predict the probability of opening for each possible time slot
# Assume we have 24 hours and 7 days to choose from
hours = np.arange(24)
days = np.arange(7)
probabilities = []
# 'sent_date_hour': 16,
#  'sent_date_wday': 3

# dia = st.sidebar.selectbox('Day',np.arange(7))
# st.sidebar.selectbox('Hour',np.arange(24))
# email_name = st.sidebar.text_area(label='email',value='Email 1 - Welcome to Wanderlust Adventures')
email_name = st.sidebar.selectbox(label='email', options=data_filtered['email_name'].unique())
transaction_amount = st.sidebar.number_input(label='transaction amount', value=613.12)
# st.write()

for day in days:
    for hour in hours:
        # Create a feature vector with the given day and hour
        time_feature = np.zeros(len(feature_names))
        # time_feature[feature_names.index(day)] = 1
        time_feature[np.where(columns_filtered==f'sent_date_wday_{day}')[0][0]] = 1
        # time_feature[feature_names.index(hour)] = 1
        time_feature[np.where(columns_filtered==f'sent_date_hour_{hour}')[0][0]] = 1
        # Combine the email features and the time feature
        # new_email = {
        #     'email_name': 'Email 1 - Welcome to Wanderlust Adventures',
        #     'sent_date_hour':hour,
        #     'sent_date_wday':day
        #     }
        new_email = [email_name,hour,day]

        # Encode the new email using the same encoder
        encoded_email = encoder.transform([new_email])        
        email_time_feature = np.concatenate([encoded_email, [[transaction_amount]]],axis=1)
        # Predict the probability of opening using the model
        probability = model.predict_proba(email_time_feature)[0][1]
        # Append the probability and the corresponding time slot to a list
        probabilities.append((probability, (day, hour)))
# Sort the list by descending probability

probabilities.sort(reverse=True)
# Choose the time slot with the highest probability as the optimal time
optimal_time = probabilities[0][1]
# Convert the optimal time to a human-readable format
# day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_names = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sabado", "domingo"]
optimal_day = day_names[optimal_time[0]]
optimal_hour = optimal_time[1]
if optimal_hour < 12:
    optimal_hour = f"{optimal_hour} AM"
elif optimal_hour == 12:
    optimal_hour = "12 PM"
else:
    optimal_hour = f"{optimal_hour - 12} PM"

st.title('Melhor horário para disparar email mkt')

st.write('Amostra do dataset')
st.write(data_filtered.head())

fig = px.histogram(data_filtered,'time_to_open',title='Distribuicao dos tempos ate abertura do email', labels={'time_to_open':"Tempo para abertura (h)"})
fig.update_layout(bargap=0.2)
fig.add_vline(x=limit,line_color="red")
st.write(fig)

st.write('Probabilidade para cada dia e horário para este usuário:')

st.write(pd.DataFrame(probabilities,columns=['Prob','Day Hour']))


st.write(f"O tempo otimo para envio deste email eh {optimal_day}, {optimal_hour}.")
