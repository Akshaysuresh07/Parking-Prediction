
import streamlit as st
import datetime as dt
import pandas as pd
import pickle

def main():

    st.set_page_config(page_title="Parking Predictor")
    st.title("Parking Predictor")

    p1, p2 = st.columns([1, 1])

    # Get date and time inputs (defaults to current date and time)
    date_predict = p1.date_input('Enter date:.', dt.date.today(), key='1') 
    time_now_predict = dt.datetime.now() + dt.timedelta(hours=2) # Based on server time
    # time_now_predict = dt.datetime.now() # Based on local machine time
    time_now_predict = time_now_predict.strftime("%H:%M")
    time_now_predict = p2.text_input("Enter time:", value=time_now_predict, key='1') 

    # Load trained model
    model1 = pickle.load(open('model/parking1.sav', 'rb'))
    model2 = pickle.load(open('model/parking2.sav', 'rb'))
    model3= pickle.load(open('model/parking3.sav', 'rb'))
    model4= pickle.load(open('model/parking4.sav', 'rb'))
    # Arrange date and time inputs as a DataFrame and extract features
    predict_df = pd.DataFrame({'date': str(date_predict) + ' ' + time_now_predict}, index=[0])

    predict_df['date'] = pd.to_datetime(predict_df['date'])
    df=predict_df
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['hour_min'] = round(df['hour'] + (df['minute'] / 60), 1)
    df['dayofweek']= df['date'].dt.dayofweek
    df.drop(['hour','minute','date'],axis=1)
    predict_df=df

    # Generate prediction and predicted probabilities
    pred=[]
    pred.append ( model1.predict(predict_df)[0])
    pred_proba=model1.predict_proba(predict_df)[0]
    pred.append( model2.predict(predict_df)[0])
    pred_proba = model1.predict_proba(predict_df)[0]
    pred.append( model3.predict(predict_df)[0])
    pred_proba = model1.predict_proba(predict_df)[0]
    pred.append( model4.predict(predict_df)[0])
    pred_proba = model1.predict_proba(predict_df)[0]
  
    
    # Result message
    i=0
    result_df=pd.DataFrame({"Parking Lot": ['Parking Lot 1','parking Lot 2','parking Lot 3','parking Lot 4'],"Occupancy":[str(pred[0]),(pred[1]),(pred[2]),(pred[3])]})
    st.markdown("Parking Availaility at :"+time_now_predict)
    print(result_df['Parking Lot'].count())
    for i in range(0,4):
        if pred[i]=='p1':
            st.markdown('Parking Lot No:'+str(i+1)+' Has the Highest Priority')
            f1=1
    for i in range(0,4):
        if pred[i]=='p2':
            st.markdown('Parking Lot No:'+str(i+1)+' Has the Second Highest Priority')
            f2=1
        
    for i in range(0,4):
        if pred[i]=='p3':
            st.markdown('Parking Lot No:'+str(i+1)+' Has the third Highest Priority')
            
    for i in range(0,4):
        if pred[i]=='p4':
            st.markdown('Parking Lot No:'+str(i+1)+' Has the Least Priority')

    #df['order']=('')
    s1=st.columns([1,7])
    st.map(data=None, zoom=None, use_container_width=True)
    result_df['Occupancy']=result_df['Occupancy'].replace('p1','Less than 50% Occupied')         
    result_df['Occupancy']=result_df['Occupancy'].replace('p2','Between 50% and 75% occupied') 
    result_df['Occupancy']=result_df['Occupancy'].replace('p3','Between 75% and 90% occupied') 
    result_df['Occupancy']=result_df['Occupancy'].replace('p4','More than 90% Occupied') 
    with st.expander('View prediction results'):
        s1 = st.columns([1, 7])
        #s1.markdown(f"<h4 style='text-align: center; color: Gray;'>Priorities of Lot-1 Lot-2 Lot-3 Lot-4:{pred} ", unsafe_allow_html=True)
        st.table(result_df)

    

if __name__ == "__main__":
    main()



