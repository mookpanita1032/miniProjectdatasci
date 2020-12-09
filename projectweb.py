import streamlit as st
import pandas as pd
import pickle 

st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    v_AcademicYear = st.sidebar.radio('AcademicYear', [' 2562','2563'])
    v_StudentType = st.sidebar.radio('StudentType', ['FOREIGN', 'THAI'])
    v_MajorName = st.sidebar.radio('MajorName', ['English', 'Aviation Services', 'Aviation Operations',
       'International Aviation Logistics Business'])  
    v_tcas = st.sidebar.selectbox('TCAS', [1,2,3,4,5])   
    v_Gpax = st.sidebar.slider('GPAX', 2.25, 4.00, 2.25)     
    v_Gpaen = st.sidebar.slider('GPA_Eng', 2.25, 4.00, 2.25)
    v_Gpamath = st.sidebar.slider('GPA_Math', 2.25, 0.24, 2.25)
    v_Gpasci = st.sidebar.slider('GPA_Sci', 2.25, 4.00, 2.25)
    v_Gpasco = st.sidebar.slider('GPA_Sco', 2.25, 4.00 , 2.25)
    v_facultyname = st.sidebar.selectbox('FacultyName', [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    

    # Change the value of sex to be {'M', 'F', 'I'} as stored in the trained dataset
    if v_facultyname == 'School of Liberal Arts':
            v_facultyname = '1'
    elif v_facultyname == 'School of Science':
            v_facultyname = '2'
    elif v_facultyname == 'School of Management':
            v_facultyname = '3'
    elif v_facultyname == 'School of Information Technology':
            v_facultyname = '4'
    elif v_facultyname == 'School of Agro-industry':
            v_facultyname = '5'
    elif v_facultyname == 'School of Law':
            v_facultyname = '6'
    elif v_facultyname == 'School of Cosmetic Science':
            v_facultyname = '7'
    elif v_facultyname == 'School of Health Science':
            v_facultyname = '8'
    elif v_facultyname == 'School of Nursing':
            v_facultyname = '9'
    elif v_facultyname == 'School of Medicine':
            v_facultyname = '10'
    elif v_facultyname == 'School of Dentistry':
            v_facultyname = '11'
    elif v_facultyname == 'School of Social Innovation':
            v_facultyname = '12'
    elif v_facultyname == 'School of Sinology':
            v_facultyname = '13'
    elif v_facultyname == 'School of Integrative Medicine':
            v_facultyname = '14'

    

    # Store user input data in a dictionary
    data = {'AcademicYear': v_AcademicYear,
            'StudentType': v_StudentType,
            'MajorName': v_MajorName,
            'TCAS': v_tcas,
            'GPAX': v_Gpax,
            'GPA_Eng': v_Gpaen,
            'GPA_Math': v_Gpamath,
            'GPA_Sci': v_Gpasci,
            'GPA_Sco': v_Gpasco,
            'FacultyName': v_facultyname}

    # Create a data frame from the above dictionary
    tcas1 = pd.DataFrame(data, index=[0])
    return tcas1

# -- Call function to display widgets and get data from user
tcas = get_input()

st.header('Status MFU')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(tcas)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('tcas16.csv')
tcas = pd.concat([tcas, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(tcas[['StudentType','MajorName']])

#Combine all transformed features together
X_new = pd.concat([cat_data,tcas], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#Drop un-used feature
X_new = X_new.drop(columns=['Unnamed: 0','StudentType','MajorName'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_sc = pickle.load(open('normalization5.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_sc.transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn5.pk1', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)