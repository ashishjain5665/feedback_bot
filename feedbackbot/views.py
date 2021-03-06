from django.shortcuts import render
from django.http import HttpResponse
import os
import sys
import time
import playsound
import speech_recognition as sr
from gtts import gTTS
import datetime
from . import settings
################################################  Speak function  ##############################################
def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
################################################ audio function(record voice)  ####################################
def get_audio():
    sample_rate = 48000
    chunk_size = 2048 
    r = sr.Recognizer()  
    with sr.Microphone(sample_rate = sample_rate,  
                            chunk_size = chunk_size) as source: 
        r.adjust_for_ambient_noise(source)                                  #adjusting ambient noise 
        print ("Say Something")
        audio = r.listen(source)                                            #listening voice
        try: 
            text = r.recognize_google(audio)                                #recognize voice using google
            print( "you said: " +text)
        except:
            print("sorry")
    return "sorry"
######################################################## for initialize the page ###################################################
def without(request):                               
    return render(request , 'home.html')                        #render to home
######################################################## for start button in home page ##############################################
def tran_to_botstart(request):                      
    return render(request , 'botstart.html')                    #render to botstart page

################################################### onloadding the botstart page ####################################################    
def output(request):                            
    speak("hello  i am sam,  i am your feedback assistant today    ,please speak your review")   
    data = get_audio()
    if data=="sorry":                                                               #data is not recognized
        speak("we did'nt uderstand please again speak")
        data=get_audio()
    f=open(os.path.join(settings.BASE_DIR,'assets\\hii.txt'),'r+')                  #pick up hii.txt from static files
    f.truncate()                                                                    #deleting all the content of file
    f.write(data)                                                                   #write review in file
    f.close()
    return render(request , 'check.html',{'data':data})                             #render to check page
################################################## function for prediction (review analysis) #########################################
def predict(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    import nltk
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    
    ############################### function to clean the reviews in the sata set 
    def clean(review):
        review = re.sub('[^a-zA-Z]', ' ', review)                                               #removing unusable character
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()                                                                    #stemmer object
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))-{'not','worst','bad'}]    #taking only useful words
        review = ' '.join(review)
        return review
    ################ reading the dataset
    dataset = pd.read_csv(os.path.join(settings.BASE_DIR,'assets\\rr.tsv'), delimiter = '\t', quoting = 3)
    corpus = []
    for i in range(0, 1001):
        review = dataset['Review'][i]
        review = clean(review)
        corpus.append(review)
    ################### Creating the Bag of Words model
    cv = CountVectorizer(max_features = 3000)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    ################### Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    ################### Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    ################### Predicting the Test set results
    y_pred = classifier.predict(X_test)
    ################### Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    text = clean(data)
    text = cv.transform([text]).toarray()
    text = classifier.predict(text)
    if text==[1]:
        return "positive"
    else:
        return "negative"
################################################## update data base and store the review ########################################        
def update_database(data):
    speak("we are analysing your review. please wait.")
    import mysql.connector
    mydb = mysql.connector.connect( host = "localhost",                                    #connection with database
                                   user = "root",
                                   passwd = "root", 
                                   database = "feedback")
    mycursor = mydb.cursor()
    mycursor.execute("select * from opinion")                                               #fetching all the reviews
    myresult = mycursor.fetchall()
    sno = myresult[-1][0]
    sno=sno+1                                                                               #increase the sno(primary key)
    s=datetime.date.today()
    ana  = predict(data)                                                                    #predicting the review
    sqlform = "Insert into opinion(sno,date,feedback,analysis) values(%s,%s,%s,%s)"
    opinions = [(sno,s,data,ana)]
    mycursor.executemany(sqlform, opinions)                                                  #inserting the new review into the 
    mydb.commit()                                                                            #commit the queriers
    return ana
########################################### Based on analysis (positive or negative) transfer the user to relative page ################### 
def yes(request):
    data=request.POST.get('param')
    ana = update_database(data)
    speak("thanks for using our service")
    if ana == "positive":
        return render(request , 'yes.html',{'data':ana})
    return render(request , 'no.html',{'data':ana})
######################################### if voice is not recognized correctly then again take the voice ###############################    
def tran_to_again(request):                      #for again button
    return render(request , 'again.html')
####################################### onloading the again.html page #############################################################
def again(request):                            
    speak("please again speak your review")
    data = get_audio()
    if data=="sorry":
        speak("we did'nt uderstand please again speak")
        data=get_audio()
    f=open(os.path.join(settings.BASE_DIR,'assets\\hii.txt'),'r+')
    f.truncate()
    f.write(data)
    f.close()
    return render(request , 'check.html',{'data':data})
############################################## transfer to login page ##############################################################
def tran_to_username(request):
    return render(request ,'username.html' )
############################################## if user credential is correct transfer it to next page ###############################
def logincheck(request):
    username = request.POST.get('param')
    password = request.POST.get('pass')
    import mysql.connector
    mydb = mysql.connector.connect( host = "localhost", 
                                   user = "root",
                                   passwd = "root", 
                                   database = "feedback")
    mycursor = mydb.cursor()
    query = "select username from admin"
    mycursor.execute(query)
    myresult = mycursor.fetchall()     
    if (username,) in myresult:
        query="select password from admin where username = " +"\'"+ username+"\'" 
        mycursor.execute(query)
        myresult = mycursor.fetchall()
        if myresult == [(password,)]:
            mycursor.execute("select analysis from opinion")
            myresult = mycursor.fetchall()
            x,y=0,0
            for i in myresult:
                if i==('positive',):
                    x+=1
                else:
                    y+=1
            mycursor.execute("select * from opinion")
            myresult = mycursor.fetchall()
            l=len(myresult)
            return render(request , 'userview.html',{'x':x,'y':y,'data':myresult,'l':l})
        else:
            return render(request , 'username.html',{'dat':'invalid password'})
    else:
        return render(request , 'username.html',{'data':'user does not exists'})
############################################ take voice(yes or no) and do appropriate action ######################################################
def spe(request):
    speak("please say yes or no to confirm")
    data=get_audio()
    if data=="sorry":
        speak("we did'nt uderstand please again speak")
        data=get_audio()
    if 'yes' in data:
        f=open(os.path.join(settings.BASE_DIR,'assets\\hii.txt'),'r+')
        dat=f.read()
        f.close()
        ana = update_database(dat)
        speak("thanks for using our service")
        if ana == "positive":
            return render(request , 'yes.html',{'data':ana})
        return render(request , 'no.html',{'data':ana})
    else:
        return render(request , 'again.html')
