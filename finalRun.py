
import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
# Title of the window screen
window.title("Smart Attendance System")

#Displaying title on main window
message = tk.Label(window, text="DexSchool Smart Attendance System using Face Recognition", bg="cyan", fg="black", width=50,
                   height=3, font=('times', 30, 'italic bold '))
message.place(x=80, y=20)



# background colour of window
window.configure(background='snow')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Logo for the window
window.iconbitmap('logo.ico')

# Label for ID
lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="black"  ,bg="PaleGreen" ,font=('times', 15, ' bold ') ) 
lbl.place(x=50, y=200)
# taking text for ID
txt = tk.Entry(window,width=20  ,bg="misty rose" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=50, y=260)
# label for name
lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="black"  ,bg="PaleGreen"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=50, y=350)
# Taking text for name
txt2 = tk.Entry(window,width=20  ,bg="misty rose"  ,fg="black" , font=('times', 15, ' bold ')  )
txt2.place(x=50, y=410)
# Lable for displaying notification
lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="black"  ,bg="SeaGreen"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=500, y=200)
# Space for displaying message notification
message = tk.Label(window, text="" ,bg="misty rose"  ,fg="black"  ,width=30  ,height=2, activebackground = "cyan" ,font=('times', 15, ' bold ')) 
message.place(x=500, y=250)
# Label for displaying Attendance
lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="black"  ,bg="SeaGreen"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=500, y=360)
# Space for displaying message related to attendance
message2 = tk.Label(window, text="" ,fg="black"   ,bg="misty rose",activeforeground = "cyan",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=500, y=410)
# Function for clearing text of ID
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)
# Function for cleraing text of name
def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
# Function to check if string in python is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
# Function for taking Images
def TakeFigs():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>70:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
# Function for  training model
def TrainFigs():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Your model is Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    #code to get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    
    #code to create empth face list
    faces=[]
    #code to create empty ID list
    Ids=[]
    #code to loop through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #Converting images into grayscale after uploading them
        pilImage=Image.open(imagePath).convert('L')
        # converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extracting the face from images training sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackFigs():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)


# Clear button for clearing IDs
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="paleGreen"  ,width=5  ,height=1 ,activebackground = "cyan" ,font=('times', 15, ' bold '))
clearButton.place(x=260, y=260)

# Clear button for clearing names
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="black"  ,bg="PaleGreen"  ,width=5  ,height=1, activebackground = "cyan" ,font=('times', 15, ' bold '))
clearButton2.place(x=260, y=410)   
 
# Code for taking images
takeImg = tk.Button(window, text="Take Images", command=TakeFigs ,fg="black"  ,bg="SeaGreen"  ,width=20  ,height=3, activebackground = "cyan" ,font=('times', 15, ' bold '))
takeImg.place(x=1000, y=200)

# Code for training images
trainImg = tk.Button(window, text="Train Images", command=TrainFigs  ,fg="black"  ,bg="PaleGreen"  ,width=20  ,height=3, activebackground = "cyan" ,font=('times', 15, ' bold '))
trainImg.place(x=1000, y=270)

# Code for tracking images
trackImg = tk.Button(window, text="Track Images", command=TrackFigs  ,fg="black"  ,bg="SeaGreen"  ,width=20  ,height=3, activebackground = "cyan" ,font=('times', 15, ' bold '))
trackImg.place(x=1000, y=350)

# Code for quitting window
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="black"  ,bg="PaleGreen"  ,width=20  ,height=3, activebackground = "cyan" ,font=('times', 15, ' bold '))
quitWindow.place(x=1000, y=425)

 
window.mainloop()
