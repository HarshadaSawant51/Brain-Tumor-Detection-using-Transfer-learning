from flask import Flask, render_template, request,session,send_from_directory,Response
from os import path
import csv
import pymysql
import os
import ctypes
from BrainTumor import BrainTumor
import numpy as np
import pandas as pd
from os import listdir
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import imutils
import matplotlib.pyplot as plt
import glob
import cv2
from os import listdir
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import pickle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
from os import listdir
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os

from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle #shuffling the data improves the model
import pathlib
from keras.applications.vgg16 import VGG16, preprocess_input
conn=pymysql.connect(host="localhost",user="root",password="",db="braintumor")
cursor=conn.cursor()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app=Flask(__name__)
app.secret_key = 'jsbcdsjkvbdjkbvdjcbkjf'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/user')
def userlogin():
    return render_template('userLogin.html')


@app.route('/admin1')
def adminlog1():
    username = request.args.get('username')
    password = request.args.get('password')
    if username == 'admin' and  password =='admin':
        return render_template('AdminHome.html')
    else:
        return render_template('admin.html')


@app.route('/adminhome')
def adminhome():

    return render_template('adminhome.html')



@app.route('/UserReg')
def UserReg():

    return render_template('UserReg.html')


@app.route('/UserRegister1')
def UserRegister1():
    name = request.args.get('name')
    username = request.args.get('username')
    password = request.args.get('password')
    email = request.args.get('email')
    phone = request.args.get('phone')
    gender = request.args.get('gender')
    address = request.args.get('address')
    result = cursor.execute(
        " insert into ureg(name,username,email,password,phone,gender,address)values('" + name + "','" + username + "','" + email + "','" + password + "','" + phone + "','"+gender+"','"+address+"')");
    conn.commit()

    if result > 0:
            ctypes.windll.user32.MessageBoxW(0, "Registration Sucess", "Registration Status", "color:green;")
            return render_template('UserLogin.html')
    else:
            ctypes.windll.user32.MessageBoxW(0, "Registration Fails", "Registration Status", "color:black;")
            return render_template('UserReg.html')

@app.route('/doctorReg')
def doctorReg():

    return render_template('doctorReg.html')


@app.route('/doctorregister1',methods=['POST'])
def doctorregister1():
    target = os.path.join(APP_ROOT, 'profiles/')
    for upload in request.files.getlist("file"):

        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        phone = request.form.get('phone')
        experience = request.form.get('experience')

        profilepic = upload.filename
        print(profilepic)
        destination = "/".join([target, profilepic])
        upload.save(destination)
        resultvalue = cursor.execute(
            "select * from dreg where username='" + username + "' and email='" + email + "'")
        conn.commit()

        userDetails = cursor.fetchall()
        if resultvalue > 0:
            ctypes.windll.user32.MessageBoxW(0, "Registration Status", "User Already Exist", "color:blue;")
            return render_template('doctorReg.html')
        else:
            result = cursor.execute(
                " insert into dreg(name,username,email,password,phone,experience,pic)values('" + name + "','" + username + "','" + email + "','" + password + "','" + phone + "','" + experience + "','" + profilepic + "')");
            conn.commit()
            if result > 0:
                ctypes.windll.user32.MessageBoxW(0, "Registration Status", "Registration Success", "color:green;")
                return doctor()
            else:
                ctypes.windll.user32.MessageBoxW(0, "Registration Status", "User Already Exist", "color:black;")
                return render_template('doctorReg.html')


@app.route('/doctor')
def doctor():

    return render_template('doctorlog.html')





@app.route('/UserLogin1')
def UserLogin1():
    username = request.args.get('username')
    password = request.args.get('password')

    result = cursor.execute(" select * from ureg where username='" + username + "' and password='" + password + "'")
    userDetails=cursor.fetchall()
    conn.commit()

    if result > 0:

        for user in userDetails:
            email=user[3]
            id = user[0]
            session['username'] = username
            session['email']=email
            session['id']=id
            return render_template('userHome.html')
    else:
        ctypes.windll.user32.MessageBoxW(0, "Login Fails", "Login Status", "color:black;")

        return render_template('userLogin.html')


@app.route('/doctorHome')
def doctorHome():
    return render_template('DoctorHome.html')


@app.route('/doctorLogin1')
def doctorLogin1():
    username = request.args.get('username')
    password = request.args.get('password')

    result = cursor.execute(
        " select * from dreg where username='" + username + "' and password='" + password + "' and status='authorized'")
    userDetails = cursor.fetchall()
    conn.commit()

    if result > 0:

        for user in userDetails:
            email = user[3]
            id = user[0]
            print(email)

            session['username'] = username
            session['email'] = email
            session['id'] = id
            return render_template('DoctorHome.html')


    else:
        ctypes.windll.user32.MessageBoxW(0, "Login Fails", "Login Status", "color:black;")

        return render_template('doctorlog.html')



@app.route('/Verifydoc')
def Verifydoc():
    image_names = os.listdir('./profiles')

    resultvalue = cursor.execute("select * from dreg")
    conn.commit()

    userDetails = cursor.fetchall()
    if resultvalue > 0:
        return render_template('alldoctors.html', userDetails=userDetails)
    else:
        ctypes.windll.user32.MessageBoxW(0, "Doctor Details are not available", "Doctor Status", "color:black;")

        return render_template('AdminHome.html')




@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("profiles", filename)




@app.route('/activatedoc')
def activatedoc():
        req_id=request.args.get('Id')
        query=cursor.execute("select * from dreg where user_id='"+req_id+"'")
        conn.commit()
        querydetails=cursor.fetchall()

        for user in querydetails:
            email=user[3]
            resultvalue = cursor.execute("update dreg set status='authorized' where  user_id='"+req_id+"' ")
            conn.commit()


            if resultvalue > 0:
                return Verifydoc()
            else:
                ctypes.windll.user32.MessageBoxW(0, "Doctor Verified Fails", "Doctor Status", "color:black;")

                return Verifydoc()



@app.route('/deactivatedoc')
def deactivatedoc():
        req_id=request.args.get('Id')
        query=cursor.execute("select * from dreg where user_id='"+req_id+"'")
        conn.commit()
        querydetails=cursor.fetchall()

        for user in querydetails:
            email=user[3]
            resultvalue = cursor.execute("update dreg set status='not verified' where  user_id='"+req_id+"' ")
            conn.commit()


            if resultvalue > 0:
                return Verifydoc()
            else:
                ctypes.windll.user32.MessageBoxW(0, "Doctor Verified Fails", "Doctor Status", "color:black;")

                return Verifydoc()





@app.route('/Logout')
def Logout():
    session.pop('username', None)
    session.pop('email', None)
    session.pop('user_id', None)

    return render_template('index.html')


@app.route('/docLogout')
def docLogout():
    session.pop('username', None)
    session.pop('email', None)
    session.pop('user_id', None)

    return render_template('index.html')



@app.route('/docprofiles')
def docprofiles():
    image_names = os.listdir('./profiles')

    resultvalue = cursor.execute("select * from dreg where status='authorized'")
    conn.commit()

    userDetails = cursor.fetchall()
    if resultvalue > 0:
        return render_template('viewdocprofiles.html', userDetails=userDetails)
    else:
        ctypes.windll.user32.MessageBoxW(0, "Doctor Details are not available", "Doctor Status", "color:black;")

        return render_template('userHome.html')




@app.route('/sendReq')
def sendReq():
    image_names = os.listdir('./profiles')

    resultvalue = cursor.execute("select * from dreg where status='authorized'")
    conn.commit()

    userDetails = cursor.fetchall()
    if resultvalue > 0:

        return render_template('sendReq.html', userDetails=userDetails)
    else:
        ctypes.windll.user32.MessageBoxW(0, "Doctor Details are not available", "Doctor Status", "color:black;")

        return render_template('userHome.html')


def crop_brain_contour(image,plot=False):

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    cv2.imwrite('result/'+session['username']+'_cropped.png', new_image)
    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Preprocessed Image')

        plt.show()


    return new_image



def preprocess_imgs(img, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    set_new.append(preprocess_input(img))
    return np.array(set_new)



@app.route('/sendReq1',methods=['POST'])
def sendReq1():
    target = os.path.join(APP_ROOT, 'profiles/')
    for upload in request.files.getlist("file"):

        dname = request.form.get('dname')
        descr = request.form.get('descr')
        uname_crop=session['username']+'_cropped.png'
        uname_thresh=session['username']+'_thresholded.png'
        uname_closed = session['username'] + '_closed.png'
        uname_canny = session['username'] + '_canny.png'
        uname_res=session['username'] + '_res.png'
        profilepic = upload.filename
        print(profilepic)
        destination = "/".join([target, profilepic])
        upload.save(destination)
        print(destination)

        IMG_SIZE = (256, 256)
        img = cv2.imread('profiles/'+profilepic)
        img = crop_brain_contour(img, plot=True)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img = img / 255.
        from keras.preprocessing import image
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)



        vggmodel = load_model("VGG_model.h5")
        test_f = vggmodel.predict(test_image)
        uname_feature = np.array_str(test_f)
        test_f = test_f.reshape(test_f.shape[0], -1)
        uname_feature_ex = test_f
        filename = 'finalized_model.sav'
        # load the model from disk
        svm_model = pickle.load(open(filename, 'rb'))

        predictions = svm_model.predict(test_f)

        if predictions[0] == 1:
            pred="Tumor predicted"
            result = cursor.execute(
                    " insert into request(reqby,pic,docname,descr,datee)values('" + session['username'] + "','" + profilepic + "','" + dname+ "','" + descr + "',now())");
            ##
            conn.commit()
            bt = BrainTumor()
            bt.detectS(profilepic,session['username'])
            #result = cursor.execute(
            #    " insert into userviewresult values('" + session['username'] + "','" + uname_crop + "','" + uname_thresh + "','" + uname_closed + "','" + uname_canny + "','" + uname_res+ "')");



            #conn.commit()
            uname=session['username']
            #data=viewresultUser(session['username'])
            return render_template("msg.html",uname_feature_ex=uname_feature_ex,uname=uname, pred=pred,uname_feature=uname_feature,msg="Tumor Detected and Your Request Sent to Doctor", color="bg-primary")
        else:
            #a = cursor.execute("insert into userviewresult(uname,img1) values('" + session['username'] + "','" + uname_crop+ "')");
            pred="Tumor not detected"
            #data=viewresultUser(session['username'])
            return render_template("msg.html",uname_feature_ex=uname_feature_ex,pred=pred,uname_feature=uname_feature,msg="Tumor Not Detected ",color="bg-warning")

@app.route('/viewreq')
def viewreq():
    image_names = os.listdir('./profiles')

    resultvalue = cursor.execute("select * from request where reqby='"+session['username']+"'")
    conn.commit()

    userDetails = cursor.fetchall()
    if resultvalue > 0:
        return render_template('viewreq.html', userDetails=userDetails)
    else:
        ctypes.windll.user32.MessageBoxW(0, "Request details are not available", "Request Status", "color:black;")

        return render_template('userHome.html')




@app.route('/char')
def char():
    username = request.args.get('dname')
    return render_template('char.html',username=username)




@app.route('/storeData')
def storeData():
    msgto=request.args.get('id')
    msg=request.args.get('msg')
    print("am in storedata")
    a=cursor.execute("insert into chat(msgfrom,msgto,message,datee) values('"+session['username']+"','"+msgto+"','"+msg+"',now())")
    conn.commit()
    if a > 0:
        return render_template('chat3.html', id=id, msg=msg)

    else:
        return render_template('chat3.html', id=id, msg=msg)


@app.route('/getdata')
def getdata():
    msgto=request.args.get('id')
    print(" am in get data")
    a=cursor.execute("select * from chat where (msgfrom='"+session['username']+"' and msgto='"+msgto+"') or (msgfrom='"+msgto+"' and msgto='"+session['username']+"')")
    mdel=cursor.fetchall()
    if a > 0:
        print(mdel)
        return render_template('getdata.html',mdel=mdel,msgto=msgto)




@app.route('/viewpatientreq')
def viewpatientreq():
    image_names = os.listdir('./profiles')

    resultvalue = cursor.execute("select * from request where docname='"+session['username']+"'")
    conn.commit()

    userDetails = cursor.fetchall()
    if resultvalue > 0:
        return render_template('dviewreq.html', userDetails=userDetails)
    else:
        ctypes.windll.user32.MessageBoxW(0, "Request details are not available", "Request Status", "color:black;")

        return render_template('DoctorHoe.html')



@app.route('/char1')
def char1():
    username = request.args.get('dname')
    return render_template('char1.html',username=username)

@app.route('/process')
def process():
    reqid=request.args.get('Id')
    resultvalue = cursor.execute("select * from request where id='" + reqid + "'")
    conn.commit()
    details=cursor.fetchall()
    if resultvalue > 0:
        image=''
        imgname=''
        lastid=''
        for r in details:
            image=r[3]
            imgname=str(r[0])
        bt = BrainTumor()
        name = str(1)
        status, img1, img2, img3, img4 = bt.detectS(image, imgname)
        if status:
            session["img1"]=img1
            session["img2"] = img2
            session["img3"] = img3
            session["img4"] = img4
            session["reqid"] = reqid
            cursor.execute("update request set status='completed' where id='"+reqid+"'")
            conn.commit()
            return descr()

        else:
           return render_template("dmsg.html",msg="Process Terminated ",color="bg-warning")
@app.route('/descr')
def descr():
    return render_template('descr.html')


@app.route('/descr1')
def descr1():
     descr=request.args.get('descr')
     a=cursor.execute("insert into result(reqid,descr,img1,img2,img3,img4,datee) values('"+session['reqid']+"','"+descr+"','"+session['img1']+"','"+session['img2']+"','"+session['img3']+"','"+session['img4']+"',now())")
     conn.commit()
     session.pop('img1', None)
     session.pop('img2', None)
     session.pop('img3', None)
     session.pop('img4', None)
     session.pop('reqid', None)
     if a > 0:
         ctypes.windll.user32.MessageBoxW(0, "Processing Success", "Request Status", "color:black;")

         return viewpatientreq()

     else:
         ctypes.windll.user32.MessageBoxW(0, "Processing Fails", "Request Status", "color:black;")

         return viewpatientreq()


def viewresultUser(uname):

    cursor.execute("select * from userviewresult where uname='"+uname+"'")
    details=cursor.fetchall()
    return details

@app.route('/viewresult')
def viewresult():
    reqid=request.args.get('Id')
    cursor.execute("select * from result where reqid='"+reqid+"'")
    details=cursor.fetchall()
    return render_template('viewresult.html',data=details)



@app.route('/upload1/<filename>')
def send_image1(filename):
    return send_from_directory("result", filename)

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)


@app.route('/uviewresult')
def uviewresult():
    reqid=request.args.get('Id')
    cursor.execute("select * from result where reqid='"+reqid+"'")
    details=cursor.fetchall()
    return render_template('uviewresult.html',data=details)



if __name__ == '__main__':
    app.run(debug=True)