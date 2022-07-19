from pickle import TRUE
from unicodedata import decimal
from graphviz import render
from tensorflow.keras.initializers import HeUniform, GlorotNormal
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.regularizers import L2
import os
from itertools import combinations, product
from random import sample, shuffle, seed

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.math import reduce_sum, square, reduce_mean, maximum, sqrt
from tensorflow import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout, BatchNormalization
from keras import layers
from keras import backend as K
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.applications import resnet
from keras.callbacks import TensorBoard

from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.applications import resnet
from tensorflow.keras import metrics
from datetime import datetime
import numpy as np
import cv2
from math import ceil
np.random.seed(42)
random.set_seed(42)

img_h, img_w = 155, 220


OPT_TH=0.7881761 #0.6903344


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

################Model###########################
input_shape=(155,220,1)
model = Sequential()



model.add(Conv2D(96, (11, 11), activation="relu", input_shape=input_shape, padding='same', kernel_initializer=HeUniform(42), kernel_regularizer=L2(0.01)))
model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(BatchNormalization())



model.add( Conv2D(256, (5, 5), activation="relu", padding='same', kernel_initializer=HeUniform(42), kernel_regularizer=L2(0.01) ))
model.add( MaxPooling2D(pool_size=(3,3),strides=(2, 2)) )
model.add(Dropout(0.4))
model.add(BatchNormalization())


model.add( Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer=HeUniform(42), kernel_regularizer=L2(0.01) ))
model.add( Conv2D(256, (3, 3), activation="relu", padding='same', kernel_initializer=HeUniform(42), kernel_regularizer=L2(0.01) ))
model.add( MaxPooling2D(pool_size=(3,3),strides=(2,2)) )
model.add(Dropout(0.4))
model.add(BatchNormalization())



model.add( GlobalAveragePooling2D() )
model.add( Dense(128) )#16
model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

################Model END###########################



def image_preprocessing(signature):
    signature = cv2.imread(signature)
    noiseless_image_bw = cv2.fastNlMeansDenoising(signature, None, 20, 7, 21)
    blurred=cv2.medianBlur(noiseless_image_bw,3)
    
    no_bg_gray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret3,thres = cv2.threshold(no_bg_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    resized=cv2.resize(thres,(220,155),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    normalize=1-(resized/255)

    signature_expanded = normalize[np.newaxis, :, :]
    signature_expanded = signature_expanded[:, :, :,np.newaxis]
    return np.array(signature_expanded)


class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):        
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)        
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=input_shape)
positive_input = layers.Input(name="positive", shape=input_shape)
negative_input = layers.Input(name="negative", shape=input_shape)

distances = DistanceLayer()(
    model(anchor_input),
    model(positive_input),
    model(negative_input),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)        
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]




siamese_model = SiameseModel(siamese_network)
siamese_model.built=True
siamese_model.load_weights("cm.h5")
siamese_model.compile(optimizer=optimizers.Adam(0.00001))


def remove_temp():
    for dir in os.listdir("D:\\Desktop\\project\\tempfiles"):
        if 'fooo' in dir:
            os.remove("D:\\Desktop\\project\\tempfiles\\"+dir)

def image_preprocessing_ver(signature):
        no_bg_gray=cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
        noiseless_image_bw = cv2.fastNlMeansDenoising(no_bg_gray, None, 20, 7, 21)
        blurred=cv2.medianBlur(noiseless_image_bw,3)
        ret3,thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        resized=cv2.resize(thres,(220,155),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        eroded=cv2.erode(resized, (3,3),iterations=1)
        normalize=1-(eroded/255)

        signature_expanded = normalize[np.newaxis, :, :]
        signature_expanded = signature_expanded[:, :, :,np.newaxis]
        return np.array(signature_expanded)

def test_protocol(org1, org2, org3, org4, test, dob_date):

    if (datetime.today()-datetime.strptime(dob_date, '%Y-%m-%d')).days>23725:
        adaptive_thresh=OPT_TH-0.1903344
    else:
        adaptive_thresh=OPT_TH

    emb_org1=model(image_preprocessing_ver(org1)).numpy()[0]
    emb_org2=model(image_preprocessing_ver(org2)).numpy()[0]
    emb_org3=model(image_preprocessing_ver(org3)).numpy()[0]
    emb_org4=model(image_preprocessing_ver(org4)).numpy()[0]
    emb_test=model(image_preprocessing_ver(test)).numpy()[0]
    
    sim_1t=cosine_similarity(emb_org1, emb_test).numpy()
    sim_2t=cosine_similarity(emb_org2, emb_test).numpy()
    sim_3t=cosine_similarity(emb_org3, emb_test).numpy()
    sim_4t=cosine_similarity(emb_org4, emb_test).numpy()
    for sim in [sim_3t,sim_1t, sim_2t, sim_4t]:
        if sim >=adaptive_thresh:
            return 'Signature is genuine'
    return 'Signature is forged'

from flask import Flask, request, render_template,redirect
from keras.metrics import cosine_similarity
from time import time
import sqlite3
sqliteConnection = sqlite3.connect('signatures.db', check_same_thread=False)
cursor = sqliteConnection.cursor()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/', methods=['GET','POST'])
def index_post():
    usr_name = request.form['usr_name']
    passwd=request.form['passwd']
    org_details=cursor.execute('SELECT * FROM ORGANIZATION WHERE USERNAME=(?)', (usr_name,)).fetchall()
    try:
        if passwd==org_details[0][1]:
            return redirect('/verify')
        else:
            return render_template('error.html', error_msg='Incorrect username or passwrod', home_url='')
    except:
        return render_template('error.html', error_msg='User not exists', home_url='')


@app.route('/orgReg')
def orgReg():
    return render_template('orgReg.html')
@app.route('/orgReg', methods=['GET','POST'])
def orgReg_post():
    org_name = request.form['org_name']
    org_pass=request.form['org_pass']
    try:
        cursor.execute("INSERT INTO ORGANIZATION VALUES (?, ?)", (org_name, org_pass))
    except:
        return render_template('error.html', error_msg='Organization already exists.', home_url='')
    return redirect('/')



@app.route('/verify')
def verify():
    return render_template('verify.html', decision='')
@app.route('/verify', methods=['GET','POST'])
def verify_post():
    cus_id = request.form['cus_id']
    mob = request.form['mob']
    mob=str(mob)
    testf=request.files['tf']

    file_name=str(int(time()))
    remove_temp()
    testf.save("D:\\Desktop\\project\\tempfiles\\"+file_name+'fooo_1.png')

    try:
        if len(cus_id)!=0:
            cus_info=cursor.execute('SELECT * FROM SIGN_INFO WHERE CUSTOMER_ID = (?)', (cus_id,)).fetchall()[0]
        elif len(mob)!=0:
            cus_info=cursor.execute('SELECT * FROM SIGN_INFO WHERE MOBILE = (?)', (mob,)).fetchall()[0]
        else:
            return render_template('error.html', error_msg='Details are insufficient', home_url='verify')
    except:
        return render_template('error.html', error_msg='Customer not exists. Try registering beforing verification or try Oneshot method.', home_url='verify')
    sign_file=cus_info[3]
    cus_dob=cus_info[2]

    org1=cv2.imread("D:\\Desktop\\project\\static\\sign_db\\"+sign_file+'\\'+sign_file+'_1.png')
    org2=cv2.imread("D:\\Desktop\\project\\static\\sign_db\\"+sign_file+'\\'+sign_file+'_2.png')
    org3=cv2.imread("D:\\Desktop\\project\\static\\sign_db\\"+sign_file+'\\'+sign_file+'_3.png')
    org4=cv2.imread("D:\\Desktop\\project\\static\\sign_db\\"+sign_file+'\\'+sign_file+'_3.png')
    test_img=cv2.imread("D:\\Desktop\\project\\tempfiles\\"+file_name+'fooo_1.png')
    decision=test_protocol(org1, org2, org3, org4, test_img, cus_dob)
    
    return render_template('verify.html', decision=decision)

@app.route('/newreg')
def newreg():
    return render_template('newReg.html')



@app.route('/newreg', methods=['GET','POST'])
def newreg_post():
    try:
        file_name=str(int(time()))
        cus_name = request.form['cus_name']
        mob=request.form['mob']
        dob=request.form['dob']

        os.makedirs("D:\\Desktop\\project\\static\\sign_db\\"+file_name)

        f1 = request.files['f1']
        f1.save("D:\\Desktop\\project\\static\\sign_db\\"+file_name+'\\'+file_name+'_1.png')

        f2 = request.files['f2']
        f2.save("D:\\Desktop\\project\\static\\sign_db\\"+file_name+'\\'+file_name+'_2.png')

        f3 = request.files['f3']
        f3.save("D:\\Desktop\\project\\static\\sign_db\\"+file_name+'\\'+file_name+'_3.png')

        f4 = request.files['f4']
        f4.save("D:\\Desktop\\project\\static\\sign_db\\"+file_name+'\\'+file_name+'_4.png')

        cursor.execute('INSERT INTO SIGN_INFO(NAME, DOB, PATH, MOBILE) VALUES (?, ?, ?, ?)', (cus_name,dob, file_name, mob))
        sqliteConnection.commit()
    except:
        return render_template('error.html', error_msg='Exception occured while creating account. Account may already exists !!!!', home_url='verify')

    return redirect('/verify')
@app.route('/oneshot')
def oneshot():
    return render_template('oneshot.html', decision='')
@app.route('/oneshot', methods=['GET','POST'])
def oneshot_submit():
    org = request.files['f1']
    test = request.files['f2']
    remove_temp()
    file_name1=str(int(time()))
    file_name2=str(int(time()))
    org.save("D:\\Desktop\\project\\tempfiles\\"+file_name1+'_ofooo.png')
    test.save("D:\\Desktop\\project\\tempfiles\\"+file_name2+'_tfooo.png')

    orginal_sign=cv2.imread("D:\\Desktop\\project\\tempfiles\\"+file_name1+'_ofooo.png')
    test_sign=cv2.imread("D:\\Desktop\\project\\tempfiles\\"+file_name1+'_tfooo.png')

    orginal_sign_arr=image_preprocessing_ver(orginal_sign)
    test_sign_arr=image_preprocessing_ver(test_sign)
    #print(test_sign_arr)
    orginal_sign_emb=model(orginal_sign_arr)
    test_sign_emb=model(test_sign_arr)

    sim_score=cosine_similarity(orginal_sign_emb, test_sign_emb)
    if sim_score>=OPT_TH:
        decision='Genuine signature'
    else:
        decision='Forged signature'
    return render_template('oneshot.html', decision=decision)


@app.route('/allCus')
def allCus():
    cuss=cursor.execute('SELECT * FROM SIGN_INFO').fetchall()
    return render_template('all_cus.html', cuss=cuss)


app.run(debug=TRUE)