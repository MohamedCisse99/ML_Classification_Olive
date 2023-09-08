from flask import Flask, render_template, request
from flask import send_from_directory
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
app = Flask(__name__)

new_model=load_model(os.path.join('D:\ALL_MY_PROJECT\Amamdou_ML_Classification_Olive\models','oliviervarietymodel.h5'))

@app.route('/')
def index() :
    return render_template('platform.html')

@app.route('/<path:filename>')
def serve_video(filename):
    return send_from_directory('D:\ALL_MY_PROJECT\Amamdou_ML_Classification_Olive/images', filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400
    save_path = 'D:\ALL_MY_PROJECT\Amamdou_ML_Classification_Olive/images/image.jpg'  # Specify the desired local path to save the image
    file.save(save_path)
    d='D:\ALL_MY_PROJECT\Amamdou_ML_Classification_Olive\dataset_clean'
    classes_names=[]
    for img_class in os.listdir(d):
        classes_names.append(img_class)
        img=cv2.imread('D:\ALL_MY_PROJECT\Amamdou_ML_Classification_Olive/images/image.jpg')
    resize=tf.image.resize(img, (256,256))
    y_pred=new_model.predict(np.expand_dims(resize/255,0))
    y_pred_classes=[np.argmax(el) for el in y_pred]
    return render_template('platform.html', pred='Le type de variété auquel appartient ce noyau d\'olivier est : {}'.format(classes_names[y_pred_classes[0]%57]))
if __name__ == '__main__':
    app.run(debug=True)