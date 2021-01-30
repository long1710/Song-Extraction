'''
outline:
- extract song path
- load in song
- cut song down to 5 seconds segment ( if not 5 second )
- extract desire model
- load in model ( if not already exist )
- run the model
- run the prediction
'''

from zipfile import ZipFile
from pydub import AudioSegment
import cli
import tensorflow as tf
import requests
import json
import os
import subprocess
import numpy as np
import shutil

with open('config_file.json', 'r') as myfile:
    data = myfile.read()

obj = json.loads(data)

#get_model: Retrieve model from a public S3 storage
def get_model(model):
    print('begin extracting model... \n \n ')

    url = obj['model'][model]

    dest_dir = "model/" + model + ".zip"
    model_dir = "model/content/" + model

    #Store model in a folder name model
    if not os.path.exists("model"):
        os.makedirs("model")

    #Download model from AWS S3 storage if not presence
    if not os.path.exists(model_dir):
        r = requests.get(url)
        with open(dest_dir, 'wb') as f:
            f.write(r.content)

        with ZipFile(dest_dir, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall('model')

    #Load the model download from web
    #TODO: delete zip file to save space after unzip
    model = tf.keras.models.load_model(model_dir)

    print('finish loading model \n')
    os.remove(dest_dir)
    return tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#get_song: Retrieve song path and process into picture
def get_song(song_path):
    print('begin preprocessing song data... \n')
    if not os.path.exists("spectrogram"):
        os.makedirs("spectrogram")

    if not os.path.exists("spectrogram/singing"):
        os.makedirs("spectrogram/singing")

    song = AudioSegment.from_wav(song_path)

    if(len(song) > 5000):
        ans = input('detect song size over 5 seconds, do you want to split into multiple 5 seconds segment ? Y/N ')
        if(ans == 'Y' or ans == 'y'):
            preprocessing_longvideo(song)
            video_to_img('wav')
            return
        else:
            print('Cannot process audio with bigger or smaller than 5 seconds, terminating...')
            raise ValueError

    #Bash script to convert wav file to spectrogram
    subprocess.run(["ffmpeg", "-i", song_path, "-lavfi", "showspectrumpic=s=1024x512:legend=disabled", "spectrogram/singing/test.jpg"])
    print('finish preprocessing song data \n')

#get_result: Load pre-train model and predict
def get_result(spectrogram_path, model):
    print('begin prediction... \n')
    dataset = tf.keras.preprocessing.image_dataset_from_directory("spectrogram")
    return model.predict(dataset)

#result_toString: pretty print of result data
def result_tostring(result):
    singing = 0
    talking = 0
    for prediction in result:
        if(np.argmax(prediction) == 0):
            singing += 1
        if(np.argmax(prediction) == 1):
            talking += 1
    print('###################')
    print('singing: ' + str(singing) + ' files')
    print('talking: ' + str(talking) + ' files')
    print('###################')

#preprocessing_longvideo: break down video with size > 5
def preprocessing_longvideo(song):
    song = song.set_channels(1)  # Channel 1 = mono, 2 = stereo
    time = 0
    index = 0
    while time + 5000 < len(song):
        temp = song[time: time + 5000]
        file_name = 'wav/file' + str(index) + '.wav'
        img_name = 'spectrogram/singing/file' + str(index) + '.wav'
        temp.export(file_name, format='wav')
        index += 1
        time += 1000 # 4s window overlap

def video_to_img(path):
    for root, dir, files in os.walk(path):
        i = 0
        for file in files:
            temp = 'wav/' + file
            subprocess.run(["ffmpeg", "-i", temp, "-lavfi", "showspectrumpic=s=1024x512:legend=disabled",
                            "spectrogram/singing/test" + str(i) + ".jpg"])
            i += 1

#clean up the wav and jpg file after prediction
def clean_up():
    shutil.rmtree('wav')
    shutil.rmtree('spectrogram')

#prediction: all process together
def prediction():
    package_installation()
    args = cli.get_args()
    model = get_model(args.m)
    get_song(args.p)
    result = get_result("spectrogram", model)
    result_tostring(result)
    clean_up()

def package_installation():
    print('The following package will be installed into your folder: ')

if __name__ == '__main__':
    prediction()

