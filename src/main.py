import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.io.wavfile as wav
import scipy.signal as signal
import random

import librosa

from pydub import AudioSegment
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import subprocess

##################################################################################################
##################################################################################################
# Global Constant
AUTOTUNE = tf.data.experimental.AUTOTUNE
commands = ['singing', 'talking']

# SINGING_PATH:
path_sing = {
    "Country": '../public/singing/country',
    "Dance_electronic": '../public/singing/dance_electronic',
    "Duet": '../public/singing/Duet',
    "Gura": '../public/singing/gura',  # gura is a vtuber, just my personal interest
    "Rap": '../public/singing/Rap',
    "Rock": '../public/singing/rock',
    "Traditional": '../public/singing/Traditional'
}

# SPEECH_PATH:
path_speech = {
    "Food_review": '../public/talking/food_review',
    "Gura": '../public/talking/gura',
    "Movie_review": '../public/talking/movie_review',
    "Techonology_review":'../public/talking/technology_review',
}

# Saving model path
MODEL_OUT_DIR = 'model/'

# Additional model will be add here
SAVED_MODEL_DIR = [
    # train with 400 sing and 900 speech, achieve 75% - 80% on country and food_review
    'model/my_model',
]

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


##################################################################################################
# Preprocess data Function

# load_data: receive a directory path, return array of all files path inside it as tensor
def load_data(file_paths):
    print(file_paths)
    data = []
    for path in file_paths:
        data = data + tf.io.gfile.glob(str(path) + '/*')
    return data


# split_data: Divide an array into train, validation and test data set
def split_data(data, train=0.8, val=0.1, test=0.1):
    data_len = len(data)
    train_len, val_len, test_len = \
        [int(data_len * train), int(data_len * val), int(data_len * test)]

    return data[:train_len], \
           data[train_len:train_len + val_len], \
           data[val_len + train_len: data_len]


# Get Label and waveform information
def get_waveform_and_label(file_path):
    # Label for each WAV file base on file path
    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-3]

    # decode_audio: return waveform information as audio and sampling rate as _
    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


# get_spectrogram: return spectrogram as binary information from input waveform
def get_spectrogram(waveform, FRAME_LENGTH=255, FRAME_STEP=128, FREQUENCY=240000):
    zero_padding = tf.zeros([FREQUENCY] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP  # overlapping frame of 255 - 128
    )
    spectrogram = tf.abs(spectrogram)
    return spectrogram


# get_spectrogram_and_label_id: return spectrogram and corresponding label
def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


'''
preprocess_dataset: 
input array of file path ( from load data )
audio file will get preprocess into spectrogram information and label id 
output is a dataset type that contains spectrogram info and label
'''


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


##################################################################################################
# Build model function

'''
build_mode: 
 input_shape: shape of audio ( typically in spectrogram info ) 
 spectrogram_ds: taken from preprocess_dataset: serve as basis for normalization incoming data 
 num_labels: number of classification ( e.g: ['singing', 'talking'] )
return a keras machine learning model
'''


def build_model(input_shape, spectrogram_ds, num_labels):
    # normalization: normalize incoming data based on original spectrogram dataset to build model
    def normalization(spectrogram_ds):
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
        return norm_layer

    # model: default model to process audio, taken from google guide
    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        normalization(spectrogram_ds),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    return model


# compile model: defines the optimizer, loss and metrics,
# mainly use when we want the model to learn
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


# train mode: train the model and validate it
# EPOCHS: number of run through the data
# history: contains various info after training model

def train_model(model, train_ds, val_ds):
    EPOCHS = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    return history


# Evaluate with test set performance
def evaluate(model, test_ds):
    test_audio = []
    test_labels = []
    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')


# Saved Model
def save_model(model, path, OVERWRITE=False):
    if os.path.exists("path") & OVERWRITE == False:
        print("theere exists a model in this path")
        return

    model.save(path)


# Load Model, model is index of SAVED_MODEL or path
def load_model(path="", INDEX=0):
    if path != "":
        return tf.keras.models.load_model(path)

    return tf.keras.models.load_model(SAVED_MODEL_DIR[INDEX])


#run_training_mode: return a model that has been build, compile and train
def run_training_model(path_array):
    ##Extract the audio files into a list
    data = load_data(path_array)

    ##Split the files into training, validation and test sets using 80:10:10 ratio
    train_files, val_files, test_files = split_data(data)
    print(data)
    # Process training data set
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    # Batch the training
    batch_size = 64
    train_ds = spectrogram_ds
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Normalize and Build CNN layer
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
        print('Input shape:', input_shape)
        num_labels = len(commands)

    model = build_model(input_shape, spectrogram_ds, num_labels)
    compile_model(model)
    train_model(model, train_ds, val_ds)
    evaluate(model, test_ds)

    return model


# Evaluate the training model with test path
def evaluate_model(model, path_array):
    test_ds = load_data(path_array)
    test_ds = preprocess_dataset(test_ds)
    evaluate(model, test_ds)

##################################################################################################
# Utilities function

# process_mp3: Convert mp3 to wav
def process_mp3(path, out):
    subprocess.call(['ffmpeg', '-i', '/input/file.mp3',
                     '/output/file.wav'])


# Process_wavFile: Split audio file into smaller audio file
def process_wavFile(path, out_path):
    filelist = []
    for root, dir, files in os.walk(path):
        for file in files:
            filelist.append(root + '/' + file)

    for i, file in enumerate(filelist):
        song = AudioSegment.from_wav(file)
        song = song.set_channels(1)  # Channel 1 = mono, 2 = stereo
        time = 0
        index = 0
        while time < len(song):
            temp = song[time: min(time + 5000, len(song))]
            temp.export(out_path + '/file' + str(index) + '.wav', format='wav')
            index += 1
            time += 5000


##########################################################################################
# Graphing
# Use this when need to visualize

# graph_waveform: input is a tensorflow type dataset
# iterate through the dataset and graph its waveform
def graph_waveforms(waveform_ds):
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)
    plt.show()


# check model efficiency
def graph_efficiency(history):
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()


## Graphing spectrogram:
## TODO: Fix this
# for waveform, label in waveform_ds.take(1):
#     label = label.numpy().decode('utf-8')
#     spectrogram = get_spectrogram(waveform)
#
# def plot_spectrogram_and_waveform(spectrogram, ax):
#     print(len(spectrogram))
#     log_spec = np.log(spectrogram.T)          #FIXME: This would cause reverse label
#     height = log_spec.shape[0]
#     X = np.arange(220500, step=height + 1)
#     Y = range(height)
#     ax.pcolormesh(X, Y, log_spec)
#
#
# fig, axes = plt.subplots(2, figsize=(12, 8))
# timescale = np.arange(waveform.shape[0])
# axes[0].plot(timescale, waveform.numpy())
# axes[0].set_title('Waveform')
# axes[0].set_xlim([0, 220500])
# plot_spectrogram_and_waveform(spectrogram.numpy(), axes[1])
# axes[1].set_title('Spectrogram')
# plt.show()

##########################################################################################
# Main body
def main():
    model = run_training_model([path_sing["Gura"], path_speech["Gura"]])
    evaluate_model(model,
                [
                    path_sing["Country"],
                    path_sing["Dance_electronic"],
                    path_speech["Food_review"],
                    path_speech["Movie_review"],
                    path_speech["technology_review"]
                ])

    #Note: When I use Gura: which is song and speech by the same person, i got good accuracy
    # When I use speech and song by difference person to train, the accuracy become really low

main()
