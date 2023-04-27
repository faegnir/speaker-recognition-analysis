import librosa
from flask import *
import pyaudio
import wave
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import os
from glob import glob
import threading
import tensorflow as tf
import pvleopard as pv
import config

FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
SAMPLING_RATE = 16000
BATCH_SIZE = 128


app = Flask(__name__)

@app.route('/')
def main():
	return render_template("index.html")



@app.route('/get_last_recorded_filename')
def get_last_recorded_filename():
    file_pattern = "record*.wav" # change this to match your file naming convention
    files = glob(file_pattern)
    if not files:
        return ""
    files = sorted(files, key=lambda f: int(os.path.splitext(f)[0][len("record"):]), reverse=True)
    return files[0]


@app.route('/mic_record/<action>')
def mic_record(action):
    is_recording = False
    global frames,stream
    print(action)
    if action == 'start':
        if not is_recording:
            
            p = pyaudio.PyAudio()

            # starts recording
            stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLING_RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER
            )

            print("start recording...")
            is_recording = True
            frames = []

            # start a new thread to continuously read frames from the stream and append to the frames list
            def read_frames():
                while is_recording:
                    if stream.is_stopped():
                        break
                    data = stream.read(FRAMES_PER_BUFFER)
                    frames.append(data)

            threading.Thread(target=read_frames).start()

    elif action == 'stop':
            if is_recording:
                print("recording stopped")
                is_recording = False
            p = pyaudio.PyAudio()

            # stop the stream and terminate the PyAudio object
            stream.stop_stream()
            stream.close()
            p.terminate()

            # write the frames to a WAV file
            recfilename = "record" + str(len(os.listdir())-13) + ".wav"
            wf = wave.open(recfilename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLING_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

    else:
        print("not recording")
    return ""

@app.route('/prediction', methods = ['POST'])
def success():
    if request.method == 'POST':
        data = request.form['audio_name']
        waveform_html, spectrogram_html = generate_plots(data)
        prediction = predict_single(data)
        leopard = pv.create(access_key=config.access_key)
        transcript, words = leopard.process_file(data)

        return render_template("prediction.html", name=data, waveform_html=waveform_html, spectrogram_html=spectrogram_html,
                               prediction=prediction,transcript=transcript,words=len(words))


def generate_plots(filename):
    # Load audio file
    y, sr = librosa.load(filename)

    # Plot waveform
    waveform_fig = go.Figure()
    waveform_fig.add_trace(go.Scatter(x=np.arange(len(y))/sr, y=y))
    waveform_fig.update_layout(title="Waveform", xaxis_title="Time (seconds)", yaxis_title="Amplitude")
    waveform_html = pio.to_html(waveform_fig, full_html=False,default_width="380px",default_height="300px")

    # Plot spectrogram
    D = librosa.stft(y)
    DB = librosa.amplitude_to_db(abs(D))
    spectrogram_fig = go.Figure()
    spectrogram_fig.add_trace(go.Heatmap(x=np.arange(DB.shape[1])/sr, y=np.arange(DB.shape[0]), z=DB))
    spectrogram_fig.update_layout(title="Spectrogram", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
    spectrogram_html = pio.to_html(spectrogram_fig, full_html=False,default_width="380px",default_height="300px")

    return waveform_html, spectrogram_html

class_names = config.cn

model = tf.keras.models.load_model('./models/students/v1/model.h5')

def paths_to_dataset(audio_paths):
	#Constructs a dataset of audios and labels.
	path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
	audio_ds = path_ds.map(lambda x: path_to_audio(x))
	return tf.data.Dataset.zip((audio_ds))


def path_to_audio(path):
	#Reads and decodes an audio file.
	audio = tf.io.read_file(path)
	audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
	return audio


def audio_to_fft(audio):
	audio = tf.squeeze(audio, axis=-1)
	fft = tf.signal.fft(
		tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
	)
	fft = tf.expand_dims(fft, axis=-1)
	return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def predict_single(audio_path):
    # Create a dataset with the single audio file
    test_ds = paths_to_dataset([audio_path])
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    for audio in test_ds:
        ffts = audio_to_fft(audio)

    # Get the predicted class index
    y_pred = model.predict(ffts)
    y_pred_index = np.argmax(y_pred, axis=-1)[0]

    # Print the prediction
    print("\033[31m[*]\033[0m Model's prediction:\33[92m {}\33[0m".format(class_names[y_pred_index]))
    return class_names[y_pred_index]


if __name__ == '__main__':
	app.run(debug=True)

