## Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation

------

#### The project is an audiovisual model reproduced by the contents of the paper Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation.

> Ephrat A, Mosseri I, Lang O, et al. Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation[J]. arXiv preprint arXiv:1804.03619, 2018.

Requirement

- Python3.7
- TensorFlow 2.0.0
- Keras 2.3.1
- librosa 0.7.0
- youtube-dl(https://github.com/ytdl-org/youtube-dl)(**Any version**)
- ffmpeg(https://www.ffmpeg.org/)（**Any version**)
- sox

To install requirements:

```
pip install -r requirements.txt
```

You can install ffmpeg and sox using homebrew:

```
brew install ffmpeg
brew install sox
```

------

### Pretreatment

#### Video Data

1. Download the dataset from [here](https://looking-to-listen.github.io/avspeech/download.html) and place files in data/csv.
2. First use this command to download the YouTube video and use ffmpeg to capture the 3 second video as 75 images.

```
python3 video_download.py
```

1. Then use mtcnn to get the image bounding box of the face, and then use the CSV x, y to locate the face center point.

```
pip install mtcnn
python3 face_detected.py
python3 check_vaild_face.py
```

#### Audio Data

1. For the audio section, use the YouTube download tool to download the audio, then set the sample rate to 16000 via the librosa library. Finally, the audio data is normalized.

```
python3 audio_downloads.py
python3 audio_norm.py # audio_data normalized
```

1. Pre-processing audio data, including stft, Power-law, blending, generating complex masks, etc....

```
python3 audio_data.py
```

#### Face embedding Feature

- Here we use Google's FaceNet method to map face images to high-dimensional Euclidean space. In this project, we use David Sandberg's open source FaceNet preprocessing model "[20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)". Then use the TensorFlow_to_Keras script in this project to convert.（**Model/face_embedding/**）

> Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.

Change the path `tf_model_dir` in Tensorflow_to_Keras.py

```
python3 Tensorflow_to_Keras.py
python3 face_emb.py
```

------

1. Create AVdataset_train.txt and AVdataset_val.txt

```
python3 AV_data_log.py
```

### Training

- **Support continuous training after interrupt training**
- **Support multi-GPU multi-process training.**
- **According to the description in the paper, set the following parameters:**

```
people_num = 2 # How many people you want to separate?
epochs = 100
initial_epoch = 0
batch_size = 1 # 2,4 need to GPU
gamma_loss = 0.1
beta_loss = gamma_loss * 2
```

- **Then use the script train.py to train**

------

2021.2.24 update: 

1、修复测试及相关文件，能正常运行 

2、建议用conda安装tensorflow环境，否则会找不到GPU 

3、上传一个训练模型，待...



Part of the code reference this github https://github.com/bill9800/speech_separation
