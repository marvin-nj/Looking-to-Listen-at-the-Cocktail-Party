import sys
sys.path.append ('./model/model')
sys.path.append ('./model/utils')
from keras.models import load_model
from option import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import tensorflow as tf
from loss import audio_discriminate_loss2 as audio_loss



#parameters
people = 2
num_gpu=1

#path
model_path = './saved_AV_models/AVmodel-2p-012-0.49488.h5'
result_path = './predict/'
os.makedirs(result_path,exist_ok=True)

database = './data/AV_model_database/mix/'
face_emb = './model/face_embedding/face1022_emb/'
print('Initialing Parameters......')

#loading data
print('Loading data ......')
test_file = []
with open('./data/AVdataset_val.txt','r') as f:
    test_file = f.readlines()[:1]

def get_data_name(line,people=people,database=database,face_emb=face_emb):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(people):
        single_idxs.append(names[i])
    file_path = database + mix_str
    mix = np.load(file_path)[:298,]
    print(mix.shape)
    face_embs = np.zeros((1,75,1,1792,people))
    for i in range(people):
        face_embs[0,:,:,:,i] = np.load(face_emb+"%05s_face_emb.npy"%single_idxs[i])
    
    print(face_embs.shape)

    return mix,single_idxs,face_embs

people_num = 2
epochs = 20
initial_epoch = 0
batch_size = 4
gamma_loss = 0.1
beta_loss = gamma_loss * 2
loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)

#result predict
print("load model start...")    
av_model = load_model(model_path,custom_objects={'loss_func':loss,'tf':tf})
print("load model ok...")
if num_gpu>1:
    parallel = ModelMGPU(av_model,num_gpu)
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database,face_emb)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = parallel.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(len(cRMs)):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fast_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)

if num_gpu<=1:
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database,face_emb)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = av_model.predict([mix_ex,face_emb])
        print("cRMs: ",cRMs.shape,len(cRMs))
        cRMs = cRMs[0]
        print("cRMs: ",cRMs.shape,len(cRMs))
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(people):
            cRM =cRMs[:,:,:,i]
            print("cRM",cRM.shape)
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            print("mix: ", mix.shape,"F: ",F.shape)
            T = utils.fast_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)
print("all OK")
