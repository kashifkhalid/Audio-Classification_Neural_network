
# coding: utf-8

# In[21]:



# Import libraries# Import 
print('running0')
import os
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from six.moves import cPickle as pickle
from six.moves import range
import librosa
import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import logfbank
import pdb
import re
import tensorflow as tf
from pydub import AudioSegment
#"uncomment this to visualize the results"
#import matplotlib.pyplot as plt 
import librosa.display
import os, argparse
import json


# In[22]:


def delete_files():
    folder = 'C:/Users/kkhalid/project_kk/temp2/'
    
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
                print(e)
    return           


# In[23]:


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


# In[24]:


def trim(file_name: str) -> tuple:
    
    folder = 'C:/Users/kkhalid/project_kk/temp2/'
    
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
                print(e)
    filepath_finder=os.path.abspath(file_name)            
    pathsp=filepath_finder.split('\\')
    newfilename=pathsp[-1].split('.')
    y, sr = librosa.load(filepath_finder)
    dur=librosa.get_duration(y=y, sr=sr)
    dur=dur/10
    inc=0
    for i in range(int(dur)):
        
        y, sr = librosa.load(file_name,offset=0.0+inc,duration=10.0)
        librosa.output.write_wav('C:/Users/kkhalid/project_kk/temp2/'+newfilename[0]+str(inc)+'.wav',y,sr)
        inc=inc+10
        
    return
    


# In[25]:


def extract_feature(file_name: str) -> tuple:
    """
    Extracts 193 chromatographic features from sound file. 
    including: MFCC's, Chroma_StFt, Melspectrogram, Spectral Contrast, and Tonnetz
    NOTE: this extraction technique changes the time series nature of the data
    """
    #inc=0
    #y, sr = librosa.load(file_name)
    #dur=librosa.get_duration(y=y, sr=sr)
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# In[26]:


#start_time = timer()
#a,b,c,d,e = extract_feature('C:/Users/kkhalid/project_kk/1980-oct27-3.wav')
#end_time = timer()
#print('time to extract features from one file: {:.3f}sec'.format((end_time-start_time)/60))


# In[27]:


#argument_filename=sys.argv[1]
#print(argument_filename)
#print(filepath)


# In[28]:


start_time = timer()

argument_filename=sys.argv[1]
#print(argument_filename)
#print(filepath)
#argument_filename=input('enter audio file path with format:')
#print(str(filepath))
#pathsp=filepath.split('/')
#newfilename=pathsp[-1].split('.')
#print(newfilename[0])
argument_filename=argument_filename+'.wav'
print(argument_filename)
trim(argument_filename)
#pdb.set_trace()
mfcc_data = []
exception_count = 0
loop_count=0
inc=0

mypath='C:/Users/kkhalid/project_kk/temp2/'
files = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
len_files=len(files)

for i in range(len_files):

    mfccs,chroma,mel,contrast,tonnetz = extract_feature(files[i])
    path,filename=os.path.split(files[i])

    features = np.empty((0,193))
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    mfcc_data.append([filename,features])
#pdb.set_trace()
#print(features)
end_time = timer()
print(print("time taken: {0} minutes {1:.1f} seconds".format((end_time - start_time)//60, (end_time - start_time)%60)))


# In[29]:


cols=["filename","features"]
mfcc_pd=pd.DataFrame(data=mfcc_data,columns=cols)
#mfcc_pd.tail(5)


# In[30]:


ll = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
mfcc_pd['sample'] = pd.Series(ll, index=mfcc_pd.index)
#del mfcc_pd['features']


# In[31]:


s = list(mfcc_pd['sample'])

s = pd.DataFrame(s)
#print(s)

data_cols = s.columns
#s['filename'] = mfcc_pd['filename']
#print('working dataframe\'s shape:', s.shape)


# In[32]:


test=s[:]
scaler1 = sk.preprocessing.StandardScaler().fit(test.loc[:,data_cols])
test.loc[:,data_cols] = scaler1.transform(test.loc[:,data_cols])
#print(test.loc[:,data_cols])
#print(data_cols)


# In[33]:


#loading frozen model

    # We use our "load_graph" function
graph = load_graph('C:/Users/kkhalid/frozen_model.pb')

#for op in graph.get_operations():
    #print(op.name)
    
x = graph.get_tensor_by_name('prefix/input_data:0')
y = graph.get_tensor_by_name('prefix/op_to_restore:0')  

with tf.Session(graph=graph) as sess:
    check_size=3000
    feed_dict={x : test.loc[0:check_size-1,data_cols]}
    y_out = sess.run(y, feed_dict)
    #print(y_out)
    #print(len(y_out))


# In[34]:


#loading saved models
def saved_model(model_name,data):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        new_saver = tf.train.import_meta_graph('CNN_model_trained.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        #for op in graph.get_operations():
         #   print(op.name)
        
        graph = tf.get_default_graph()
        #op = sess.graph.get_operations()
        #[m.values() for m in op][:]
        input_data= graph.get_tensor_by_name("input_data:0")
        #data = tf.placeholder(tf.float32, shape=[None, 193])
        check_size=3000
        out = graph.get_tensor_by_name("op_to_restore:0")
        feed_dict={input_data : test.loc[0:check_size-1,data_cols]}
        y_out = sess.run(out, feed_dict)

    #print(len(y_out))
    #print(y_out)


# In[35]:


filepath_finder=os.path.abspath(argument_filename)            
X, sample_rate = librosa.load(filepath_finder)
dur=librosa.get_duration(y=X, sr=sample_rate)
#xaxis=np.empty()
durations=[]
#plt.figure()
count=0
inc=0
time=[]
tag=[]
for out in range(len(y_out)):
        while count <= dur:
            begin=count
            count=count+10
            end=count
            durations.append([begin,end])
            #xaxis=sum(durations,[])
            xaxis=durations
        pred=y_out[out]
        max_argu=np.argmax(pred)
        
        #for a in range(len(pred)):
            #value=np.argmax(pred)
        if max_argu==0:
                time_instant=xaxis[inc]
                classification='music'
                time.append([time_instant])
                tag.append([classification])
                #print(xaxis[inc])
                #print('music')
        elif max_argu==1:
            time_instant=xaxis[inc]
            classification='noise'
            time.append([time_instant])
            tag.append([classification])
            #print(xaxis[inc])
            #print('noise')
        elif max_argu==2:
                time_instant=xaxis[inc]
                classification='silence'
                time.append([time_instant])
                tag.append([classification])
                #print(xaxis[inc])
                #print('silence')
        else:
                time_instant=xaxis[inc]
                classification='speech'
                time.append([time_instant])
                tag.append([classification])
                #print(xaxis[inc])
                #print('speech')
        #xaxis.append([xvalue])  
        inc=inc+1
            
print(dur)
time=sum(time,[])
tag=sum(tag,[])
print(tag)
print(time)
#dur=int(dur)
#dur=np.linspace(0,dur,len(y_out))
#result=sum(xaxis,[])
#print(np.array(result))
#plt.figure(1)
#plt.plot(dur,result)
#plt.xlabel('audio length in seconds')
#plt.ylabel('class')
#plt.figure(2)
#librosa.display.waveplot(X, sr=sample_rate)


# In[36]:


#final=sum(final_results,[])
#json_data=json.dumps(final)
#print(json_data)
#print(final[:])
#type(json_data)
my_dict={}
json_string=""
i=0
for i in range (len(tag)):
    
    my_dict[i]={'label':tag[i],'end':time[i][1],'begin':time[i][0]}

    #json_data=json.dumps(my_dict[i],indent=4)
    #json_string+=str(json_data)
    #json_data=json.loads(json_data)
    #print(my_dict[i])
    #print(json_data)
    


# In[37]:



#print(len(my_dict))
final_dict={}
objects=0
curr_dict={}
curr_dict=my_dict[objects]
final_dict_count=0
for objects in range (len(my_dict)):
    if curr_dict['label']==my_dict[objects]['label']:
        
        continue
        
    else:
        #print('executing_else')
        curr_dict['end']=my_dict[objects-1]['end']
        final_dict.update(curr_dict)
        curr_dict.update(my_dict[objects])
        #print((final_dict))
        #print('curr_dict',objects,curr_dict)
        json_data=json.dumps(final_dict,indent=4)
        json_string+=str(json_data)
        #print(json_data)
        


    #print(my_dict[objects])
#with open('data.txt', 'a') as outfile:
        #json.dump(argument_filename+'::'+json_string, outfile)


# In[38]:


print(json_string)
del (json_string,curr_dict,my_dict,final_dict)


# In[39]:


delete_files()

