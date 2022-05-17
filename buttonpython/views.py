from django.shortcuts import render 
import requests
import re
import json
import torch
import string
import pandas as pd
import speech_recognition as sr 
import moviepy.editor as mp
from os import path
from pydub import AudioSegment
import wave
import audioop
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
from django.core.files.storage import FileSystemStorage
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from summarizer import Summarizer,TransformerSummarizer

def index(request):
    video = Video.objects.all()
    return render(request,"index.html",{"video":video})

def button(request):
    return render(request,"index.html")

def output(request):
    print(" In Progress...")
    
    inp = request.POST.get('param', False)
    video = request.FILES['video']
    print("video is ", video)
    fs = FileSystemStorage()
    filename = fs.save(video.name, video)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print(filename)
    
    if (os.path.exists("media/meet-recording.mp4")):
        print("File exists")
        os.remove("media/meet-recording.mp4")
        print("File removed")

        print("Saving the new file..")
        
        # Get the current working directory
        cwd = os.getcwd()

        # Print the current working directory
        print("Current working directory: {0}".format(cwd))

        dir = './media/'
        allfiles = os.listdir(dir)
        files = [ fname for fname in allfiles if fname.endswith('.mp4')]
        print(files)
        filename = ' '.join(files)
        print(filename)
        new_name = r"meet-recording.mp4"
        os.chdir('./media')
        print("Current working directory: {0}".format(os.getcwd()))
        os.rename(filename, new_name)
        print(filename)
        os.chdir('..')
        print("Current working directory: {0}".format(os.getcwd()))

    else:
        print("File Does Not Exists")
        data = "File Does Not Exists. Please select a file!"
        return render(request,'index.html',{'data': data})
        
 
    clip = mp.VideoFileClip(r"media/meet-recording.mp4") 
 
    clip.audio.write_audiofile(r"meet-recording.wav")

    # files                                                                         
    src = "meet-recording.wav"
    dst = "new-meet-recording.wav"

    # start time in seconds
    startTime = 60.0
    # end time in seconds
    endTime = 120.0

    # create audiosegment                                                            
    sound = AudioSegment.from_mp3(src)
    # cut it to length
    cut = sound[startTime * 1000:endTime * 1000]
    # export sound cut as wav file
    cut.export(dst, format="wav")

    # create a speech recognition object
    r = sr.Recognizer()

    # a function that splits the audio file into chunks
    # and applies speech recognition
    def get_large_audio_transcription(path):
        """
        Splitting the large audio file into chunks
        and apply speech recognition on each of these chunks
        """
        # open the audio file using pydub
        sound = AudioSegment.from_wav(path)  
        # split audio sound where silence is 700 miliseconds or more and get chunks
        chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjust this per requirement
            silence_thresh = sound.dBFS-14,
            # keep the silence for 1 second, adjustable as well
            keep_silence=500,
        )
        folder_name = "audio-chunks"
        # create a directory to store the audio chunks
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""
        # process each chunk 
        for i, audio_chunk in enumerate(chunks, start=1):
            # export audio chunk and save it in
            # the `folder_name` directory.
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            # recognize the chunk
            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                # try converting it to text
                try:
                    text = r.recognize_google(audio_listened)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    print(chunk_filename, ":", text)
                    whole_text += text
        # return the text for all chunks detected

        outfileResults = 'meeting-minutes.txt'

        with open(outfileResults, 'w') as output:
            print(whole_text, file=output)
        return whole_text

    path = "meet-recording.wav"
    data = get_large_audio_transcription(path)
    
    def clean_text(text):
            '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub('[‘’“”…]', '', text)
            text = re.sub('\n', '', text)
            return text
    
    def summarize(path):
        with open(path) as f:
            input_data = f.read()
#             data_clean = clean_text(input_data)
#             tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
#             model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
#             inputs = tokenizer.encode(data_clean, max_length = 1000, return_tensors='pt', truncation=True)
#             outputs = model.generate(inputs,max_length = 50, do_sample=True)
#             output_data = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             return output_data
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        output = ''.join(GPT2_model(input_data, min_length=90))
        return output
 
    #     model = T5ForConditionalGeneration.from_pretrained('t5-large')
    #     tokenizer = T5Tokenizer.from_pretrained('t5-large')
    #     device = torch.device('cpu')
    #     tokenized_text = tokenizer.encode(input_data,max_length = 1000,return_tensors='pt').to(device)
    #     summary_ids = model.generate(tokenized_text, min_length = 30, max_length = 1000)
    #     summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    #     return summary

    path = "meeting-minutes.txt"      
    output = summarize(path)
    data1 = output
    return render(request,'index.html', {'data': data1})

