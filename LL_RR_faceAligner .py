import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from imutils import face_utils
from imutils.face_utils import FaceAligner
import numpy as np
import time
import imutils
import dlib
import cv2
from skimage.metrics import structural_similarity as ssim
#import os
import glob
import librosa
#import time
from tkinter import *
from scipy.ndimage import gaussian_filter
from tkinter import filedialog
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webrtcvad
import audioread
import soundfile as sf
import os
import moviepy as mp
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import gc
import tkinter as tk
import wave
def plot_true_speech_below_baseline(ssid, true_list, base_ssid):
    """
    Plot the percentage of false speech values below baseline vs their similarity index (SSID).
    
    Args:
    - ssid: The similarity index list.
    - true_list: The list containing true/false labels for speech or non-speech.
    - base_ssid: The baseline value.
    """
    # Ensure both lists have the same length
    if len(ssid) != len(true_list):
        raise ValueError("ssid and true_list must have the same length")

    # Identify false speech values (those marked as -1 in true_list)
    false_speech_values = [ssid[i] for i in range(len(true_list)) if true_list[i] == 1]
    print(false_speech_values)
    # Create percentage bins from 0 to 100
    percentages = np.linspace(0, 100, 101)
    
    # Calculate how many false speech values are below each percentage of the baseline
    y_values = []
    for pct in percentages:
        threshold = base_ssid * ((100-pct) / 100)
        values_below_threshold = [value for value in false_speech_values if value < threshold]
        
        # Calculate the proportion of the false speech values below the threshold
        proportion_below_threshold = len(values_below_threshold) / len(false_speech_values) if len(false_speech_values) > 0 else 0
        y_values.append(proportion_below_threshold)
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, y_values, color='orange', label='Proportion below threshold')
    plt.xlabel("Percentage Below Baseline (%)")
    plt.ylabel("Proportion of true Speech")
    plt.title("Proportion of true Speech Below Baseline")
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_false_speech_below_baseline(ssid, true_list, base_ssid):
    """
    Plot the percentage of false speech values below baseline vs their similarity index (SSID).
    
    Args:
    - ssid: The similarity index list.
    - true_list: The list containing true/false labels for speech or non-speech.
    - base_ssid: The baseline value.
    """
    # Ensure both lists have the same length
    if len(ssid) != len(true_list):
        raise ValueError("ssid and true_list must have the same length")

    # Identify false speech values (those marked as -1 in true_list)
    false_speech_values = [ssid[i] for i in range(len(true_list)) if true_list[i] == -1]
    print(false_speech_values)
    # Create percentage bins from 0 to 100
    percentages = np.linspace(0, 100, 101)
    
    # Calculate how many false speech values are below each percentage of the baseline
    y_values = []
    for pct in percentages:
        threshold = base_ssid * ((100-pct) / 100)
        values_below_threshold = [value for value in false_speech_values if value < threshold]
        
        # Calculate the proportion of the false speech values below the threshold
        proportion_below_threshold = len(values_below_threshold) / len(false_speech_values) if len(false_speech_values) > 0 else 0
        y_values.append(proportion_below_threshold)
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, y_values, color='orange', label='Proportion below threshold')
    plt.xlabel("Percentage Below Baseline (%)")
    plt.ylabel("Proportion of False Speech")
    plt.title("Proportion of False Speech Below Baseline")
    plt.grid(True)
    plt.legend()
    plt.show()
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order)
    return lfilter(b, a, data)
def show_the_final_result_with_baseline_and_ssim(base_ssid, t_pass, sim_list, true_list, speech_list):
    # Normalize t_pass and sim_list
    t_passn = np.array(t_pass)
    t_passn = 100 * t_passn / t_pass[-1]  # Normalize t_pass to range 0-100
    sim_listn = 100 * np.array(sim_list)  # Convert similarity index to percentage (0-100)
    speech_listn = 100 * np.array(speech_list)
    true_listn = 100 * np.array(true_list, dtype=float)
    
    # Create the data for plotting
    data = {
        'Time': t_passn,
        'Similarity_index': sim_listn,
        'True List': true_listn,
        'Speech List': speech_listn
    }
    df = pd.DataFrame(data, columns=['Time','Similarity_index','True List','Speech List'])

    # Create subplots: 3 rows, 1 column
    res = Tk(className='Final Results')
    figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=100, sharex=True)

    # Plot Speech List on the first subplot
    df[['Time', 'Speech List']].groupby('Time').sum().plot(kind='line', legend=True, ax=ax1, fontsize=10, color='green')
    ax1.set_title('Speech List over Time')
    ax1.set_ylabel('Speech List (%)')
    ax1.legend(['Speech List'])

    # Plot VAD on the second subplot (You can add your VAD logic here)
    df[['Time', 'True List']].groupby('Time').sum().plot(kind='line', legend=True, ax=ax2, fontsize=10, color='blue')
    ax2.set_title('VAD (True List) over Time')
    ax2.set_ylabel('VAD (%)')
    ax2.legend(['True List'])

    # Plot Similarity Index and Baseline on the third subplot
    df[['Time', 'Similarity_index']].groupby('Time').sum().plot(kind='line', legend=True, ax=ax3, fontsize=10, color='orange')
    ax3.axhline(y=base_ssid * 100, color='red', linestyle='--', label='Baseline')
    ax3.set_title('Similarity Index with Baseline over Time')
    ax3.set_ylabel('Similarity Index (%)')
    ax3.legend(['Similarity Index', 'Baseline'])

    # Label x-axis
    ax3.set_xlabel('Time (%)')  # Label for x-axis as percentage time (0-100)

    # Embed the plot in the Tkinter window
    line = FigureCanvasTkAgg(figure, res)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

    # Start Tkinter main loop
    res.mainloop()
def extract_audio_segment_from_video( start_time, end_time,video_clip):
    """
    Extract a segment of the audio corresponding to the start_time and end_time from the video.
    """
    video_duration=video_clip.duration
    end_time=min(end_time,video_duration)
    print("video_duration:",video_duration,"audio_duration:",video_clip.audio.duration)
    if start_time < 0 or end_time > video_duration:
        raise ValueError(f"Invalid time range: start_time should be >= 0 and end_time should be <= {video_duration} seconds.")
    
    # Ensure that end_time is greater than start_time
    print("start time:",start_time,"end time:",end_time)
    if start_time > end_time:
        raise ValueError("start_time must be less than end_time.")
    audio_clip = video_clip.audio.subclip(start_time, end_time)  # Extract audio segment
    print("type audio:",type(audio_clip))
    audio_data=audio_clip.to_soundarray()# This extracts audio directly into a numpy array
    audio_clip.close()
    return audio_data
# def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from the video file using moviepy
    """
    # video_clip = mp.VideoFileClip(video_path)
    # audio_clip = video_clip.audio
    # audio_clip.write_audiofile(audio_path)
    # return audio_path
def convert_audio_format(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_sample_width(2)  # 16-bit PCM
    audio = audio.set_frame_rate(16000)  # 16kHz sample rate
    audio.export(audio_path, format="wav")
    return audio_path
def detect_speech_in_audio(audio_data, sample_rate=48000):
    # 将音频数据转为短时能量
    energy = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)

    # 设置一个阈值，超过阈值即认为是语音
    threshold = 0.001
    speech_activity = energy > threshold
    # 判断是否有语音活动
    return speech_activity
def process_audio_for_frame(start_frame, frame_duration,video_clip):
    """
    Extracts a specific frame's corresponding audio segment and detects speech activity.
    """
    # Calculate the start and end times for audio extraction
    frame_start = start_frame * frame_duration/8
    frame_end = (start_frame/8 + 1) * frame_duration
    
    # First extract audio from video
    audio_data=extract_audio_segment_from_video(frame_start, frame_end, video_clip)
    
    # Now process the extracted audio for speech detection
    speech_activity = detect_speech_in_audio(audio_data)
    # os.remove("extracted_audio_segment.wav")
    del audio_data
    gc.collect
    return speech_activity
def calculate_base_ssid(speech_list, sim_list):
    base_line_arr=[]
    for i in range(0,len(speech_list)):
        if speech_list[i]==0:
            base_line_arr.append(sim_list[i])
    if len(base_line_arr)==0:
        raise ValueError("have no no speech time")
    base_line=sum(base_line_arr)/len(base_line_arr)
    return base_line
def calculate_vad_feature(video_path):
    speech_arr = []
    no_speech_count = 0
    frms = 0  # Frame count
      # Duration of each frame in seconds
    camera = cv2.VideoCapture(video_path)
    fps=camera.get(cv2.CAP_PROP_FPS)
    print("fps")
    frame_duration = 8/fps
    print("frame_duration:",frame_duration)
    frame_count=int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    video_clip = VideoFileClip(video_path)
    while frms<frame_count:
        # Process the video frame for speech detection
        speech_activity = process_audio_for_frame(frms, frame_duration, video_clip)
        # Calculate SSIM only for frames with no speech
        if not any(speech_activity):  # No speech detected
            # Face detection and landmark extraction
        

            for i in range(0,8):
                speech_arr.extend([0])
            no_speech_count += 8
        else:
            for i in range(0,8):
                speech_arr.extend([1])
        frms += 8
        print(frms)
    camera.release()
    # Calculate the baseline SSIM value (average SSIM for frames with no speech)
    print("total count:",frame_count)
    print("no speech count:",no_speech_count)
    if len(speech_arr)>frame_count:
        x=len(speech_arr)
        for i in range(frame_count,x):
            del speech_arr[frame_count]
    print(len(speech_arr))
    print("arr:",speech_arr)
    return speech_arr
# loading models for face detection and set defaults
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
camera = cv2.VideoCapture(0)
main_option=1
# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

#Beginning GUIq
gui = Tk(className=' LLRR_Facial_Similarity')
# set window size and position it at center of screen
windowWidth=800
windowHeight=400
positionRight = int(gui.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(gui.winfo_screenheight()/2 - windowHeight/2)
gui.geometry("{}x{}+{}+{}".format(windowWidth,windowHeight,positionRight,positionDown))
xx=gui.winfo_screenwidth()/2
filepath=""
w = Label(gui, text="\nWelcome! \n\nThis tool helps to analyze similarity of dynamic composite faces\n",font=("Helvetica", 15))
w.pack()
v = IntVar()# identifies which one is selected

Label(gui, text="Select one of the following ways of capturing a video:",justify = LEFT,padx = 20).pack()
Radiobutton(gui, text="Real-time Analysis via webcam",padx = 20, variable=v, value=1).pack(anchor=W)
Radiobutton(gui, text="Analysis of a pre-recorded video",padx = 20, variable=v, value=2).pack(anchor=W)
def interpolate_similarities(sim_list, t_pass):
    # 去除 -0.1 的值，并替换为 NaN
    sim_list = [x if x != -0.1 else np.nan for x in sim_list]
    
    # 找到所有 NaN 的索引
    invalid_indices = np.where(np.isnan(sim_list))[0]
    
    for i in invalid_indices:
        # 查找前一个有效值
        prev_valid_index = i - 1
        while prev_valid_index >= 0 and np.isnan(sim_list[prev_valid_index]):
            prev_valid_index -= 1
        
        # 查找后一个有效值
        next_valid_index = i + 1
        while next_valid_index < len(sim_list) and np.isnan(sim_list[next_valid_index]):
            next_valid_index += 1
        
        # 如果存在有效值，则进行插值
        if prev_valid_index >= 0 and next_valid_index < len(sim_list):
            prev_value = sim_list[prev_valid_index]
            next_value = sim_list[next_valid_index]
            
            # 使用线性插值填补 NaN 值
            sim_list[i] = np.linspace(prev_value, next_value, next_valid_index - prev_valid_index + 1)[i - prev_valid_index]
        elif prev_valid_index < 0 and next_valid_index < len(sim_list):
            sim_list[i] = sim_list[next_valid_index]  # 使用后续有效值填补
        elif next_valid_index >= len(sim_list) and prev_valid_index >= 0:
            sim_list[i] = sim_list[prev_valid_index]  # 使用前值填补
            
    # 使用高斯滤波平滑曲线
    sim_list = gaussian_filter(sim_list, sigma=2)  # 通过调整sigma来控制平滑程度

    return sim_list
def helloCallBack():
    global camera
    global main_option
    if v.get()==1:
        gui.destroy()
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        Label(tempp,text="\nWebCam Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin Real-time streaming!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop recording anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
        
    if v.get()==2:
        global filepath
        root = Tk(className=' Choose Video...')
        root.geometry("500x100+10+10")#width x heigth
        w1 = Label(root, text="\nBrowse your system for the Test Video...",font=("Helvetica", 15))
        w1.pack()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("All files","*.*"),("jpeg files","*.jpg")))
        test_video_path = root.filename
        filepath=test_video_path
        root.destroy()
        
        camera = cv2.VideoCapture(test_video_path)# from the pre recorded video in path
        main_option=2
        gui.destroy()
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        Label(tempp,text="\nPreliminary Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin video analysis!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
button = Button(gui, text='Confirm', width=25, command=helloCallBack)
button.pack()
gui.mainloop()
# starting video streaming
cv2.namedWindow('TestVideo')
cv2.namedWindow('Aligned')
cv2.namedWindow('LL RR composites')
cv2.moveWindow('TestVideo', int(xx-400),75)# width wise centerscreen
sim_list=[]
t_pass = []
def caculate_sim():
    tlt = 35  # number of pixels of tilt allowance (allow if <tlt)
    t_pass = []
    frms = 0
    sim_list = []  # Initialize sim_list
    
    while camera.isOpened():
        ret, frame = camera.read()  # by default the webcam reads at around 30fps, can be changed by other codes
        if ret == False:
            break

        # reading the frame
        frame = imutils.resize(frame, width=800)
        if main_option == 1:
            frame = cv2.flip(frame, 1)
        frameClone = frame.copy()
        frameClone = cv2.putText(frameClone, 'Press Q to stop', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        t_pass.append(frms)
        if frms%3==0:

        # Insert -0.1 for every frame in sim_list to keep the size consistent
                sim_list.append(-0.1)  
            
                frms += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to stop
                    break
                
                # Begin finding 68 facial landmarks using dlib
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray_frame, 1)
                
                if len(rects) != 0:
                    shape = predictor(gray_frame, rects[0])
                    shape = face_utils.shape_to_np(shape)

                    # Face alignment checking
                    ylj = shape[0][1]  # y coordinate of left jaw
                    yrj = shape[16][1]  # y coordinate of right jaw
                    xtn = shape[27][0]  # x coordinate of top of nose
                    xbn = shape[30][0]  # x coordinate of bottom of nose

                    faceAligned = fa.align(frame, gray_frame, rects[0])
                    cv2.imshow('Aligned', faceAligned)

                    if abs(ylj - yrj) >= tlt or abs(xtn - xbn) >= tlt:
                        cv2.imshow('TestVideo', frameClone)
                        continue

                    (h, w) = faceAligned.shape[:2]
                    crop_face = faceAligned[h // 10:h * 9 // 10, w // 10:w * 9 // 10]
                    
                    # Compute LL and RR composites
                    (hh, ww, dd) = crop_face.shape
                    if ww % 2 == 0:
                        ww1 = ww // 2 - 1
                    else:
                        ww1 = ww // 2
                    flipHorizontal = cv2.flip(crop_face, 1)
                    
                    img1 = crop_face[:, 0:ww1]
                    img2 = flipHorizontal[:, ww1 + 1:]
                    LL = np.concatenate((img1, img2), axis=1)
                    img1 = flipHorizontal[:, 0:ww1]
                    img2 = crop_face[:, ww1 + 1:]
                    RR = np.concatenate((img1, img2), axis=1)
                    llrr = np.concatenate((LL, RR), axis=0)
                    cv2.imshow('LL RR composites', llrr)

                    LL_resized = cv2.resize(LL, (300, 300))
                    RR_resized = cv2.resize(RR, (300, 300))

                    # Calculate similarity index (0-1)
                    sim_index = ssim(cv2.cvtColor(LL_resized, cv2.COLOR_BGR2GRAY), cv2.cvtColor(RR_resized, cv2.COLOR_BGR2GRAY))
                    
                    # Update sim_list with the calculated SSIM value
                    sim_list[frms - 1] = sim_index
                    
                    cv2.imshow('TestVideo', frameClone)

                else:
                    cv2.imshow('TestVideo', frameClone)
                    continue
        else:
            frms+=1
            sim_list.append(-0.1) 
    # Release resources
    camera.release()
    cv2.destroyAllWindows()

    return sim_list, t_pass

video_path = filepath
root = Tk(className=' Choose true txt...')
root.geometry("500x100+10+10")#width x heigt
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select txt",filetypes = (("All files","*.*"),("Text files","*.txt")))
test_txt_path = root.filename
root.destroy()
file=open(test_txt_path,"r+")
str=file.read()
true_list=str.split(",")
print("true_list:",true_list,type(true_list))
file.close()
start_time=time.time()
sim_list,t_pass=caculate_sim()
end_time=time.time()

start_time=time.time()
sim_list=interpolate_similarities(sim_list,t_pass)
speech_list = calculate_vad_feature(video_path)
end_time=time.time()
print("vad times:",end_time-start_time)
print(len(sim_list), len(speech_list))
print(type(sim_list),type(speech_list))
true_listn=np.array(true_list, dtype=float)
base_ssid = calculate_base_ssid(speech_list, sim_list)
print(base_ssid)
print("ssim times:",end_time-start_time)
plot_true_speech_below_baseline(sim_list,true_listn,base_ssid)
plot_false_speech_below_baseline(sim_list,true_listn,base_ssid)
show_the_final_result_with_baseline_and_ssim(base_ssid, t_pass, sim_list, true_list, speech_list)