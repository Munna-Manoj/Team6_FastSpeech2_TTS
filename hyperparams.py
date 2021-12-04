import os

# Audio
num_mels = 80
n_fft = 2048
sr = 22050
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude

hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20

outputs_per_step = 1

n_iter = 60
epochs = 10000
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 32

'''
english = True
if english:
	cleaners= 'english_cleaners'
	data_path = './data/LJSpeech-1.1'
else:
	cleaners= 'korean_cleaners'  # 'english_cleaners'
	data_path = './data/kss' #  './data/LJSpeech-1.1'
'''

#cleaners= 'english_cleaners'
#data_path = './data/LJSpeech-1.1'

cleaners= 'korean_cleaners'  
data_path = './data/kss' #  

max_wav_value = 32768.0

checkpoint_path = './checkpoints'
sample_path = './samples'
vocoder_pretrained_model_name = "vocgan_kss_pretrained_model_epoch_4500.pt"
vocoder_pretrained_model_path = os.path.join("./checkpoints", vocoder_pretrained_model_name) 