#Audio
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

epochs = 10000
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 32

cleaners='english_cleaners' #'korean_cleaners'

data_path = './data/LJSpeech-1.1'
#korean_data_path = ''
checkpoint_path = './checkpoints'
sample_path = './samples'
#korean_data_energy_pitch_min_max
pitch_min = 71.0
pitch_max = 792.8
energy_min = 0.0
energy_max = 283.72


sampling_rate =  22050
hop_length = 256
filter_length = 1024
