import os
import glob
import gc
from os.path import basename

files = glob.glob('audio_conversion/*.avi')

for file in files:
	os.system('ffmpeg -i '+str(file)+' audio/'+str(basename(file))+'.wav')
	gc.collect()
	print('finished: '+str(file))
