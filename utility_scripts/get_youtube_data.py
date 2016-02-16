from __future__ import unicode_literals
import youtube_dl
import sys,os


class MyLogger(object):
	def debug(self, msg):
		pass

	def warning(self, msg):
		pass

	def error(self, msg):
		print(msg)


def my_hook(d):
	if d['status'] == 'finished':
		print('Done downloading\n\n\n')
	if d['status'] == 'downloading':
		print 'Video count: ' + str(count) + '\tVideo title: ' + str(d['filename']) + '\tETA: ' + str(d['eta'])

name='starting'
ydl_opts = {
	'format': 'mp4',
	'ignoreerrors': True,
	'outtmpl': name,
	'logger': MyLogger(),
	'progress_hooks': [my_hook],
}
os.chdir('../videos/train')
count=0
with open('../../dataset/trainVidID.txt') as f:
	for i,vidId in enumerate(f):
		vidId=vidId[:-1]
		ydl_opts['outtmpl']=vidId+''
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			try:
				count+=1
				ydl.download([vidId])
			except:
				print sys.exc_info()[0]

os.chdir('../videos/test')
count=0
with open('../../dataset/testVidID.txt') as f:
	for i,vidId in enumerate(f):
		vidId=vidId[:-1]
		ydl_opts['outtmpl']=vidId+''
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			try:
				count+=1
				ydl.download([vidId])
			except:
				print sys.exc_info()[0]