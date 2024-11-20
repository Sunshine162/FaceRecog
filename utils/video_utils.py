import os.path as osp
import shutil
import uuid

from moviepy.editor import VideoFileClip


def copy_audio(video_path1, video_path2):
	"""add audio of video1 to video2"""

	video_clip1 = VideoFileClip(video_path1)
	video_clip2 = VideoFileClip(video_path2)

	video_clip2 = video_clip2.set_audio(video_clip1.audio)
	tmp_file = uuid.uuid4().hex + osp.splitext(video_path2)[-1]
	video_clip2.write_videofile(tmp_file)
	shutil.move(tmp_file, video_path2)
	