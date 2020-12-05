from datetime import datetime

import imageio
import os
import numpy as np
import logging

import LogHelper


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def reset(self):
        self.frames = []

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self,episode):
        if self.enabled:
            filename = f"video_{datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{episode}.mp4"
            path = os.path.join(self.dir_name,filename)
            imageio.mimsave(path, self.frames, fps=self.fps)

    def save_and_reset(self, _episode):
        self.save(_episode)
        self.reset()
        LogHelper.print_step_log("SAVE VIDEO")