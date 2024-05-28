import gymnasium
from gymnasium import spaces
import numpy as np
import time
from eventstreaming import stream
from screen_games.config import bool_ENV, int_ENV

class ScreenEnv(gymnasium.Env):
    metadata = { 'render_modes': [], 'render_fps': 30 }

    def __init__(self, render_mode=None, left=None, top=None, width=None, height=None, new_width=None, new_height=None, fullscreen=None):
        if left is None:
            left = int_ENV('SCREENENV_LEFT', 0)
        if top is None:
            top = int_ENV('SCREENENV_TOP', 0)
        if width is None:
            width = int_ENV('SCREENENV_WIDTH', -1)
            if width == -1: width = None
        if height is None:
            height = int_ENV('SCREENENV_HEIGHT', -1)
            if height == -1: height = None
        if new_width is None:
            new_width = int_ENV('SCREENENV_NEW_WIDTH', -1)
            if new_width == -1: new_width = width
        if new_height is None:
            new_height = int_ENV('SCREENENV_NEW_HEIGHT', -1)
            if new_height == -1: new_height = height
        if fullscreen is None:
            fullscreen = bool_ENV('SCREENENV_FULLSCREEN', False)
        
        self.observation_space = spaces.Dict(
            {
                'screen': spaces.Box(low=0, high=255, shape=(new_height, new_width, 3), dtype=np.uint8),
                'timestamp': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.frame = None
        self.start_time = 0

        self.pause = False

        stream.start(left=left, top=top, width=width, height=height, new_width=new_width, new_height=new_height, fullscreen=fullscreen)

    def reset(self, seed=None, options=None, on_reset=None):
        inputs = []
        while self.pause:
            time.sleep(1.0 / ScreenEnv.metadata['render_fps'])
            inputs = stream.get_io_events()
            if 'p esc' in list(map(lambda x: x['event'], inputs)):
                self.pause = not self.pause
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if on_reset is not None:
            on_reset()

        frame = stream.get_video_frame()
        while frame is None:
            frame = stream.get_video_frame()
        self.frame = frame
        self.start_time = self.frame['timestamp'] / 1_000_000.0
        
        observation = {
            'screen': np.array(self.frame['image']),
            'timestamp': np.array([self.start_time], dtype=np.float32)
        }

        return observation, { 'inputs': inputs }

    def step(self, action):
        inputs = stream.get_io_events()
        if 'p esc' in list(map(lambda x: x['event'], inputs)):
            self.pause = not self.pause

        time.sleep(1.0 / ScreenEnv.metadata['render_fps'])
        frame = stream.get_video_frame()
        if frame is not None:
            self.frame = frame
        
        observation = {
            'screen': np.array(self.frame['image']),
            'timestamp': np.array([self.frame['timestamp'] / 1_000_000.0 - self.start_time], dtype=np.float32)
        }

        return observation, 0, False, self.pause, { 'inputs': inputs }

    def render(self):
        pass

    def close(self):
        stream.close()