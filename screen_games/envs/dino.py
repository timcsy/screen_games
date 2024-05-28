import gymnasium
from gymnasium import spaces
import numpy as np
import time
from PIL import ImageChops
from eventstreaming import stream

class DinoEnv(gymnasium.Env):
    metadata = { 'render_modes': [], 'render_fps': 4 }

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict(
            {
                'screen': spaces.Box(low=0, high=255, shape=(36, 144, 3), dtype=np.uint8)
            }
        )

        # We have 3 actions, corresponding to 0: "none", 1: "up", 2: "down"
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.prev_img = None
        self.img = None
        self.count = 0

        self.pause = False

        stream.start(top=330, width=1440, height=360)

    def reset(self, seed=None, options=None):
        inputs = []
        while self.pause:
            time.sleep(0.01)
            inputs = stream.get_io_events(timestamp=False)
            if 'p esc' in inputs:
                self.pause = not self.pause

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Focus on the game
        stream.send_io_event('m200,860')
        stream.send_io_event('cpleft')
        stream.send_io_event('crleft')
        time.sleep(1)
        stream.send_io_event('p space')
        stream.send_io_event('r space')

        frame = stream.get_video_frame()
        while frame is None:
            frame = stream.get_video_frame()
        self.img = frame['image']
        img = self.img
        self.prev_img = img
        
        # observation = { 'screen': np.array(img) }
        observation = { 'screen': np.array(img.resize((144, 36))) }

        return observation, {}

    def step(self, action):
        inputs = stream.get_io_events(timestamp=False)
        if 'p esc' in inputs:
            self.pause = not self.pause
        
        if action == 0 or self.count > 0:
            pass
        elif action == 1:
            stream.send_io_event('p up')
            time.sleep(0.2)
            stream.send_io_event('r up')
        # elif action == 2:
        #     stream.send_io_event('p down')
        #     time.sleep(0.5)
        #     stream.send_io_event('r down')

        time.sleep(0.1)
        frame = stream.get_video_frame()
        if frame is not None:
            self.img = frame['image']
        img = self.img
        
        reward = 0
        
        terminated = False
        diff = ImageChops.difference(self.img, self.prev_img)
        if not diff.getbbox():
            self.count += 1
        if self.count > 5:
            self.count = 0
            terminated = True
        self.prev_img = self.img
        
        # observation = { 'screen': np.array(img) }
        observation = { 'screen': np.array(img.resize((144, 36))) }

        return observation, reward, terminated, self.pause, {}

    def render(self):
        pass

    def close(self):
        stream.close()