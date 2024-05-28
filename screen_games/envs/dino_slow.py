import gymnasium
from gymnasium import spaces
import pytesseract
import numpy as np
import time
from eventstreaming.websocket import stream

class DinoEnv(gymnasium.Env):
    metadata = { "render_modes": [], "render_fps": 4 }

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(0, 255, shape=(36, 144, 1), dtype=np.uint8)

        # We have 3 actions, corresponding to 0: "none", 1: "up", 2: "down"
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.prev_score = 0
        self.img = None

        self.pause = False

        stream.start()

    def reset(self, seed=None, options=None):
        inputs = []
        while self.pause:
            time.sleep(0.01)
            inputs = stream.get_inputs()
            if 'p esc' in inputs:
                self.pause = not self.pause

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.prev_score = 0

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
        
        img = img.resize((144, 36)).convert('L')
        observation = np.array(img)[..., np.newaxis]

        return observation, {}

    def step(self, action):
        inputs = stream.get_io_events(timestamp=False)
        if 'p esc' in inputs:
            self.pause = not self.pause
        
        if action == 0:
            pass
        elif action == 1:
            stream.send_io_event('p up')
            # time.sleep(0.2)
            stream.send_io_event('r up')
        # elif action == 2:
        #     stream.send_io_event('p down')
        #     time.sleep(0.5)
        #     stream.send_io_event('r down')

        frame = stream.get_video_frame()
        if frame is not None:
            self.img = frame['image']
        img = self.img
        
        # texts = pytesseract.image_to_string(img.crop((1100, 0, 1440, 60))).strip().split(" ")
        # score = 0
        # if len(texts) == 2:
        #     try:
        #         highest = int(texts[0])
        #         score = int(texts[1])
        #     except ValueError:
        #         score = self.prev_score + 1000
        #     if score - self.prev_score > 100:
        #         score = self.prev_score + 1
        # reward = score - self.prev_score
        # if reward < 0:
        #     reward = 0
        #     score = self.prev_score
        # self.prev_score = score
        reward = 0
        
        text = pytesseract.image_to_string(img.crop((480, 60, 960, 180))).strip()
        terminated = False
        if 'GAME' in text:
            terminated = True
        
        img = img.resize((144, 36)).convert('L')
        observation = np.array(img)[..., np.newaxis]

        return observation, reward, terminated, self.pause, {}

    def render(self):
        pass

    def close(self):
        stream.close()