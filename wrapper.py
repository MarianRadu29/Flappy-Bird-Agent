import gymnasium as gym
import numpy as np
import cv2
import pygame
from collections import deque


class Wrapper(gym.Wrapper):
    """
    Wrapper all-in-one pentru Flappy Bird:
    - Image processing (extract objects, grayscale, resize 84x84)
    - Frame stacking (4 frames consecutive as state)
    """

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames

        # Frame stack - deque pentru ultimele n_frames
        self.frames = deque(maxlen=n_frames)

        # Observation space: (n_frames, 84, 84) pentru CNN
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, 84, 84), dtype=np.uint8
        )

    def _process_frame(self, frame):
        if frame.ndim == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # mask pentru cer (albastru)
        sky_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))

        # mask pentru nori
        cloud_mask = cv2.inRange(hsv, np.array([40, 0, 130]), np.array([100, 120, 255]))

        # Verde de background (NU tevi / sol)
        bg_green_mask = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([75, 255, 255]))

        # Combin toate mastile de background
        background_mask = cv2.bitwise_or(sky_mask, cloud_mask)
        background_mask = cv2.bitwise_or(background_mask, bg_green_mask)

        # remove small holes in the background mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

        # reverse mask to get objects(pipes, bird, ground)
        objects_mask = cv2.bitwise_not(background_mask)

        # create img
        gray = np.zeros(frame.shape[:2], dtype=np.uint8)
        gray[objects_mask > 0] = 255

        # Crop partea de jos si resize la 84x84
        gray = gray[:410, :]
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        return gray.astype(np.uint8)

    def _get_render_frame(self):
        render_output = self.env.render()

        if render_output is None:
            render_output = pygame.surfarray.array3d(pygame.display.get_surface())
            # W,H,C -> H,W,C
            render_output = np.transpose(render_output, (1, 0, 2))

        return render_output

    def _get_stacked_state(self):
        return np.array(self.frames, dtype=np.uint8)

    def reset(self, **kwargs):
        """reset env and init frame stack."""
        obs, info = self.env.reset(**kwargs)

        # process initial frame
        frame = self._get_render_frame()
        processed = self._process_frame(frame)

        # init frame stack
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(processed)

        return self._get_stacked_state(), info

    def step(self, action):
        """execute action and return stacked state."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # process new frame
        frame = self._get_render_frame()
        processed = self._process_frame(frame)

        # add to frame stack
        self.frames.append(processed)

        return self._get_stacked_state(), reward, terminated, truncated, info
