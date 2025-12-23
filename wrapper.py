import gymnasium as gym
import numpy as np
import cv2
import pygame


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        render_output = self.env.render()

        if render_output is None:
            try:
                raw_pixels = pygame.surfarray.array3d(pygame.display.get_surface())
                raw_pixels = np.transpose(raw_pixels, (1, 0, 2))
            except Exception:
                raw_pixels = np.zeros((512, 288, 3), dtype=np.uint8)
        else:
            raw_pixels = render_output

        processed = cv2.resize(raw_pixels, (84, 84))
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        processed = np.expand_dims(processed, axis=-1)
        return processed
