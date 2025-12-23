import gymnasium as gym
import numpy as np
import cv2
import pygame


def process_frame(frame):
    if frame.ndim == 2 or frame.shape[-1] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    sky_mask = cv2.inRange(
        hsv,
        np.array([90, 50, 50]),
        np.array([130, 255, 255])
    )

    # nori
    cloud_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 70, 255])
    )

    # verde de background (NU tevi / sol)
    bg_green_mask = cv2.inRange(
        hsv,
        np.array([35, 40, 80]),
        np.array([85, 120, 255])
    )

    background_mask = cv2.bitwise_or(sky_mask, cloud_mask)
    background_mask = cv2.bitwise_or(background_mask, bg_green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    background_mask = cv2.morphologyEx(
        background_mask, cv2.MORPH_CLOSE, kernel
    )

    objects_mask = cv2.bitwise_not(background_mask)

    gray = np.zeros(frame.shape[:2], dtype=np.uint8)
    gray[objects_mask > 0] = 255

    gray = gray[:400, :]
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    gray = np.expand_dims(gray, axis=-1)

    return gray.astype(np.float32)


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        render_output = self.env.render()

        if render_output is None:
            render_output = pygame.surfarray.array3d(pygame.display.get_surface())
            # W,H,C -> H,W,C
            render_output = np.transpose(render_output, (1, 0, 2))

        return process_frame(render_output)
