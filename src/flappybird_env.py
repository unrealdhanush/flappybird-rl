import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from pygame.locals import *
from itertools import cycle
import logging

# Define constants for the game
SCREENWIDTH = 288
SCREENHEIGHT = 512
FPS = 30
PIPEGAPSIZE = 100  # Gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# List of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = [
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
]

# List of pipes
PIPES_LIST = [
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlappyBirdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=False):
        super(FlappyBirdEnv, self).__init__()

        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        self.clock = pygame.time.Clock()
        self.game_over = False
        self.running = True
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)  # Do nothing or flap
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(SCREENHEIGHT, SCREENWIDTH, 1),  # 1 channel grayscale image
            dtype=np.uint8
        )

        logger.info("Loading assets...")
        self.load_assets()
        logger.info("Assets loaded successfully.")
        self.reset()

    def load_assets(self):
        pygame.init()
        try:
            IMAGES['numbers'] = (
                pygame.image.load('assets/sprites/0.png').convert_alpha(),
                pygame.image.load('assets/sprites/1.png').convert_alpha(),
                pygame.image.load('assets/sprites/2.png').convert_alpha(),
                pygame.image.load('assets/sprites/3.png').convert_alpha(),
                pygame.image.load('assets/sprites/4.png').convert_alpha(),
                pygame.image.load('assets/sprites/5.png').convert_alpha(),
                pygame.image.load('assets/sprites/6.png').convert_alpha(),
                pygame.image.load('assets/sprites/7.png').convert_alpha(),
                pygame.image.load('assets/sprites/8.png').convert_alpha(),
                pygame.image.load('assets/sprites/9.png').convert_alpha()
            )

            IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
            IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
            IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

            randPlayer = np.random.choice(len(PLAYERS_LIST))
            IMAGES['player'] = (
                pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
                pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
                pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
            )

            pipeindex = np.random.choice(len(PIPES_LIST))
            IMAGES['pipe'] = (
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
            )

            # Initialize HITMASKS
            HITMASKS['player'] = (
                self.get_hitmask(IMAGES['player'][0]),
                self.get_hitmask(IMAGES['player'][1]),
                self.get_hitmask(IMAGES['player'][2]),
            )
            HITMASKS['pipe'] = (
                self.get_hitmask(IMAGES['pipe'][0]),
                self.get_hitmask(IMAGES['pipe'][1]),
            )
        except Exception as e:
            logger.error(f"Error loading assets: {e}")
            raise e

    def get_hitmask(self, image):
        """Returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def reset(self, seed=None, return_info=False, options=None):
        logger.info("Resetting environment...")
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        pygame.init()
        self.playerIndex = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        self.loopIter = 0

        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int(
            (SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2
        )

        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - SCREENWIDTH

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        self.upperPipes = [
            {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]

        self.lowerPipes = [
            {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        self.pipeVelX = -4

        self.playerVelY = -9
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        self.running = True

        # Return initial observation
        obs = self.get_obs()
        info = {}
        return obs, info

    def get_obs(self):
        """Returns the current state as a 2D observation."""
        obs = np.zeros((SCREENHEIGHT, SCREENWIDTH), dtype=np.uint8)
        
        # Draw player
        player_height = IMAGES['player'][0].get_height()
        obs[int(self.playery):int(self.playery) + player_height, 
            int(self.playerx):int(self.playerx) + IMAGES['player'][0].get_width()] = 255
        
        # Draw pipes
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            obs[:int(uPipe['y']) + IMAGES['pipe'][0].get_height(), 
                int(uPipe['x']):int(uPipe['x']) + IMAGES['pipe'][0].get_width()] = 128
            obs[int(lPipe['y']):, 
                int(lPipe['x']):int(lPipe['x']) + IMAGES['pipe'][0].get_width()] = 128
        
        return obs.reshape(SCREENHEIGHT, SCREENWIDTH, 1)

    def step(self, action):
        pygame.event.pump()
        reward = 0
        done = False
        truncated = False

        if action == 1:
            if self.playery > -2 * IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                reward -= 0.1

        crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                                    self.upperPipes, self.lowerPipes)
        if crashTest[0]:
            self.game_over = True
            reward = -1
            done = True
            self.running = False
            return self.get_obs(), reward, done, truncated, {}

        # Negative reward if bird goes out of the window
        if self.playery < 0 or self.playery > SCREENHEIGHT:
            reward = -1
            done = True
            self.running = False
            return self.get_obs(), reward, done, truncated, {}

        for pipe in self.upperPipes:
            if pipe['x'] < self.playerx < pipe['x'] + IMAGES['pipe'][0].get_width():
                pipe_center = pipe['y'] + PIPEGAPSIZE / 2
                bird_distance_from_center = abs(self.playery - pipe_center)
                reward += 0.1 * (1 - bird_distance_from_center / (SCREENHEIGHT / 2))
                
        playerMidPos = self.playerx + IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                reward += 1 

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        playerHeight = IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, BASEY - self.playery - playerHeight)

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        if self.render_mode:
            self.screen.fill((255, 255, 255))  # Fill the screen with a white background
            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                self.screen.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.screen.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            self.screen.blit(IMAGES['base'], (self.basex, BASEY))
            self.showScore(0)
            self.screen.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))
            pygame.display.update()
            self.clock.tick(FPS)

        return self.get_obs(), reward, done, truncated, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pygame.quit()

    def getRandomPipe(self):
        gapY = np.random.randint(0, int(BASEY * 0.6 - PIPEGAPSIZE))
        gapY += int(BASEY * 0.2)
        pipeHeight = IMAGES['pipe'][0].get_height()
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
        ]

    def checkCrash(self, player, upperPipes, lowerPipes):
        pi = player['index']
        player['w'] = IMAGES['player'][0].get_width()
        player['h'] = IMAGES['player'][0].get_height()

        if player['y'] + player['h'] >= BASEY - 1:
            return [True, True]
        else:
            playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
            pipeW = IMAGES['pipe'][0].get_width()
            pipeH = IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def showScore(self, score):
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0

        for digit in scoreDigits:
            totalWidth += IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.screen.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
            Xoffset += IMAGES['numbers'][digit].get_width()
