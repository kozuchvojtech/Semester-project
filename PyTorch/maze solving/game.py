import pygame
import sys
import time
import random
from pygame.locals import *
from utils import text_object, message_display, button
import threading
import numpy as np
import json
from move import Move
from agent_train import agent_train
from environment import MazeEnvironmentWrapper
from multiprocessing import Process
from multiprocessing.managers import BaseManager

######################## Initialize colours ########################
# RED
redColour = pygame.Color(200,0,0)
brightRedColour = pygame.Color(255,0,0)
# GREEN
brightGreenColour = pygame.Color(0,255,0)
greenColour = pygame.Color(0,200,0)
brightGreenColour1 = (150, 255, 150)
darkGreenColour1 = (0, 155, 0)
# BLACK
blackColour = pygame.Color(0,0,0)
# WHITE
whiteColour = pygame.Color(255,255,255)
# GRAY
greyColour = pygame.Color(150,150,150)
LightGrey = pygame.Color(220,220,220)
####################################################################

TILE_SIZE = 36

train_1 = np.matrix([
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', 'C', '%', '%', 'C', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', 'C', '%', '%', 'C', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%']
])

train_2 = np.matrix([
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', 'C', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', 'C', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', 'C', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', 'C', '%', '%', '%', '%', '%', '%', '%']
])

train_3 = np.matrix([
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', 'C', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', 'C', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', 'C', '%', '%', '%', '%', '%', 'C', '%']
])

testing = np.matrix([
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', 'C', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', 'C', '%', '%', '%', '%', '%', '%', 'C', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%']
])

class Coin:
    def __init__(self,x,y):
        self.file_name_pattern = "static/coin/000{animation_index:n}.png"
        self.animation_index = 1
        self.img = pygame.image.load(self.file_name_pattern.format(animation_index = self.animation_index))
        self.x = x*TILE_SIZE+TILE_SIZE
        self.y = y*TILE_SIZE+TILE_SIZE

    def show(self, game_display):
        try:
            self.animation_index += 1

            if self.animation_index > 10:
                self.animation_index = 1

            self.img = pygame.image.load(self.file_name_pattern.format(animation_index = self.animation_index))
            game_display.blit(self.img, (self.x, self.y))
        except:
            pass

class Road:
    def __init__(self,x,y,road_file):
        self.file_name = f'static/road/{road_file}'
        self.img = pygame.image.load(self.file_name)
        self.img_rect = self.img.get_rect()
        self.x = x*TILE_SIZE+TILE_SIZE
        self.y = y*TILE_SIZE+TILE_SIZE

    def show(self, game_display):
        try:
            self.img_rect.topleft = (self.x, self.y)
            game_display.blit(self.img, self.img_rect)
        except:
            pass

class Player:
    def __init__(self,x,y):        
        self.init_x = x*TILE_SIZE+TILE_SIZE
        self.init_y = y*TILE_SIZE+TILE_SIZE
        self.x, self.y = self.init_x, self.init_y
        self.episodes = []
        self.curr_episode = []

    def get_position(self, move):
        try:
            prev_x = self.x
            prev_y = self.y

            if move is Move.UP:
                self.y -= TILE_SIZE0
            elif move is Move.RIGHT:
                self.x += TILE_SIZE
            elif move is Move.DOWN:
                self.y += TILE_SIZE
            elif move is Move.LEFT:
                self.x -= TILE_SIZE

            # print('('+str(prev_x)+','+str(prev_y)+') '+str(move)+' => '+ '('+str(self.x)+','+str(self.y)+')')

            return (self.x, self.y)
        except:
            pass
    
    def append_episode(self, episode):
        self.episodes.append(episode)
    
    def get_episodes(self):
        return self.episodes
    
    def iterate_episodes(self):
        if self.episodes and not self.curr_episode:
            self.curr_episode = self.episodes.pop(0)
            self.x = self.init_x
            self.y = self.init_y
        
        if self.curr_episode:
            return Move(self.curr_episode.pop(0))

class Game:
    def __init__(self, player, map_path, display_width = 432, display_height = 432):
        self.fpsClock = pygame.time.Clock()

        self.display_width = display_width
        self.display_height = display_height
        self.playSurface = pygame.display.set_mode((self.display_width, self.display_height))
                
        self.player = player
        self.player_file_name = f'static/player/right/01.png'
        self.player_img = pygame.image.load(self.player_file_name)
        self.player_img = pygame.transform.scale(self.player_img, (TILE_SIZE,TILE_SIZE))
        self.player_img_rect = self.player_img.get_rect()

        self.roads = self.populate_map(self.playSurface)
        self.populate_coins(map_path)

    def gameOver(self, score):
        # Set fonts of caption
        gameOverFont = pygame.font.Font('arial.ttf', 72)
        gameOverSurf, gameOverRect = text_object('Game Over', gameOverFont, greyColour)
        gameOverRect.midtop = (320, 125)
        self.playSurface.blit(gameOverSurf, gameOverRect)
        # Display scores and set fonts
        scoreFont = pygame.font.Font('arial.ttf', 48)
        scoreSurf, scoreRect = text_object('SCORE:'+str(score), scoreFont, greyColour)
        scoreRect = scoreSurf.get_rect()
        scoreRect.midtop = (320, 225)
        self.playSurface.blit(scoreSurf, scoreRect)
        #pygame.display.update() # Refresh display

        button(self.playSurface, 'Again', self.display_width//4, self.display_height//8*7, self.display_width//2, self.display_height//8, greenColour, brightGreenColour, self.init_game)
        # https://stackoverflow.com/questions/55881619/sleep-doesnt-work-where-it-is-desired-to/55882173#55882173
        pygame.display.update()
        
    def play(self):
        land_image = pygame.image.load('static/land.png')
        land_image = pygame.transform.scale(land_image, (self.playSurface.get_width()//6,self.playSurface.get_height()//6))
        land_rect = land_image.get_rect()

        self.playSurface.fill(blackColour)

        for i in np.linspace(0, self.playSurface.get_width(), num=7):
            for j in np.linspace(0, self.playSurface.get_height(), num=7):
                land_rect.topleft = (i,j)
                self.playSurface.blit(land_image, land_rect)

        for road in self.roads:
            road.show(self.playSurface)

        for coin in self.coins:
            coin.show(self.playSurface)

        self.iterate_player_episodes()

        pygame.display.flip()
        self.fpsClock.tick(15)
    
    def iterate_player_episodes(self):
        move = self.player.iterate_episodes()
        player_position = self.player.get_position(move)

        if move and player_position:
            player_x, player_y = player_position
            player_x += 10
            player_y -= 10
            self.player_img_rect.topleft = (player_x,player_y)

            self.playSurface.blit(self.player_img, self.player_img_rect)

    def populate_map(self, playground):
        with open('static/map/default.json') as f:
            records = json.load(f)

        roads = []

        for record in records:
            road = Road(record['x'], record['y'], record['img_file_name'])
            roads.append(road)
        
        return roads

    def populate_coins(self, map_path):
        print(map_path)
        with open(map_path) as f:
            records = json.load(f)
        
        coins = []

        for record in records:
            coin = Coin(record['x'], record['y'])
            coins.append(coin)

        self.coins = coins

def training_process(agent, env, player, callback):
    agent.train(env, player, callback)

def game_loop(player, map_path):
    game = Game(player, map_path)

    while True:
        pygame.event.get()
        game.play()
        # if done:
        #     game.gameOver(score)

def snapshot_episode(player, episode):
    player.append_episode(episode)

if __name__ == '__main__':
    pygame.display.set_caption('Grab the coin!')
    pygame.init()

    BaseManager.register('Player', Player)
    manager = BaseManager()
    manager.start()
    player = manager.Player(0,0)
  
    env = MazeEnvironmentWrapper(train_1)
    agent = agent_train(env.observations_count, env.actions_count)
    agent_training = Process(target=training_process, args=(agent, env, player, snapshot_episode))

    main_loop = Process(target=game_loop, args=(player,'static/map/training_1.json'))

    agent_training.start()
    main_loop.start()

    agent_training.join(None)

    # env = MazeEnvironmentWrapper(train_2)
    # agent = agent_train(env.observations_count, env.actions_count)
    # agent_training = Process(target=training_process, args=(agent, env, player, snapshot_episode))
    # agent_training.start()
    # agent_training.join(None)

    # env = MazeEnvironmentWrapper(train_3)
    # agent = agent_train(env.observations_count, env.actions_count)
    # agent_training = Process(target=training_process, args=(agent, env, player, snapshot_episode))
    # agent_training.start()
    # agent_training.join(None)

    main_loop.join(None)