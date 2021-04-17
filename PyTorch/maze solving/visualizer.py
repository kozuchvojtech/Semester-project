import pygame
import numpy as np
import json
from simple_value_object import ValueObject

from move import Move

TILE_SIZE = 36
HERO_SIZE = 44

class Episode(ValueObject):
    def __init__(self, actions, map_path):
        pass

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
    
class Hero:
    def __init__(self,x,y,on_episode_changed):
        self.move = Move.RIGHT
        self.animation_index = 1
        self.init_x, self.init_y = x*TILE_SIZE+TILE_SIZE, y*TILE_SIZE+TILE_SIZE
        self.x, self.y = self.init_x, self.init_y 
        self.episodes = []
        self.curr_episode = Episode([], '')
        self.on_episode_changed = on_episode_changed
    
    def populate_image(self):
        self.file_name = f'static/player/{self.move.name.lower()}/{self.animation_index}.png'
        self.img = pygame.image.load(self.file_name)
        self.img = pygame.transform.scale(self.img, (HERO_SIZE,HERO_SIZE))
        self.img_rect = self.img.get_rect()
        self.img_rect.topleft = (self.x, self.y-20)

    def get_position(self, move):
        try:
            prev_x = self.x
            prev_y = self.y

            if move is Move.UP:
                self.y -= TILE_SIZE/6
            elif move is Move.RIGHT:
                self.x += TILE_SIZE/6
            elif move is Move.DOWN:
                self.y += TILE_SIZE/6
            elif move is Move.LEFT:
                self.x -= TILE_SIZE/6

            return (self.x, self.y)
        except:
            pass
    
    def append_episode(self, episode):
        self.episodes.append(Episode(list(np.repeat(episode.actions, TILE_SIZE/6)), episode.map_path))
    
    def iterate_episodes(self):
        if self.episodes and not self.curr_episode.actions:
            self.curr_episode = self.episodes.pop(0)            
            self.on_episode_changed(self.curr_episode)
            self.x = self.init_x
            self.y = self.init_y
        
        if self.curr_episode.actions:
            curr_episode_actions = list(self.curr_episode.actions)
            self.move = Move(curr_episode_actions.pop(0))
            self.curr_episode = Episode(curr_episode_actions, self.curr_episode.map_path)
            return self.move
    
    def show(self, game_display):
        prev_move = self.move

        if self.iterate_episodes() and self.get_position(self.move):

            if prev_move != self.move:
                self.animation_index = 1
            else:
                self.animation_index += 1

                if self.animation_index > 30:
                    self.animation_index = 1

            self.populate_image()
            game_display.blit(self.img, self.img_rect)

            return self.curr_episode.map_path

class EpisodeSnapshot():
    def __init__(self, map_path):
        self.episode = []
        self.map_path = map_path

    def snapshot(self, episode, map_path):
        self.episode = episode
        self.map_path = map_path
    
    def get_episode(self):
        return Episode(self.episode, self.map_path)

class Game:
    def __init__(self, episodeSnapshot, display_width = 432, display_height = 432):
        pygame.display.set_caption('Grab the coin!')
        pygame.init()

        self.fpsClock = pygame.time.Clock()

        self.display_width = display_width
        self.display_height = display_height
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height))

        self.episodeSnapshot = episodeSnapshot
        self.last_appended_episode = self.episodeSnapshot.get_episode()
        self.last_displayed_episode = self.last_appended_episode
        self.curr_map_path = self.last_appended_episode.map_path
        
        self.populate_map(self.game_display)
        self.populate_coins(self.curr_map_path)
        self.populate_hero()
        
    def play(self):
        self.draw_background()

        for road in self.roads:
            road.show(self.game_display)
            
        if any(self.hero.x == coin.x and self.hero.y == coin.y for coin in self.coins):
            for coin in self.coins:
                if coin.x == self.hero.x and coin.y == self.hero.y:
                    self.coins.remove(coin)

        for coin in self.coins:
            coin.show(self.game_display)

        self.append_episode()
        self.hero.show(self.game_display)

        pygame.display.flip()
        self.fpsClock.tick(30)
    
    def on_episode_changed(self, episode):
        if episode.actions:
            if episode.map_path != self.curr_map_path:
                self.curr_map_path = episode.map_path
            self.populate_coins(self.curr_map_path)

    def append_episode(self):
        prev_episode = self.last_appended_episode
        self.last_appended_episode = self.episodeSnapshot.get_episode()

        if self.last_appended_episode.actions and prev_episode.actions != self.last_appended_episode.actions:
            self.hero.append_episode(self.last_appended_episode)
    
    def draw_background(self):
        land_image = pygame.image.load('static/land.png')
        land_image = pygame.transform.scale(land_image, (self.game_display.get_width()//6,self.game_display.get_height()//6))
        land_rect = land_image.get_rect()

        self.game_display.fill(pygame.Color(0,0,0))

        for i in np.linspace(0, self.game_display.get_width(), num=7):
            for j in np.linspace(0, self.game_display.get_height(), num=7):
                land_rect.topleft = (i,j)
                self.game_display.blit(land_image, land_rect)

    def populate_map(self, playground):
        with open('static/map/default.json') as f:
            records = json.load(f)

        roads = []

        for record in records:
            road = Road(record['x'], record['y'], record['img_file_name'])
            roads.append(road)
        
        self.roads = roads

    def populate_coins(self, map_path):
        with open(map_path) as f:
            records = json.load(f)
        
        coins = []

        for record in records:
            coin = Coin(record['x'], record['y'])
            coins.append(coin)

        self.coins = coins
    
    def populate_hero(self):
        self.hero = Hero(0,0, self.on_episode_changed)

class Visualizer:
    def __init__(self, game):
        self.game = game

    def main_loop(self):
        while True:
            pygame.event.get()
            self.game.play()