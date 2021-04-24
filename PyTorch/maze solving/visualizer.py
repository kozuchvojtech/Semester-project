import pygame
import numpy as np
import json
from simple_value_object import ValueObject
from utils import text_object, message_display, button
import asyncio

from move import Move

TILE_SIZE = 36
HERO_SIZE = 48

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

class Episode(ValueObject):
    def __init__(self, actions, reward, map_path, coin_positions):
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
        self.curr_episode = Episode([], 0, '', [])
        self.on_episode_changed = on_episode_changed
    
    def populate_image(self):
        self.file_name = f'static/player/{self.move.name.lower()}/{self.animation_index}.png'
        self.img = pygame.image.load(self.file_name)
        self.img = pygame.transform.scale(self.img, (HERO_SIZE,HERO_SIZE))
        self.img_rect = self.img.get_rect()
        self.img_rect.topleft = (self.x, self.y-25)

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
        self.episodes.append(Episode(list(np.repeat(episode.actions, TILE_SIZE/6)), episode.reward, episode.map_path, episode.coin_positions))
    
    def iterate_episodes(self):
        if self.episodes and not self.curr_episode.actions:
            self.curr_episode = self.episodes.pop(0)            
            self.on_episode_changed(self.curr_episode)
            self.x = self.init_x
            self.y = self.init_y
        
        if self.curr_episode.actions:
            curr_episode_actions = list(self.curr_episode.actions)
            self.move = Move(curr_episode_actions.pop(0))
            self.curr_episode = Episode(curr_episode_actions, self.curr_episode.reward, self.curr_episode.map_path, self.curr_episode.coin_positions)
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
            
            return (not self.curr_episode.actions, self.curr_episode.reward)
        
        return (False, 0)

class EpisodeSnapshot():
    def __init__(self, map_path):
        self.init_episode = Episode([], 0, map_path, [])
        self.episodes = [self.init_episode]

    def snapshot(self, actions, reward, map_path, coin_positions):
        self.episodes.append(Episode(actions, reward, map_path, coin_positions))
    
    def get_episode(self):
        if self.episodes:
            return self.episodes.pop(0)
        else:
            return self.init_episode

class AnimatedHero(pygame.sprite.Sprite):
    def __init__(self, position_x, position_y, hero_type='greeting'):
        super().__init__()

        self.images = []
        self.shadow_img = pygame.image.load('static/player/shadow.png')

        for i in range(30):
            self.images.append(pygame.image.load(f'static/player/{hero_type}/{i+1}.png'))
        
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.shadow_rect = self.shadow_img.get_rect()
        self.rect.midtop = (position_x, position_y)
        self.shadow_rect.midtop = (position_x, position_y + (self.image.get_height()-14))

    def update(self):
        self.index += 1
        if self.index >= len(self.images):
            self.index = 0

        self.image = self.images[self.index]
    
    def draw(self, screen):
        screen.blit(self.shadow_img, self.shadow_rect)
        screen.blit(self.image, self.rect)

class Game:
    def __init__(self, episodeSnapshot, display_width = 432, display_height = 432):
        pygame.display.set_caption('Grab the coin!')
        pygame.init()

        self.fpsClock = pygame.time.Clock()

        self.display_width = display_width
        self.display_height = display_height
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height))

        self.episodeSnapshot = episodeSnapshot
        self.curr_episode = self.episodeSnapshot.get_episode()
        self.last_appended_episode = None
        
        self.populate_map()
        self.populate_coins()
        self.populate_hero()
        
    def play(self):
        self.draw_background()
            
        if any(self.hero.x == coin.x and self.hero.y == coin.y for coin in self.coins):
            for coin in self.coins:
                if coin.x == self.hero.x and coin.y == self.hero.y:
                    self.coins.remove(coin)

        for coin in self.coins:
            coin.show(self.game_display)

        self.append_episode()
        done, reward = self.hero.show(self.game_display)

        pygame.display.flip()
        self.fpsClock.tick(30)
        return done, reward
    
    def gameOver(self, score):
        player_sprite = AnimatedHero(self.display_width//2, self.display_height//2 - 20)

        for _ in range(30):
            self.draw_background()

            surface = pygame.Surface([self.display_width,self.display_height], pygame.SRCALPHA)
            surface.set_alpha(120)
            surface.fill(pygame.Color(255,255,255))

            gameOverFont = pygame.font.Font('arcade.ttf', 48)
            gameOverSurf, gameOverRect = text_object('Game Over', gameOverFont, pygame.Color(120,120,120))
            gameOverRect.midtop = (self.display_width//2, self.display_height//2 - 120)
            
            scoreFont = pygame.font.Font('arcade.ttf', 36)
            scoreSurf, scoreRect = text_object('Reward '+str(int(score)), scoreFont, pygame.Color(120,120,120))
            scoreRect = scoreSurf.get_rect()
            scoreRect.midtop = (self.display_width//2, self.display_height//2 + 100)

            self.game_display.blit(surface, (0,0))
            self.game_display.blit(gameOverSurf, gameOverRect)        
            self.game_display.blit(scoreSurf, scoreRect)

            player_sprite.update()
            player_sprite.draw(self.game_display)
            
            pygame.display.flip()
            self.fpsClock.tick(30)        
    
    def on_episode_changed(self, episode):
        if episode.actions:
            prev_map_path = self.curr_episode.map_path
            self.curr_episode = episode

            if self.curr_episode.map_path != prev_map_path:
                self.populate_map()

            self.populate_coins()

    def append_episode(self):
        prev_episode = self.last_appended_episode
        self.last_appended_episode = self.episodeSnapshot.get_episode()

        if not prev_episode or (self.last_appended_episode.actions and prev_episode.actions != self.last_appended_episode.actions):
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
        
        for road in self.roads:
            road.show(self.game_display)

    def populate_map(self):
        with open(self.curr_episode.map_path) as f:
            records = json.load(f)

        roads = []

        for record in records:
            road = Road(record['x'], record['y'], record['img_file_name'])
            roads.append(road)
        
        self.roads = roads

    def populate_coins(self):
        coins = []

        for coin_position in self.curr_episode.coin_positions:
            coin = Coin(coin_position[1], coin_position[0])
            coins.append(coin)

        self.coins = coins
    
    def populate_hero(self):
        self.hero = Hero(0,0, self.on_episode_changed)

class Visualizer:
    def __init__(self, game=None):
        self.game = game

    def main_loop(self):
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            episode_done, episode_reward = self.game.play()
            if episode_done:
                self.game.gameOver(episode_reward)

    def intro(self, training, testing):
        pygame.display.set_caption('Grab the coin!')
        pygame.init()
        display_width, display_height = 432,432
        game_display = pygame.display.set_mode((display_width, display_height))
        fpsClock = pygame.time.Clock()
        
        intro = True
        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            width = game_display.get_width()
            height = game_display.get_height()

            land_image = pygame.image.load('static/land.png')
            land_image = pygame.transform.scale(land_image, (width//6, height//6))
            land_rect = land_image.get_rect()

            for i in np.linspace(0, width, num=7):
                for j in np.linspace(0, height, num=7):
                    land_rect.topleft = (i,j)
                    game_display.blit(land_image, land_rect)
            
            gameOverFont = pygame.font.Font('arcade.ttf', 36)
            gameOverSurf, gameOverRect = text_object('Grab The Coin!', gameOverFont, pygame.Color(120,120,120))
            gameOverRect.midtop = (width//2, height//2 - 120)
            game_display.blit(gameOverSurf, gameOverRect)
            
            player_sprite = AnimatedHero(game_display.get_width()//2, game_display.get_height()//2 - 20, hero_type='idle')

            player_sprite.update()
            player_sprite.draw(game_display)
            
            button(game_display,'train', width//8, height//8*7, width//6, height//8, pygame.Color(120,120,120), pygame.Color(150,150,150), training)
            button(game_display,'test', width//4*3, height//8*7, width//6, height//8, pygame.Color(120,120,120), pygame.Color(150,150,150), testing)
                        
            pygame.display.update()