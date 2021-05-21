import pygame
import numpy as np

def button(playSurface, msg, x, y, w, h, inactive, active, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if (x+w > mouse[0] > x) and (y+h > mouse[1] > y):
        pygame.draw.rect(playSurface, active, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(playSurface, inactive, (x, y, w, h))
    
    smallText = pygame.font.Font('arcade.ttf', 14)
    textSurf, textRect = text_object(msg, smallText, pygame.Color(0,0,0))
    textRect.center = (x+w//2, y+h//2)
    playSurface.blit(textSurf, textRect)

def text_object(text, font, color):
	textSurface = font.render(text, True, color)
	return textSurface, textSurface.get_rect()

def message_display(display, text, display_width, display_height):
	largeText = pygame.font.Font('arcade.ttf', 115)
	TextSurf, TextRect = text_object(text, largeText)
	TextRect.center = (display_width//2, display_height//2)
	display.blit(TextSurf, TextRect)
	pygame.display.flip()