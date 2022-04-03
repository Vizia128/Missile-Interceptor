import random
import sys
import time
import math
import numpy as np
import pygame
from pygame.locals import *
import pygad
import threading

g = 1   # gravity
t = 0.02    # time scale
f = 2500    # fuel
width, height = 1800, 1000  # screen dimensions


class Missile:
    def __init__(self, x, y, v_x, v_y):
        self.x = x
        self.y = y
        self.velocity_x = v_x
        self.velocity_y = v_y

        self.acceleration = 0
        self.acceleration_x = 0
        self.acceleration_y = 0

        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False

        self.launch = False
        self.destroyed = False
        self.fuel = f
        self.fitness = 0

    def get_fitness(self, distance):
        curr_fit = 100 / (distance + 1) + 1

        if self.y < 100 and self.fitness < 1:
            self.fitness = self.y / 100

        elif self.fitness < curr_fit:
            self.fitness = curr_fit

    def move_missile(self):
        if self.launch is False or self.destroyed is True:
            return

        if not self.up_pressed and not self.down_pressed:
            self.acceleration_y = 0
        elif self.up_pressed:
            self.acceleration_y = -2 * g
            self.fuel -= 1
        elif self.down_pressed:
            self.acceleration_y = 2 * g
            self.fuel -= 1

        if not self.left_pressed and not self.right_pressed:
            self.acceleration_x = 0
        elif self.left_pressed:
            self.acceleration_x = -2 * g
            self.fuel -= 1
        elif self.right_pressed:
            self.acceleration_x = 2 * g
            self.fuel -= 1

        if self.fuel < 0:
            self.acceleration_x = 0
            self.acceleration_y = 0

        self.x = self.x + self.velocity_x * t + self.acceleration_x / 2 * t**2
        self.y = self.y + self.velocity_y * t + (self.acceleration_y + g) / 2 * t**2
        self.velocity_x = self.velocity_x + self.acceleration_x * t
        self.velocity_y = self.velocity_y + (self.acceleration_y + g) * t


def get_distance(m1, m2):
    return math.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)


def render(disp_win, enemy_missiles, friend_missiles):
    disp_win.fill(pygame.Color(0, 0, 0))
    pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (0, height), (width / 3, height), 20)
    pygame.draw.line(disp_win, pygame.Color(0, 128, 255), (width * 2 / 3, height), (width, height), 20)

    for mis in enemy_missiles:
        if mis.destroyed is False:
            pygame.draw.circle(disp_win, pygame.Color(255, 0, 0), (mis.x, mis.y), 5)
        else:
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (mis.x - 4, mis.y - 4), (mis.x + 4, mis.y + 4), 2)
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (mis.x - 4, mis.y + 4), (mis.x + 4, mis.y - 4), 2)

    for mis in friend_missiles:
        if mis.destroyed is False:
            pygame.draw.circle(disp_win, pygame.Color(0, 0, 255), (mis.x, mis.y), 5)
        else:
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (mis.x - 5, mis.y), (mis.x + 5, mis.y), 2)
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (mis.x, mis.y - 5), (mis.x, mis.y + 5), 2)

    pygame.display.update()


def game():
    pygame.init()
    display_window = pygame.display.set_mode((width, height))
    enemy_missile_list = []
    friend_missile_list = []
    time_steps = 0

    for i in range(1):
        x = random.uniform(0, width / 3)
        x_enemy_target = random.uniform(width * 2 / 3, width)
        x_range = x_enemy_target - x
        v_0 = math.sqrt(width * g) * 1.1
        launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * g / v_0 ** 2)
        enemy_missile_list.append(Missile(x, height - 11, v_0 * math.cos(launch_angle), -v_0 * math.sin(launch_angle)))

    for i in range(1):
        x = random.uniform(width * 2 / 3, width)
        friend_missile_list.append(Missile(x, height - 11, 0, 0))

    while True:

        for e_missile, f_missile in zip(enemy_missile_list, friend_missile_list):
            if time_steps == 1:
                e_missile.launch = True

            e_missile.move_missile()
            f_missile.move_missile()

            distance = get_distance(e_missile, f_missile)
            f_missile.get_fitness(distance)

            if e_missile.y > height - 10:
                e_missile.destroyed = True
            if f_missile.y > height - 10:
                f_missile.destroyed = True
            if distance < 5:
                e_missile.destroyed = True
                f_missile.destroyed = True

        render(display_window, enemy_missile_list, friend_missile_list)
        time_steps += 1

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    friend_missile_list[0].up_pressed = True
                elif event.key == pygame.K_DOWN:
                    friend_missile_list[0].down_pressed = True
                elif event.key == pygame.K_LEFT:
                    friend_missile_list[0].left_pressed = True
                elif event.key == pygame.K_RIGHT:
                    friend_missile_list[0].right_pressed = True
                elif event.key == pygame.K_SPACE:
                    friend_missile_list[0].launch = True

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    friend_missile_list[0].up_pressed = False
                elif event.key == pygame.K_DOWN:
                    friend_missile_list[0].down_pressed = False
                elif event.key == pygame.K_LEFT:
                    friend_missile_list[0].left_pressed = False
                elif event.key == pygame.K_RIGHT:
                    friend_missile_list[0].right_pressed = False


if __name__ == '__main__':
    game()


