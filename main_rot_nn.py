import os
import random
import sys
import time
import math
import numpy as np
import pygame
from pygame.locals import *
import neat
import pickle

pygame.init()

grav = 1  # gravity
t = 0.04  # time scale
width, height = 1800, 1000  # screen dimensions


class Missile:
    def __init__(self, x, y, v_x, v_y, acc, f):
        self.x = x
        self.y = y
        self.velocity_x = v_x
        self.velocity_y = v_y

        self.acceleration_ang = math.pi / 2
        self.acceleration = acc
        self.acceleration_x = 0
        self.acceleration_y = 0

        self.launch = False
        self.destroyed = False
        self.success = False
        self.fuel = f
        self.fuel_0 = f
        self.fitness = 0
        self.total_fitness = 0

    def reset(self, x, y, v_x, v_y, acc, f):
        self.x = x
        self.y = y
        self.velocity_x = v_x
        self.velocity_y = v_y

        self.acceleration_ang = math.pi / 2
        self.acceleration = acc
        self.acceleration_x = 0
        self.acceleration_y = 0

        self.launch = False
        self.destroyed = False
        self.success = False
        self.fuel = f
        self.fuel_0 = f
        self.total_fitness += self.fitness
        self.fitness = 0

    def turn_missile(self, direction):
        self.acceleration_ang += math.pi * direction / 64
        self.acceleration_ang %= 2 * math.pi

    def move_missile(self):
        if self.launch is False or self.destroyed is True:
            return

        self.fuel -= 1
        if self.fuel <= 0:
            self.acceleration = 0
            self.acceleration_x = 0
            self.acceleration_y = 0

        else:
            self.acceleration_x = math.cos(self.acceleration_ang) * self.acceleration * \
                                  (1 + 1.4 * (self.fuel_0 - self.fuel) / self.fuel_0)

            self.acceleration_y = -math.sin(self.acceleration_ang) * self.acceleration * \
                                  (1 + 1.4 * (self.fuel_0 - self.fuel) / self.fuel_0)

        self.x = self.x + self.velocity_x * t + self.acceleration_x / 2 * t ** 2
        self.y = self.y + self.velocity_y * t + (self.acceleration_y + grav) / 2 * t ** 2
        self.velocity_x = self.velocity_x + self.acceleration_x * t
        self.velocity_y = self.velocity_y + (self.acceleration_y + grav) * t


def get_distance(m1, m2):
    return math.sqrt((m1.x - m2.x) ** 2 + (m1.y - m2.y) ** 2)


def render(disp_win, enemy_missiles, friend_missiles):
    disp_win.fill(pygame.Color(0, 0, 0))
    pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (0, height), (width / 3, height), 20)
    pygame.draw.line(disp_win, pygame.Color(0, 128, 255), (width * 2 / 3, height), (width, height), 20)

    for e_mis, f_mis in zip(enemy_missiles, friend_missiles):
        if e_mis.destroyed is False:
            pygame.draw.circle(disp_win, pygame.Color(255, 0, 0), (e_mis.x, e_mis.y), 5)
        else:
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (e_mis.x - 4, e_mis.y - 4), (e_mis.x + 4, e_mis.y + 4), 2)
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (e_mis.x - 4, e_mis.y + 4), (e_mis.x + 4, e_mis.y - 4), 2)

        if f_mis.destroyed is False:
            pygame.draw.circle(disp_win, pygame.Color(0, 0, 255), (f_mis.x, f_mis.y), 5)
            pygame.draw.line(disp_win, pygame.Color(0, 0, 255), (f_mis.x, f_mis.y),
                             (f_mis.x + 10 * math.cos(f_mis.acceleration_ang),
                              f_mis.y - 10 * math.sin(f_mis.acceleration_ang)), 2)
            if f_mis.fuel == 0:
                pygame.draw.circle(disp_win, pygame.Color(255, 128, 0), (f_mis.x, f_mis.y), 2)

        elif f_mis.success is False:
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (f_mis.x - 4, f_mis.y), (f_mis.x + 4, f_mis.y), 1)
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (f_mis.x, f_mis.y - 4), (f_mis.x, f_mis.y + 4), 1)

        else:
            pygame.draw.line(disp_win, pygame.Color(0, 255, 0), (f_mis.x - 8, f_mis.y), (f_mis.x + 8, f_mis.y), 3)
            pygame.draw.line(disp_win, pygame.Color(0, 255, 0), (f_mis.x, f_mis.y - 8), (f_mis.x, f_mis.y + 8), 3)

    pygame.display.update()


def fitness_func(f_missile, distance):
    if f_missile.destroyed is True:
        return f_missile.fitness

    if (height - f_missile.y - 10) < 100 or -f_missile.velocity_y <= -2 or f_missile.destroyed is True:
        curr_fit = 0
    else:
        curr_fit = 5 * (2 ** (-distance / 128) + 2 ** (-distance / 32) + 2 ** (-distance / 8))

    if f_missile.fitness <= curr_fit:
        f_missile.fitness = curr_fit
    else:
        f_missile.destroyed = True

    return f_missile.fitness


def game(genomes, config):
    display_window = pygame.display.set_mode((width, height))
    nets = []
    ge = []
    enemy_missile_list = []
    friend_missile_list = []
    time_steps = 0
    reset_num = 0

    for _, g in genomes:
        # Create Neural Network
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        # Create enemy missile
        x = random.uniform(0, width / 3)
        x_enemy_target = random.uniform(width * 2 / 3, width)
        x_range = x_enemy_target - x
        v_0 = math.sqrt(width * grav) * 1.1
        launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * grav / v_0 ** 2)
        enemy_missile_list.append(Missile(x, height - 11, v_0 * math.cos(launch_angle),
                                          -v_0 * math.sin(launch_angle), 0, 1))

        # Create friendly AI missile
        x = random.uniform(width * 2 / 3, width)
        friend_missile_list.append(Missile(x, height - 11, 0, -1, 1.1 * grav, 1000))

        # Create genome for AI missile
        g.fitness = 0
        ge.append(g)

    running = True
    while running:

        for x, (e_missile, f_missile) in enumerate(zip(enemy_missile_list, friend_missile_list)):
            if time_steps == 1:
                e_missile.launch = True

            e_missile.move_missile()
            f_missile.move_missile()

            distance = get_distance(e_missile, f_missile)
            ge[x].fitness = fitness_func(f_missile, distance)

            # neural network control
            output = nets[x].activate((f_missile.x, f_missile.y, f_missile.velocity_x, f_missile.velocity_y,
                                       e_missile.x, e_missile.y, e_missile.velocity_x, e_missile.velocity_y,
                                       f_missile.fuel))

            if f_missile.launch:
                if (output[0] > 0) and (output[1] < 0):
                    f_missile.turn_missile(1)

                elif (output[0] < 0) and (output[1] > 0):
                    f_missile.turn_missile(-1)

            if output[2] > 0.5:
                friend_missile_list[x].launch = True

            # collision detection
            if e_missile.y > height - 10:
                e_missile.destroyed = True
            if f_missile.y > height - 10:
                f_missile.destroyed = True
            if distance < 5 and (height - e_missile.y) < 100 and f_missile.destroyed is False:
                e_missile.destroyed = True
                f_missile.destroyed = True
                f_missile.success = True
                f_missile.fitness += 5

        # if time_steps >= 2500:
        #     for e_missile, f_missile in zip(enemy_missile_list, friend_missile_list):
        #         # Reset enemy missile
        #         x = random.uniform(0, width / 3)
        #         x_enemy_target = random.uniform(width * 2 / 3, width)
        #         x_range = x_enemy_target - x
        #         v_0 = math.sqrt(width * grav) * 1.1
        #         launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * grav / v_0 ** 2)
        #         e_missile.reset(x, height - 11, v_0 * math.cos(launch_angle), -v_0 * math.sin(launch_angle), 0, 1)
        #
        #         # Reset friendly AI missile
        #         f_missile.reset(5 / 6 * width, height - 11, 0, -1, 1.1 * grav, 1000)

        if time_steps >= 2500:
            break

        if time_steps % 32 == 1:
            render(display_window, enemy_missile_list, friend_missile_list)
        time_steps += 1

        # if reset_num > 12:
        #     break

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    neat.checkpoint.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix='neat_checkpoint')

    # Run for up to 1000 generations.
    winner = p.run(game, 1000)
    pickle.dump(winner, open("save.p", "wb"))

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_rot.txt")
    run(config_path)
