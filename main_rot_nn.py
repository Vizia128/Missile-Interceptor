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
fuel = 1000
width, height = 1800, 1000  # screen dimensions
noise = 10
enemy_base = [0 + noise, 1 / 4 * width - noise]
ally_base = [3/4*width + noise, width - noise]
generation = 0
max_gen = 10000


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
        self.turned = 0
        self.destroyed = False
        self.success = False
        self.max_round = 0

        self.fuel = f
        self.fuel_0 = f
        self.fitness = 0

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
        self.turned = 0
        self.destroyed = False
        self.success = False
        self.fuel = f
        self.fuel_0 = f

    def turn_missile(self, direction):
        self.acceleration_ang += math.pi * direction / 64
        self.acceleration_ang %= 2 * math.pi
        self.turned += 1

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
    # return abs(m1.x - m2.x) + abs(m1.y - m2.y)


def render(disp_win, enemy_missiles, friend_missiles):
    disp_win.fill(pygame.Color(0, 0, 0))
    pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (enemy_base[0], height), (enemy_base[1], height), 20)
    pygame.draw.line(disp_win, pygame.Color(0, 128, 255), (ally_base[0], height), (ally_base[1], height), 20)

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


def fitness_func(f_missile, distance, stage):
    if f_missile.destroyed is True or (height - f_missile.y) < 50 or f_missile.destroyed is True:
        return f_missile.fitness

    curr_fit = 750 / (distance + 75) - 0.5

    if f_missile.fitness < curr_fit + stage * 10:
        f_missile.fitness = curr_fit + stage * 10

    return f_missile.fitness


def set_stage(e_missile, f_missile, stage):
    if stage <= 10:
        if stage % 2 == 1:
            x_e = (enemy_base[0] + enemy_base[1]) / 2
            x_t = (ally_base[0] + ally_base[1]) / 2 + (ally_base[0] - ally_base[1]) * stage / 20
            x_f = (ally_base[0] + ally_base[1]) / 2
        else:
            x_e = (enemy_base[0] + enemy_base[1]) / 2
            x_t = (ally_base[0] + ally_base[1]) / 2 - (ally_base[0] - ally_base[1]) * stage / 20
            x_f = (ally_base[0] + ally_base[1]) / 2
    elif stage <= 20:
        if stage % 2 == 1:
            x_e = (enemy_base[0] + enemy_base[1]) / 2 + (enemy_base[0] - enemy_base[1]) * (stage - 10) / 20
            x_t = random.uniform(ally_base[0], ally_base[1])
            x_f = (ally_base[0] + ally_base[1]) / 2
        else:
            x_e = (enemy_base[0] + enemy_base[1]) / 2 - (enemy_base[0] - enemy_base[1]) * (stage - 10) / 20
            x_t = random.uniform(ally_base[0], ally_base[1])
            x_f = (ally_base[0] + ally_base[1]) / 2
    elif stage <= 30:
        if stage % 2 == 1:
            x_e = random.uniform(enemy_base[0], enemy_base[1])
            x_t = random.uniform(ally_base[0], ally_base[1])
            x_f = (ally_base[0] + ally_base[1]) / 2 + (ally_base[0] - ally_base[1]) * (stage - 20) / 20
        else:
            x_e = random.uniform(enemy_base[0], enemy_base[1])
            x_t = random.uniform(ally_base[0], ally_base[1])
            x_f = (ally_base[0] + ally_base[1]) / 2 - (ally_base[0] - ally_base[1]) * (stage - 20) / 20

    else:
        x_e = random.uniform(enemy_base[0], enemy_base[1])
        x_t = random.uniform(ally_base[0], ally_base[1])

        x_f = random.uniform(ally_base[0], ally_base[1])

    x_range = x_t - x_e
    v_0 = math.sqrt(width * grav) * 1.1
    launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * grav / v_0 ** 2)
    e_missile.reset(x_e, height - 11, v_0 * math.cos(launch_angle), -v_0 * math.sin(launch_angle), 0, 1)

    f_missile.reset(x_f, height - 11, 0, -1, 1.1 * grav, fuel)
    f_missile.fitness = 10 * stage


def game(genomes, config):
    display_window = pygame.display.set_mode((width, height))
    nets = []
    ge = []
    enemy_missile_list = []
    friend_missile_list = []
    max_stage_missile_list = []
    reset_num = 0

    global generation
    generation += 1

    for _, g in genomes:
        # Create Neural Network
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        # Create enemy missile
        x = (enemy_base[0] + enemy_base[1]) / 2
        x_enemy_target = (ally_base[0] + ally_base[1]) / 2
        x_range = x_enemy_target - x
        v_0 = math.sqrt(width * grav) * 1.1
        launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * grav / v_0 ** 2)
        enemy_missile_list.append(Missile(x, height - 11, v_0 * math.cos(launch_angle),
                                          -v_0 * math.sin(launch_angle), 0, 1))

        # Create friendly AI missile
        x = (ally_base[0] + ally_base[1]) / 2
        friend_missile_list.append(Missile(x, height - 11, 0, -1, 1.1 * grav, fuel))

        # Track maximum stage reached
        max_stage_missile_list.append(0)

        # Create genome for AI missile
        g.fitness = 0
        ge.append(g)

    stage = 0
    while max(max_stage_missile_list) == stage:
        time_steps = 0

        running = True
        while running:

            for x, (e_missile, f_missile) in enumerate(zip(enemy_missile_list, friend_missile_list)):
                if time_steps == 1:
                    e_missile.launch = True

                e_missile.move_missile()
                f_missile.move_missile()

                distance = get_distance(e_missile, f_missile)
                ge[x].fitness = fitness_func(f_missile, distance, stage)

                # neural network control
                output = nets[x].activate((f_missile.x, f_missile.y, f_missile.velocity_x, f_missile.velocity_y,
                                           e_missile.x - f_missile.x, e_missile.y - f_missile.y,
                                           e_missile.velocity_x - f_missile.velocity_x,
                                           e_missile.velocity_y - f_missile.velocity_y,
                                           f_missile.fuel))

                if f_missile.launch:
                    f_missile.turn_missile(output[0])

                if output[1] > 0.5:
                    friend_missile_list[x].launch = True

                # collision detection
                if e_missile.y > height - 10:
                    e_missile.destroyed = True
                if f_missile.y > height - 10 or f_missile.fuel == -100:
                    f_missile.destroyed = True
                if distance < (25 - stage / 5) and (height - f_missile.y) > 80 and f_missile.destroyed is False \
                        and f_missile.turned > 12:
                    e_missile.destroyed = True
                    f_missile.destroyed = True
                    f_missile.success = True
                    max_stage_missile_list[x] += 1

            if time_steps >= 2500:
                break

            if time_steps % 32 == 1:
                render(display_window, enemy_missile_list, friend_missile_list)
            time_steps += 1

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

        stage += 1
        for i, (e_missile, f_missile) in enumerate(zip(enemy_missile_list, friend_missile_list)):
            if max_stage_missile_list[i] == stage:
                set_stage(e_missile, f_missile, stage)

            else:
                e_missile.reset(0,0,0,0,0,0)
                e_missile.destroyed = True
                f_missile.reset(0,0,0,0,0,0)
                f_missile.destroyed = True


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
    winner = p.run(game, max_gen)
    pickle.dump(winner, open("save.p", "wb"))

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_rot.txt")
    run(config_path)
