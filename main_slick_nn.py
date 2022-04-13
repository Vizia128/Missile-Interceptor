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
fuel_0 = 1000
width, height = 1800, 1000  # screen dimensions
noise = 10
enemy_base = [0 + noise, 1 / 4 * width - noise]
ally_base = [3/4*width + noise, width - noise]
generation = 0
max_gen = 10000
max_stage = 0


class Missile:
    def __init__(self, x=(ally_base[0]+ally_base[1])//2, y=height-11, v_x=0, v_y=0, a_x=0, a_y=0):
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.a_x = a_x
        self.a_y = a_y

        self.launched = False
        self.destroyed = False
        self.success = False

        self.fuel = fuel_0
        self.fitness_current = 0
        self.fitness_total = 0
        self.atm_den = 1
        self.drag_on = True

    def reset(self, x=(ally_base[0]+ally_base[1])//2, y=height-11, v_x=0, v_y=-12, a_x=0, a_y=0,
              fuel=fuel_0, launched=False, drag=True):
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.a_x = a_x
        self.a_y = a_y

        self.launched = launched
        self.destroyed = False
        self.success = False

        self.fuel = fuel
        self.fitness_current = 0
        # self.fitness_total carries over
        self.atm_den = 1
        self.drag_on = drag

    def set_atm_den(self):
        self.atm_den = math.log2((height - self.y) / 128)

    def missile_thrust(self, a_x, a_y, thrust):
        if thrust <= 0 or self.fuel < 0:
            return 0, 0
        self.a_x = a_x * math.sqrt(1 - a_y * a_y / 2)
        self.a_y = a_y * math.sqrt(1 - a_x * a_x / 2)
        a = 1.4 * thrust * grav  # * (2 * fuel_0 - self.fuel) / 2 / fuel_0

        self.fuel -= thrust
        return self.a_x * a, self.a_y * a

    def aero_turn(self):
        if (self.v_x == 0 and self.v_y == 0) or self.drag_on is False:
            return 0, 0
        v_x_perp, v_y_perp = self.v_y, -self.v_x
        a_dot_v = self.a_x * v_x_perp + self.a_y * v_y_perp
        v_dot_v = v_x_perp**2 + v_y_perp**2
        scale = a_dot_v / v_dot_v

        return math.tanh(v_x_perp * scale) * self.atm_den, math.tanh(v_y_perp * scale) * self.atm_den

    def drag(self):
        if self.drag_on is False:
            return 0, 0
        v_norm = math.sqrt(self.v_x**2 + self.v_y**2)
        a_drag = self.atm_den * ((self.v_x/42)**2 + (self.v_y/42)**2) / v_norm
        return -a_drag * self.v_x, -a_drag * self.v_y

    def move_missile(self, a_x=0, a_y=0, thrust=0):
        if self.launched is False or self.destroyed is True:
            return

        a_x_thrust, a_y_thrust = self.missile_thrust(a_x, a_y, thrust)
        a_x_aero, a_y_aero = self.aero_turn()
        a_x_drag, a_y_drag = self.drag()

        self.a_x, self.a_y = a_x_thrust + a_x_aero + a_x_drag, a_y_thrust + a_y_aero + a_y_drag + grav

        self.x = self.x + self.v_x * t + self.a_x / 2 * t ** 2
        self.y = self.y + self.v_y * t + self.a_y / 2 * t ** 2
        self.v_x = self.v_x + self.a_x * t
        self.v_y = self.v_y + self.a_y * t


def get_distance(m1, m2):
    return math.sqrt((m1.x - m2.x) ** 2 + (m1.y - m2.y) ** 2)


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
            v_scale = 15 / (math.sqrt(f_mis.v_x**2 + f_mis.v_y**2)+1)
            pygame.draw.line(disp_win, pygame.Color(0, 0, 255), (f_mis.x, f_mis.y),
                             (f_mis.x + f_mis.v_x * v_scale, f_mis.y + f_mis.v_y * v_scale), 2)
            if f_mis.fuel < 0:
                pygame.draw.circle(disp_win, pygame.Color(255, 128, 0), (f_mis.x, f_mis.y), 2)
            if f_mis.success is True:
                pygame.draw.circle(disp_win, pygame.Color(0, 255, 0), (f_mis.x, f_mis.y), 4)

        elif f_mis.success is True and f_mis.fuel > 0:
            pygame.draw.line(disp_win, pygame.Color(0, 255, 0), (f_mis.x - 8, f_mis.y), (f_mis.x + 8, f_mis.y), 3)
            pygame.draw.line(disp_win, pygame.Color(0, 255, 0), (f_mis.x, f_mis.y - 8), (f_mis.x, f_mis.y + 8), 3)

        else:
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (f_mis.x - 4, f_mis.y), (f_mis.x + 4, f_mis.y), 1)
            pygame.draw.line(disp_win, pygame.Color(255, 64, 0), (f_mis.x, f_mis.y - 4), (f_mis.x, f_mis.y + 4), 1)

    pygame.display.update()


def fitness_func(e_missile, a_missile, distance, stage):
    if (a_missile.destroyed is True or a_missile.launched is False or (height - e_missile.y)) < 50 \
            or a_missile.x < width // 2:
        return a_missile.fitness_total + a_missile.fitness_current

    curr_fit = 750 / (distance + 75) - 0.5
    curr_fit *= 0.2 * max(a_missile.fuel, 0) / fuel_0 + 1

    if (height - a_missile.y) < 100:
        curr_fit = curr_fit * (height - a_missile.y) / 100

    if a_missile.fitness_current < curr_fit:
        a_missile.fitness_current = curr_fit

    return a_missile.fitness_total + a_missile.fitness_current


def set_stage(ally_missile, enemy_missile, stage):
    random.seed(generation + stage)
    global max_stage
    if stage > max_stage:
        max_stage = stage

    if stage < 2 and max_stage < 20:
        ally_missile.reset(x=(ally_base[0] + ally_base[1]) // 2)
        enemy_missile.reset(x=(ally_base[(generation + stage) % 2]), y=600, drag=False)
        return

    elif stage < 4 and max_stage < 20:
        ally_missile.reset(x=(ally_base[0] + ally_base[1]) // 2)

        x = random.uniform(ally_base[0], ally_base[1])
        y = random.uniform(height*(3/4), height*1/3)
        enemy_missile.reset(x=x, y=y, drag=False)
        return

    elif stage < 6 and max_stage < 20:
        ally_missile.reset(x=(ally_base[0] + ally_base[1]) // 2)

        x = random.uniform(ally_base[0], ally_base[1])
        y = random.uniform(height*(3//4), height*(1/3))
        enemy_missile.reset(x=x, y=y, launched=True, drag=False)
        return

    elif stage < 8 and max_stage < 20:
        ally_missile.reset(x=(ally_base[0] + ally_base[1]) // 2)

        x = random.uniform(ally_base[0], ally_base[1])
        y = random.uniform(height*(3//4), height*(1/3))
        v_x = random.gauss(0, 3)
        enemy_missile.reset(x=x, y=y, v_x=v_x, launched=True, drag=False)
        return

    fuel = fuel_0 - stage * 2
    ally_missile.reset(x=random.uniform(ally_base[0], ally_base[1]), fuel=fuel)

    x_enemy = random.uniform(enemy_base[0], enemy_base[1])
    x_enemy_target = random.uniform(ally_base[0], ally_base[1])
    x_range = x_enemy_target - x_enemy
    v_0 = math.sqrt(width * grav) * 1.1
    launch_angle = math.pi / 2 - 0.5 * math.asin(x_range * grav / v_0 ** 2)
    enemy_missile.reset(x=x_enemy, v_x=v_0*math.cos(launch_angle), v_y=-v_0*math.sin(launch_angle),
                        launched=True, drag=False)


def run(ally_missiles, enemy_missiles, max_missile_stages, stage):



def game(genomes, config):
    display_window = pygame.display.set_mode((width + 100, height))
    nets = []
    ge = []
    enemy_missile_list = []
    ally_missile_list = []
    max_stage_missile_list = []

    global generation
    generation += 1

    for _, g in genomes:
        # Create Neural Network
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        # Create genome for AI missile
        g.fitness = 0
        ge.append(g)

        # Create missiles
        ally_missile_list.append(Missile())
        enemy_missile_list.append(Missile())

        # Track maximum stage reached
        max_stage_missile_list.append(0)

    stage = 0
    while max(max_stage_missile_list) == stage or (generation > 64 and stage < 12):
        print("Stage: ", stage)
        time_steps = 0

        for i, (e_missile, a_missile) in enumerate(zip(enemy_missile_list, ally_missile_list)):
            a_missile.fitness_total += a_missile.fitness_current

            if max_stage_missile_list[i] == stage or (generation > 64 and stage < 12):
                set_stage(a_missile, e_missile, stage)

            else:
                a_missile.reset()
                a_missile.destroyed = True
                e_missile.reset()
                e_missile.destroyed = True
        stage += 1

        running = True
        while running:
            for x, (e_missile, a_missile) in enumerate(zip(enemy_missile_list, ally_missile_list)):
                # neural network control
                output = nets[x].activate((a_missile.x, a_missile.y,
                                           a_missile.v_x, a_missile.v_x,
                                           a_missile.a_x, a_missile.a_y,
                                           e_missile.x - a_missile.x, e_missile.y - a_missile.y,
                                           e_missile.v_x - a_missile.v_x, e_missile.v_y - a_missile.v_y))

                # Missile launch and movement
                if output[0] > 0.5:
                    a_missile.launched = True

                if a_missile.launched is True:
                    if output[3] < 0:
                        output[3] = 0
                    a_missile.move_missile(a_x=output[1], a_y=output[2], thrust=output[3])

                e_missile.move_missile()

                # collision detection
                distance = get_distance(a_missile, e_missile)
                if e_missile.y > height - 20 and e_missile.x > width // 2:
                    e_missile.destroyed = True
                if a_missile.y > height or (a_missile.v_y > 0 and e_missile.y + 12 < a_missile.y):
                    a_missile.destroyed = True
                if distance < max((32 - stage), 4) and (height - a_missile.y) > 80 and a_missile.destroyed is False \
                        and a_missile.x > width // 2:
                    if a_missile.success is False:
                        a_missile.success = True
                        max_stage_missile_list[x] += 1
                    elif distance < 5:
                        e_missile.destroyed = True
                        a_missile.destroyed = True
                        a_missile.fitness_total += 1

                # fitness function
                ge[x].fitness = fitness_func(e_missile, a_missile, distance, stage)

            time_steps += 1
            if time_steps >= 2500:
                break

            if time_steps % 32 == 1:
                render(display_window, enemy_missile_list, ally_missile_list)

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
    winner = p.run(game, max_gen)
    pickle.dump(winner, open("save.p", "wb"))

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_slick.txt")
    run(config_path)