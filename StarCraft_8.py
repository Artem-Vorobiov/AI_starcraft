import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
import random
import cv2
import numpy as np


#   1.   Вывести через PRINT() переменные и тип переменной
#   2.   Изучить новые функции - где берутся и как работают
#   3.   Выписать логику в тетрадь

class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

    def random_location_variance(self, enemy_start_location):
        #   Цель найти координаты ВРАГА и отправить СКАУТА!

        #   Определили начальные координаты ВРАГА
        x = enemy_start_location[0]                             
        print('\n x: \n', x)                                #    161.5      
        print('\n type(x): \n', type(x))                    #   <class 'float'>
        y = enemy_start_location[1]
        print('\n y: \n', y)                                #   21.5
        print('\n type(y): \n', type(y))                    #   <class 'float'> 

        #   Мы не можем посылать разведчика прямо в центр базы ВРАГА, по этому
        #   Задаем отклонение X  и  Y
        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        print('\n x: \n', x)                                #   142.12
        print('\n type(x): \n', type(x))                    #   <class 'float'> 
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]
        print('\n y: \n', y)                                #   17.63
        print('\n type(y): \n', type(y))                    #   <class 'float'> 

        #   Координаты с учетом отклонения могут уйти за пределы игровой карты, для этого ставим проверки
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        #   Используем полученные координаты с учетом проверок и отправляем СКАУТА к базе врага
        go_to = position.Point2(position.Pointlike((x,y)))
        print('\n go_to: \n', go_to)                        #   (142.12, 17.63)
        print('\n type(go_to): \n', type(go_to))            #   <class 'sc2.position.Point2'>
        return go_to

    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]                 #   ТАКИМ ОБРАЗОМ МОЖНО ВЫБРАТЬ ОДИН ЮНИТ
            if scout.is_idle:

                enemy_location = self.enemy_start_locations[0]
                # print('\n enemy_location: \n', enemy_location)                #    (161.5, 21.5)               
                # print('\n type(enemy_location): \n', type(enemy_location))    #    <class 'sc2.position.Point2'>

                move_to = self.random_location_variance(enemy_location)
                # print('\n move_to: \n', move_to)                            #    (161.5, 21.93)               
                # print('\n type(move_to): \n', type(move_to))                #    <class 'sc2.position.Point2'>
                print(move_to)                                                #    (161.5, 21.93)
                await self.do(scout.move(move_to))
                # print('\n self.do(scout.move(move_to)): \n', self.do(scout.move(move_to)))              #   <coroutine object BotAI.do at 0x1198560f8>              
                # print('\n type(self.do(scout.move(move_to))): \n', type(self.do(scout.move(move_to))))  #   <class 'coroutine'>

        else:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
                    # print('\n self.do(rf.train(OBSERVER)): \n', self.do(rf.train(OBSERVER)))                # <coroutine object BotAI.do at 0x1198591a8>          
                    # print('\n type(self.do(rf.train(OBSERVER))): \n', type(self.do(rf.train(OBSERVER))))    # <class 'coroutine'>

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY'''

        #   Присвоили цвета каждому из юнитов
        draw_dict = {
                     NEXUS: [15, (0, 255, 0)],
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],

                     VOIDRAY: [3, (255, 100, 0)],
                     #OBSERVER: [3, (255, 255, 255)],
        # print('\n draw_dict: \n', draw_dict)
#   {<UnitTypeId.NEXUS: 59>: [15, (0, 255, 0)], <UnitTypeId.PYLON: 60>: [3, (20, 235, 0)], <UnitTypeId.PROBE: 84>: [1, (55, 200, 0)],
#   <UnitTypeId.ASSIMILATOR: 61>: [2, (55, 200, 0)], <UnitTypeId.GATEWAY: 62>: [3, (200, 100, 0)], <UnitTypeId.CYBERNETICSCORE: 72>: [3, (150, 150, 0)], 
#   <UnitTypeId.STARGATE: 67>: [5, (255, 0, 0)], <UnitTypeId.ROBOTICSFACILITY: 71>: [5, (215, 155, 0)], <UnitTypeId.VOIDRAY: 80>: [3, (255, 100, 0)]}
        # print('\n type(draw_dict): \n', type(draw_dict))    #   <class 'dict'>
                    }


        #   Перебираем юнитов и выводич их на карту в виду кругов разного цвета
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:

                # print('\n unit: \n', unit)                #     Unit(name='CyberneticsCore', tag=4359192577)
                #   ИЛИ  Unit(name='RoboticsFacility', tag=4361814017)      ИЛИ      Unit(name='Assimilator', tag=4364173313)
                #   ИЛИ  Unit(name='Probe', tag=4353163265)                 ИЛИ      Unit(name='Pylon', tag=4360503297)              
                # print('\n type(unit): \n', type(unit))    #    <class 'sc2.unit.Unit'>

                pos = unit.position
                # print('\n pos: \n', pos)                #   For first unit in loop =  (161.5, 21.5)                
                # print('\n type(pos): \n', type(pos))    #    <class 'sc2.position.Point2'>
                #   cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)


        #   Разделяю базу противника на 2 цвета. Один цвет соотвествует заданному списку, другой все остальное.
        main_base_names = ["nexus", "supplydepot", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
            #   cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)


        #   Разделяем вражеских юнитов на 2 части - если рабочие один цвет, все другие другой цвет
        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            # print('\n obs: \n', obs)                #    Unit(name='Observer', tag=4369416193)               
            # print('\n type(obs): \n', type(obs))    #    <class 'sc2.unit.Unit'>
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # flip horizontally to make our final fix in visual representation:
        flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)

        cv2.imshow('Intel', resized)
        cv2.waitKey(1)


    async def build_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        #print(self.iteration / self.ITERATIONS_PER_MINUTE)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {VOIDRAY: [8, 3]}


        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))


run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Hard)
    ], realtime=False)