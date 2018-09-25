import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
import random
import cv2
import numpy as np
import time


#   1.   Вывести через PRINT() переменные и тип переменной      + Выполнил
#   2.   Изучить новые функции - где берутся и как работают
#   3.   Выписать логику в тетрадь
#   4.   Смотри какие изменения были внесены в  sc2

#os.environ["SC2PATH"] = '/starcraftstuff/StarCraftII/'

HEADLESS = False


class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.train_data = []


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))



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
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))

        else:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY'''
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
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "supplydepot", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

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
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)


        line_max = 50                               #   Максимальная длина линии на КАРТЕ
        mineral_ratio = self.minerals / 1500        #   Максимальное значение = 1
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500         #   Максимальное значение = 1
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap   #   Максимальное значение = 1
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left)     #   Процент военной техники ко всей технике
        if military_weight > 1.0:
            military_weight = 1.0


        #   cv.Line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) 
        a = cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (350, 25, 100), 3)  # worker/supply ratio -- ЖЕЛТЫЙ
        # print('\n cv2.line: \n', a)                #     массив с данными               
        # print('\n type(cv2.line): \n', type(a))    #     <class 'numpy.ndarray'>
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)  --  БЕЖЕВЫЙ
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (0, 100, 100), 3)  # population ratio (supply_left/supply) -- СЕРЫЙ
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500. -- ОРАНЖЕВЫЙ
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500. -- САЛАТНЕВЫЙ

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
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
        if len(self.units(VOIDRAY).idle) > 0:
            choice = random.randrange(0, 4)
            target = False
            #   Первая итерация (iteration) 1 > 0 (do_something_after); Заходим в 0 == 0, do_something_after = 1 + допустим 30 = 31
            #   Вторая итерация (iteration) 2 > 31 (do_something_after), не верно условие уходим на след итерацию
            #   Третья итрерация iteration) 3 > 31 (do_something_after), не верно условие уходим на след итерацию
            #   Это все работа с цифрами
            if self.iteration > self.do_something_after:
                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    #attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        #   Выбираем врага ближайщего к Центру
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
                        # print('\n choice-1: \n', target)                #     Unit(name='CommandCenter', tag=8874229807)  или   Unit(name='SCV', tag=4351852545)
                        # print('\n choice-1(target): \n', type(target))    #   <class 'sc2.unit.Unit'>

                elif choice == 2:
                    #attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                        # print('\n choice -- 2: \n', target)                #     Unit(name='Refinery', tag=4356046849)
                        #   Unit(name='BarracksTechLab', tag=8858173487)   and   <class 'sc2.unit.Unit'>
                        #   Unit(name='Barracks', tag=8871215111)          and   <class 'sc2.unit.Unit'>
                        # print('\n choice--2(target): \n', type(target))    #     <class 'sc2.unit.Unit'>

                elif choice == 3:
                    #attack_enemy_start
                    target = self.enemy_start_locations[0]
                    # print('\n choice -- 3: \n', target)                #     (161.5, 21.5)
                    # print('\n choice--3 (target): \n', type(target))   #     <class 'sc2.position.Point2'>

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))

                        # print('\n self.do(vr.attack(target)) : \n', \
                        # self.do(vr.attack(target)))                # <coroutine object BotAI.do at 0x1192e92b0>
                        # print('\n TYPE (self.do(vr.attack(target))): \n',\
                        # type(self.do(vr.attack(target))))          # <class 'coroutine'>

                y = np.zeros(4)
                y[choice] = 1
                print(y)
                # print('\n self.flipped: \n', self.flipped)               # массив - это наша картинка с кругами
                # print('\n TYPE(self.flipped): \n', type(self.flipped))   # <class 'numpy.ndarray'>
                self.train_data.append([y,self.flipped])
                print('\nself.train_data.append([y,self.flipped]): \n', \
                self.train_data)               # Лист который включает [[ array([0,1,0,0]),array([[ [],[],[],[] ]]) ],[array([1., 0., 0., 0.]), array([[[  0,   0,   0] ...]
                print('\n TYPE(self.train_data.append([y,self.flipped])): \n', \
                type(self.train_data))         # <class 'list'>

run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime=False)