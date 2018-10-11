import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2 import position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, SCV, DRONE, ROBOTICSFACILITY, OBSERVER
import random
import cv2
import numpy as np
import os
import time
import math
import keras

#os.environ["SC2PATH"] = '/starcraftstuff/StarCraftII/'
HEADLESS = False

class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.use_model = use_model

        ###############################
        # DICT {UNIT_ID:LOCATION}
        # every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}
        ###############################
        self.train_data = []
        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("gameout-random-vs-medium.txt","a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

    async def on_step(self, iteration):

        self.time = (self.state.game_loop/22.4) / 60
        print('Time:',self.time)

        ###############################
        await self.build_scout()
        await self.scout()
        ###############################

        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        ###############################
        #  FIXED THIS
        x += random.randrange(-5,5)
        y += random.randrange(-5,5)
        ###############################

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    ###############################
    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.time/4):
            print('\n\t My Observers = {}, My Time/4 = {}'.format(len(self.units(OBSERVER)), math.floor(self.time/3)))
            #   My Observers = 0, My Time/4 = 1
            #   My Observers = 1, My Time/3 = 2
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                print(len(self.units(OBSERVER)), self.time/4)               # 1 , 2.186011904761905
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
                    print('\n self.do', self.do(rf.train(OBSERVER)))        # <coroutine object BotAI.do at 0x1424c13b8>
                    print('\n self.do', type(self.do(rf.train(OBSERVER))))  # <class 'coroutine'>


    async def scout(self):

        # {DISTANCE_TO_ENEMY_START:EXPANSIONLOC}
        # Словарь где первый элемент расстояние до объекта, второй элемент это Кортеж(координаты x and y)
        # 
        self.expand_dis_dir = {}

        #   self.expansion_locations - it's build-in function inside bot_ai.py
        #   It's => """List of possible expansion locations."""
        #   
        for el in self.expansion_locations:             #   {(x,y) : {'sc2.unit.Unit'}}. Dict = { Keys(Tuple) : Values(Set) }

            #   self.expansion_locations -- это метод предоставленный разработчиками. Его суть - найти на карте минералы или газ
            #   Метод возращает Словарь, В этом словаре Ключ(Key) - это координаты(x, y) Ресурса, Значение(Value) - это 
            #   В этом словаре Ключ(Key)  это расстояние др
            W  = self.expansion_locations .values()     
            WW = list(W)
            # print('\n Expansion Locations: {}'.format(self.expansion_locations))
            # print('\n Извлекаю Values', WW)             #   
            #   {(34,26): {Unit(name='MineralField', tag=8790147075), ... tag=8787525635)}, ...
            #   ... (166, 18): {Unit(name='MineralField', tag=8831565825), .. Unit(name='MineralField750', tag=8807120898)},
            # print('\n Тип Values', type(WW))            #   <class 'list'>

            # print('\n Извлекаю Values [0]', WW[0])      #   {Unit(name='MineralField750', tag=8800960515), ... Unit(name='MineralField', tag=8802402307)}
            # print('\n Тип Values [0]', type(WW[0]))     #   <class 'set'>

            WWW = list(WW[0])
            # print('\n Трансформировали Сет в Лист. Извлекаем первый элемент листа', WWW[0])     #   Unit(name='VespeneGeyser', tag=8796569603)
            # print('\n Тип Values [0]', type(WWW[0]))                                            #   <class 'sc2.unit.Unit'>

            # print('\n EL ----> : {}'.format(el))
            #   EL ----> : (34, 126)
            #   EL ----> : (166, 18)

            #   Расстояние от Базы противника до Возможных Мест Дислокации около ресурсов
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            # print('\n \tdistance_to_enemy_start : {}'.format(distance_to_enemy_start))
            #   distance_to_enemy_start : 5.70087712549569
            #   distance_to_enemy_start : 164.85296478983932

            self.expand_dis_dir[distance_to_enemy_start] = el
            # print('\n expand_dis_dir  ----> : {}'.format(self.expand_dis_dir))
            #   expand_dis_dir  ----> : {5.70087712549569: (34, 126)}
            #   expand_dis_dir  ----> : {5.70087712549569: (34, 126), 133.91228472399385: (127, 22), ... 164.85296478983932: (166, 18)}

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)
        #   Получаем как результат: List => [ 5.70087712549569, 133.91228472399385, ... , 164.85296478983932 ]

        existing_ids = [unit.tag for unit in self.units]
        # print('\n existing_ids: {}'.format(existing_ids))               #   [4346347521, 4346609665, .....  4347920385, 4347658241, 4346871809]
        # print('\n TYPE OF existing_ids: {}'.format(type(existing_ids))) #   <class 'list'>
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        # print('\n REMOVE: {}'.format(to_be_removed))                    #   
        # print('\n TYPE OF REMOVE: {}'.format(type(to_be_removed)))      #   <class 'list'>

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]
        # end removing of scouts that are dead now.

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            # print('\n Я внутри ORDEREDdist:', dist)                             #   5.70087712549569, 24.50510150968569
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # print('\n Got Location, that should be (x,y):', location)       #   (166, 18), (161, 46)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]
                                # print('\n active_locations = that loop through scouts_and_spots', active_locations)     #   [], [(166, 18)] - из-за next()

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    # print('\n This line is last line where I put Coordinates in scouts_and_spots\
                                    #     \n self.scouts_and_spots[obs.tag] = location', scouts_and_spots)
                                    break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

        print('What and Where => {}'.format(self.scouts_and_spots))     #   {4372037633: (34, 126), 4399562754: (39, 98), 4356046863: (36, 94)} 
        print('Type => {}'.format(type(self.scouts_and_spots)))         #   <class 'dict'>

    async def intel(self):

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        draw_dict = {
                     NEXUS: [15, (0, 255, 0)],
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],
                     #VOIDRAY: [3, (255, 100, 0)],
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        # from Александр Тимофеев via YT
        main_base_names = ['nexus', 'commandcenter', 'orbitalcommand', 'planetaryfortress', 'hatchery']
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

        for vr in self.units(VOIDRAY).ready:
            pos = vr.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (255, 100, 0), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not HEADLESS:
            if self.use_model:
                cv2.imshow('Model Intel', resized)
                cv2.waitKey(1)
            else:
                cv2.imshow('Random Intel', resized)
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
        try:
            if self.units(NEXUS).amount < self.time/2 and self.can_afford(NEXUS) and self.units(NEXUS).amount < 4:
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def offensive_force_buildings(self):
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
                if len(self.units(STARGATE)) < 2:
                    print('\n\t BUILDING FIRST and SECOND STARGATES ...')
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)
                elif len(self.units(STARGATE)) < 4 and self.minerals >= 500:
                    print('\n\t BUILDING SECOND STARGATE ...')
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

        if len(self.units(VOIDRAY).idle) > 2:

            target = False
            if self.time > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])
                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no attack
                    wait = random.randrange(7, 100)/100
                    self.do_something_after = self.time + wait

                elif choice == 1:
                    #attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))

                elif choice == 2:
                    #attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    #attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))

                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y, self.flipped])

run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot(use_model=True)),
    Computer(Race.Protoss, Difficulty.Medium),
    ], realtime=False)