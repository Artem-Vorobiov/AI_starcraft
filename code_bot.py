import datetime
import pytz

global time_point
start_point = datetime.datetime.now().strftime("%S")
my_list = []

import sc2
from sc2 import run_game, maps, Race, Difficulty, bot_ai
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random



class Tim(sc2.BotAI):

	def __init__(self):
		self.ITERATIONS_PER_MINUTE 	= 165
		self.MAX_WORKERS 			= 50

	async def on_step(self, iteration):
		global time_changed
		time_changed = datetime.datetime.now().strftime("%S")
		self.iteration = iteration
		await self.distribute_workers()
		await self.build_workers()
		await self.build_pylon()
		await self.build_assimilator()
		await self.expand()
		await self.cacl()

	async def cacl(self):
		print(self.iteration)
		print('\n')
		if int(time_changed) == 15:
			my_list.append(self.iteration)
			print(self.iteration)
			print(my_list)
			print('\n\n\n\n\n')

	async def build_workers(self):
		print('Variable time_point: \t', start_point)
		print('self.TIME_NOW:       \t', time_changed)
		nexuses = self.units(NEXUS).ready
		if  len(self.units(PROBE)) <= self.MAX_WORKERS:
			for nexus in nexuses.noqueue:
				if self.can_afford(PROBE) and not self.already_pending(PROBE):
					await self.do(nexus.train(PROBE))


	async def build_pylon(self):
		nexuses = self.units(NEXUS).ready
		if self.supply_used >= self.supply_tim:
			for nexus in nexuses.noqueue:
				if self.can_afford(PYLON) and not self.already_pending(PYLON):
					await self.build(PYLON, near = nexus)

	async def build_assimilator(self):
		assimilator_num = self.units(ASSIMILATOR).amount
		nexuses = self.units(NEXUS).ready
		pls = self.units(PYLON).amount
		if assimilator_num < 3 and pls >= 1:
			for nexus in self.units(NEXUS).ready:
				vgs = self.state.vespene_geyser.closer_than(20.0, nexus)
				for vg in vgs:
					if not self.can_afford(ASSIMILATOR):
						break
					elif not self.already_pending(ASSIMILATOR):
						if not self.units(ASSIMILATOR).closer_than(1.0, vg).exists:
							worker = self.select_build_worker(vg.position)
							await self.do(worker.build(ASSIMILATOR, vg))

	async def expand(self):
# ДОБАВИТЬ если прошло меньше 10 минут то строй 3 базы, больше 10 минут то строй 6 баз.
		assimilator_num = self.units(ASSIMILATOR).amount
		nexuses = self.units(NEXUS).ready.amount
		if assimilator_num >= 1 and nexuses <= 3:
			if self.can_afford(NEXUS) and not self.already_pending(NEXUS):
				await self.expand_now()



run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, Tim()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)
