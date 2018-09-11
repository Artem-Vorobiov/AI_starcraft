import sc2
from sc2 import run_game, maps, Race, Difficulty, bot_ai
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random

#		===   ЗАДАЧИ   ===
# 1. Описать в редакторе и на листе
# 2. Вновь воссоздать бота

class Tim(sc2.BotAI):

# 		Вопросы для решения 
#   1. Моя задача - ОТОБРАЖАТЬ ВРЕМЯ через принт
#   2. Моя задача выяснить сколько итерация в минуту используя пункт 1
#   3. Задача воссоздать оставшиеся функции

	def __init__(self):
		self.ITERATIONS_PER_MINUTE = 165
		self.MAX_WORKERS = 50

	async def on_step(self, iteration):
		self.iteration = iteration
		await self.distribute_workers()
		await self.build_workers()
		await self.build_pylon()
		await self.build_assimilator()
		await self.expand() 

	async def build_workers(self):
		if len(self.units(PROBE)) < self.MAX_WORKERS:
			for nexus in self.units(NEXUS).ready.noqueue:	# LOOK! FIX
			    if self.can_afford(PROBE):
				    await self.do(nexus.train(PROBE))

	async def build_pylon(self):
		if self.supply_used >= self.supply_tim:
			# print('\n')
			# print(self.supply_cap)
			# print(self.supply_tim)
			# print(self.supply_left)
			# print('\n')
			nexus = self.units(NEXUS).ready
			if self.can_afford(PYLON) and not self.already_pending(PYLON):
				await self.build(PYLON, near=nexus.first)

	async def expand(self):
		count = 0
		for asmltr in self.units(ASSIMILATOR).ready:
			count += 1
			# print('\n')
			# print('ASSIMILATOR :  ---- >')
			# print(count)
			if count >= 1 and self.can_afford(NEXUS) and self.units(NEXUS).amount <=2 :
				await self.expand_now()

	async def build_assimilator(self):
		# Сколько задействовано рабочих, сколько освоено гейзеров,
		# count = 0 
		# for pl in self.units(PYLON).ready:
		# 	count += 1 
		# 	print('\n')
		# 	print('Pylons :  --- >')
		# 	print(count)
		pl = self.units(PYLON).ready
		print(type(pl))
		print('\n')
		count = len(pl)
		print(count)
		if count >= 2:
			for nexus in self.units(NEXUS).ready:
				vgs = self.state.vespene_geyser.closer_than(20.0, nexus)
				for vg in vgs:
					if not self.can_afford(ASSIMILATOR):
						break
					elif not self.already_pending(ASSIMILATOR):
						if not self.units(ASSIMILATOR).closer_than(1.0, vg).exists:
							worker = self.select_build_worker(vg.position)
							await self.do(worker.build(ASSIMILATOR, vg))



run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, Tim()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)