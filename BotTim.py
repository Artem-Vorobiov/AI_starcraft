import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random

class Tim(sc2.BotAI):

	def __init__(self):
		self.ITERATIONS_PER_MINUTE = 165
		self.MAX_WORKERS = 50

	async def on_step(self, iteration):
		self.iteration = iteration
		await self.distribute_workers()
		await self.build_workers()

	async def build_workers(self):
		if len(self.units(PROBE)) < self.MAX_WORKERS:
			for nexus in self.units(NEXUS).ready.noqueue:	# LOOK! FIX
			    if self.can_afford(PROBE):
			    	await self.do(nexus.train(PROBE))



run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, Tim()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)