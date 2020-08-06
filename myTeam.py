# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
			   first = 'TopAgent', second = 'BottomAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class OffenseAgent(CaptureAgent):
	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
		CaptureAgent.registerInitialState(self, gameState)
		self.Y = 0.0
		self.defenseTimer = 0.0
		self.numReturned = 0.0

	def __init__(self, index):
		CaptureAgent.__init__(self, index)
		self.weights = util.Counter()
		self.weights['nextScore'] = 160
		self.weights['weightedDistance'] = -5
		self.weights['ghostDistance'] = 2.5
		self.weights['powerPelletValue'] = 80
		self.weights['chase'] = -100
		self.weights['stop'] = -1000
		self.weights['retreat'] = -1
		print(self.weights)
		self.keys = ['nextScore', 'weightedDistance', 'ghostDistance', 'powerPelletValue', 'chase', 'stop', 'retreat']

	def scoreDelta(self, gameState):
		if self.red:
			return gameState.getScore()
		else:
			return -1 * gameState.getScore()


	def evaluate(self, state, action):
		total = 0
		weights = self.getWeights()
		features = self.getFeatures(state, action)
		# print(features)
		# print(weights, features)
		for feature in self.keys:
			total += features[feature] * weights[feature]
		return total

	def chooseAction(self, gameState):
		actions = gameState.getLegalActions(self.index)
		values = [self.evaluate(gameState, a) for a in actions]
		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		return random.choice(bestActions)

	def getSuccessor(self, gameState, action):
		"""
		Finds the next successor which is a grid position (location tuple).
		"""
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def getWeights(self):
		return self.weights

	def weightedDistance(self, pos, food):
		return self.getMazeDistance(pos, food) + abs(food[1] - self.Y)

	def nearestEnemy(self, myPos, enemies):
		if len(enemies) > 0:
			distance, closestEnemy = min([(self.getMazeDistance(myPos, enemy.getPosition()), enemy.getPosition()) for enemy in enemies])
			return distance
		return 0

	def shouldRunHome(self, gameState):
		delta = self.scoreDelta(gameState)
		numCarrying = gameState.getAgentState(self.index).numCarrying
		return (gameState.data.timeleft < 80.0
			and delta <= 0
			and numCarrying > 0
			and numCarrying >= abs(delta))


	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
		foodList = self.getFood(successor).asList()
		features['nextScore'] = -len(foodList)

		if myState.numReturned != self.numReturned:
			self.defenseTimer = 100.0
			self.numReturned = myState.numReturned
		if self.defenseTimer > 0:
			self.defenseTimer -= 1
			features['chaseEnemyValue'] *= 100

		if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
			features['chaseEnemyValue'] *= 100

		if len(foodList) > 0:
			features['weightedDistance'] = min([self.weightedDistance(myPos, food) for food in foodList])

		if action == Directions.STOP:
			features['stop'] = 1

		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
		nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
		scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

		features['powerPelletValue'] = self.getPowerPelletValue(myPos, successor, scaredGhosts)
		features['chase'] = self.getChaseEnemyWeight(myPos, enemyPacmen)
		features['ghostDistance'] = self.nearestEnemy(myPos, nonScaredGhosts)
		features['retreat'] = self.getCashInValue(myPos, gameState, myState) + self.getBackToStartDistance(myPos, features['ghostDistance'])

		if self.shouldRunHome(gameState):
			features['retreat'] = self.getMazeDistance(self.start, myPos) * 10000

		return features

	def getPowerPelletValue(self, myPos, successor, scaredGhosts):
		powerPellets = self.getCapsules(successor)
		minDistance = 0
		if len(powerPellets) > 0 and len(scaredGhosts) == 0:
			distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
			minDistance = min(distances)
		return max(5 - minDistance, 0)

	def getCashInValue(self, myPos, gameState, myState):
		if myState.numCarrying >= 4:
			return self.getMazeDistance(self.start, myPos)
		return 0

	def getBackToStartDistance(self, myPos, smallestGhostPosition):
		if smallestGhostPosition > 5 or smallestGhostPosition == 0:
			return 0
		return self.getMazeDistance(self.start, myPos) * 1000

	def getChaseEnemyWeight(self, myPos, enemyPacmen):
		if len(enemyPacmen) > 0:
			dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
			return min(dists)
		return 0

class TopAgent(OffenseAgent):
	def registerInitialState(self, gameState):
		OffenseAgent.registerInitialState(self, gameState)
		self.Y = gameState.data.layout.height

class BottomAgent(OffenseAgent):
	def registerInitialState(self, gameState):
		OffenseAgent.registerInitialState(self, gameState)
