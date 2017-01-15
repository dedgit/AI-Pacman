# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
  """
A base class for reflex agents that chooses score-maximizing actions
"""

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
Picks among the actions with the highest Q(s,a).
"""

    actions = gameState.getLegalActions(self.index)

    # print gameState
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):

    """
Finds the next successor which is a grid position (location tuple).
"""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()

    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
Computes a linear combination of features and feature weights
"""
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
Returns a counter of features for the state
"""
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
Normally, weights do not depend on the gamestate.  They can be either
a counter or a dictionary.
"""
    return {'successorScore': 1.0}

  def getMostDenseArea(self, gameState):
    ourFood = self.getFoodYouAreDefending(gameState).asList()
    distance = [self.getMazeDistance(gameState.getAgentPosition(self.index), a) for a in ourFood]
    nearestFood = ourFood[0]
    nearestDstance = distance[0]

    for i in range(len(distance)):
      if distance[i] < nearestDstance:
        nearestFood = ourFood[i]
        nearestDstance = distance[i]
    return nearestFood


class DefensiveReflexAgent(ReflexCaptureAgent):
  lastSuccess = 0
  flag = 1
  flag2 = 0
  currentFoods = []
  """
A reflex agent that keeps its side Pacman-free. Again,
this is to give you an idea of what a defensive agent
could be like.  It is not the best or only way to make
such an agent.
"""

  def getFeatures(self, gameState, action):
    self.start = self.getMostDenseArea(gameState)
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    if (self.flag2 == 0):
      self.flag2 = 1
      self.currentFoods = self.getFoodYouAreDefending(gameState).asList()
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      pos = [a.getPosition() for a in invaders]
      nearestPos = pos[0]
      nearestDst = dists[0]

      for i in range(len(dists)):
        if dists[i] < nearestDst:
          nearestPos = pos[i]
          nearestDst = dists[i]

      features['invaderPDistance'] = nearestDst
      # print len(self.currentFoods), len(self.getFoodYouAreDefending(gameState).asList())
      if (features['invaderDistance'] == 1 or features['invaderPDistance'] == 1 or features[
        'invaderLDistance'] == 1):
        print "here1"
        self.flag = 0
        self.lastSuccess = nearestPos
        features['invaderLDistance'] = self.getMazeDistance(myPos, self.lastSuccess)
        self.currentFoods = self.getFoodYouAreDefending(gameState).asList()
        # print "Got Him", self.lastSuccess , self.flag

      if (len(self.currentFoods) > len(self.getFoodYouAreDefending(gameState).asList())):
        print "here2"
        nextFoods = self.getFoodYouAreDefending(gameState).asList()
        print "Found Him"
        for i in range(len(self.currentFoods)):
          # print self.currentFoods[i][0], self.currentFoods[i][1], nextFoods[i][0], nextFoods[i][1]
          if (len(self.currentFoods) > 0 and len(nextFoods) > i):
            print "i: ", i, len(nextFoods), self.currentFoods[i][0], nextFoods[i][0], self.currentFoods[i][
              1], nextFoods[i][1]
            if (self.currentFoods[i][0] != nextFoods[i][0] or self.currentFoods[i][1] != nextFoods[i][1]):
              features['invaderPDistance'] = self.getMazeDistance(myPos, self.currentFoods[i])
              self.lastSuccess = self.currentFoods[i]
              self.currentFoods = nextFoods
              break

              # elif(self.flag==0):
              # print "here4"
              # features['invaderDistance']=self.getMazeDistance(myPos, self.lastSuccess)

              # elif(self.flag==1):
              # print "here3"
              # if(self.lastSuccess==0):
              # print "hii"
              # self.lastSuccess=self.getMostDenseArea(gameState)
              # features['invaderDistance']=self.getMazeDistance(myPos, self.lastSuccess)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'invaderPDistance': -20,
            'invaderLDistance': -5, 'stop': -100, 'reverse': -2}


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

