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
import random, time, util, sys
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DefensiveReflexAgent'):
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
  # return [eval(first)(firstIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

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


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()

    successor = self.getSuccessor(gameState, action)

    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    action = random.choice(actions)

    return action


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
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


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    start = time.time()
    nearestFood = self.nearestFood(gameState)
    nearestEnemy = self.getNearestEnemy(gameState)
    escapepath = self.escapePath(gameState, 36, 17, nearestEnemy)  # TODO
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, nearestFood, nearestEnemy, escapepath, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # if self.index == 0:
    #     if gameState.getAgentPosition(self.index)[0] == 1:
    #         if Directions.NORTH in actions:
    #             return Directions.NORTH
    #         else:
    #             return Directions.EAST

    '''
    You should change this in your own agent.
    '''
    action = random.choice(bestActions)
    # return Directions.STOP
    return action

  def escapePath(self, game_state, width, height, enemy):

    stack = util.Stack()
    myState = game_state.getAgentState(self.index)
    myPos = myState.getPosition()

    visited = []
    if len(enemy) > 0:
      visited = [[enemy[0][0], enemy[0][1]]]
    stack.push([myPos[0], myPos[1]])
    path = []

    while not stack.isEmpty():

      myPos = [int(myPos[0]), int(myPos[1] + 0.5)]
      psize = len(visited)
      loop = []

      right = [myPos[0] - 1, myPos[1]]
      if right[0] >= 0 and right not in visited and not game_state.hasWall(right[0], right[1]):
        stack.push(right)
        visited.append(right)
      if right in visited: loop.append(right)

      up = [myPos[0], myPos[1] + 1]
      if up[1] < height and up not in visited and not game_state.hasWall(up[0], up[1]):
        stack.push(up)
        visited.append(up)
      if up in visited:  loop.append(up)

      down = [myPos[0], myPos[1] - 1]
      if down[1] >= 0 and down not in visited and not game_state.hasWall(down[0], down[1]):
        stack.push(down)
        visited.append(down)
      if down in visited: loop.append(down)

      left = [myPos[0] + 1, myPos[1]]
      if left[0] < width and left not in visited and not game_state.hasWall(left[0], left[1]):
        stack.push(left)
        visited.append(left)
      if left in visited: loop.append(left)

      if len(loop) > 0:
        for i in reversed(path):
          if abs(i[0] - myPos[0]) + abs(i[1] - myPos[1]) > 1:
            path.remove(i)
          else:
            break

      if myPos[0] == 1 and myPos[1] == 2:
        # self.debugDraw(path, [1.0, 1.0, 1.0], True)
        # print path
        return path

      myPos = stack.pop()

      for i in reversed(path):
        if abs(i[0] - myPos[0]) + abs(i[1] - myPos[1]) > 1:
          path.remove(i)
        else:
          break
      path.append(myPos)
    return path

  def nearestFood(self, gameState):

    food = self.getFood(gameState).asList()
    distance = [self.getMazeDistance(gameState.getAgentPosition(self.index), a) for a in food]

    if len(food) < 1:
      previous = self.getFoodYouAreDefending(gameState).asList()[0]
      return [previous, self.getMazeDistance(gameState.getAgentPosition(self.index), previous)]
    nearestFood = food[0]
    nearestDstance = distance[0]

    for i in range(len(distance)):
      if distance[i] < nearestDstance:
        nearestFood = food[i]
        nearestDstance = distance[i]

    return [nearestFood, nearestDstance]

  def getNearestEnemy(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes distance to invaders we can see
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

    if len(invaders) == 0:
      return []
    else:
      nearestEnemy = invaders[0].getPosition()
      nearestDstance = dists[0]
      for i in range(len(dists)):
        if dists[i] < nearestDstance:
          nearestEnemy = invaders[i].getPosition()
          nearestDstance = dists[i]
    return [nearestEnemy, nearestDstance]

  def evaluate(self, gameState, nearestFood, nearestEnemy, escapepath, action):

    score = 0
    next = gameState.generateSuccessor(self.index, action)
    # nextpos = next.getAgentPosition(self.index)
    nextscore = next.getScore()
    if nextscore > gameState.getScore():
      score += 5

    if self.getMazeDistance(next.getAgentPosition(self.index), nearestFood[0]) < nearestFood[1]:
      score += 1

    if len(nearestEnemy) > 0:
      if next.getAgentState(self.index).isPacman:
        if self.getMazeDistance(next.getAgentPosition(self.index), nearestEnemy[0]) < nearestEnemy[1]:
          score -= 2

        nextActions = next.getLegalActions(self.index)
        if len(nextActions) == 2:
          score -= 100
    # if [nextpos[0], nextpos[1]] in escapepath:
    #     score += 20
    if next.getAgentState(self.index).isPacman and action == Directions.STOP:
      score = -10

    return score
