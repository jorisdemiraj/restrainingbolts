#
# SpaceInvader game with reward automa for non-Markovian rewards
#
# Luca Iocchi 2017
#

import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

from SpaceInvader import *

np.set_printoptions(precision=3)

# Reward automa

class RewardAutoma(object):

    def __init__(self, alien_cols=0,alien_rows=0,nrstates=0, left_right=True): # alien_cols=0 -> RA disabled
        # RA states
        self.alien_cols = alien_cols
        self.alien_rows = alien_rows
        if (self.alien_rows>0):
            self.nRAstates = nrstates+2  # number of RA states
            self.RAGoal = self.nRAstates-2
            self.RAFail = self.nRAstates-1
        else: # RA disabled
            self.nRAstates = 2  # number of RA states
            self.RAGoal = 1 # never reached
            self.RAFail = 2 # never reached

        self.STATES = {
            'RAGoalStep':100,   # goal step of reward automa
            'RAGoal':1000,      # goal of reward automa
            'RAFail':-5,         # fail of reward automa
            'GoodAlien':10,      # good alien removed for next RA state
            'WrongAlien':1      # wrong alien removed for next RA state
        }

        self.left_right = left_right
        self.goalreached = 0 # number of RA goals reached for statistics
        self.visits = 0 # number of visits for each state
        self.success = 0 # number of good transitions for each state
        self.successrate=[]
        self.goal= False
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = []
        self.last_node = self.current_node
        self.lastalien=0
        self.countupdates = 0 # count state transitions (for the score)

    # check if a column is free (used by RA)
    def check_thresholds(self, alien):
        
        if alien.y> 180:
            return True
        return False

    # RewardAutoma Transition
    def update(self):
        reward = 0
        state_changed = False

        # RA disabled
        if (len(self.game.aliens)==0):
            return (reward, state_changed)

        f = []
        for c in self.game.aliens:
            if c not in f and self.check_thresholds(c):
                f.append(c)  # if alien is not in f and its under threshold we add it to f

       
        #check if the last alien removed is our alien (if we actually have an alien removed)
        if(self.game.last_alienremoved!=[] and self.game.last_alienremoved[len(self.game.last_alienremoved)-1]!=self.lastalien):
            self.lastalien=self.game.last_alienremoved[len(self.game.last_alienremoved)-1]
            self.visits+=1
            if self.lastalien in self.current_node :
                reward += self.STATES['GoodAlien']
                self.success+=1
                #f.remove(self.lastalien)
                #print ('Hit right alien for next RA state')
            elif (self.lastalien in f):
                reward += self.STATES['WrongAlien']
                self.success+=0.5
                #print ('Hit wrong alien for next RA state')
        #create a list of aliens under threshold
            elif(f!=[] and self.lastalien not in f ):
                reward += self.STATES['RAFail']
                


        result =  all(elem in f  for elem in self.current_node)
 
        if(not result):
            state_changed = True
            self.countupdates += 1   
            reward += self.STATES['RAGoal'] * self.countupdates / self.alien_rows
            #self.goal=True
            #print("  -- RA state transition to %d, " %(self.current_node))
            if (not f):
                # print("  <<< RA GOAL >>>")
                reward += self.STATES['RAGoal']
                self.goalreached += 1




        self.last_node = self.current_node
                  
                   
        #get closest alien (under threshold) to the shooter and make it current node
        #tempx=0
        #tempy=0
        distance=0
        mins=0
        samerows=[]
        #for alien in f:
        #    tempx= alien.x-self.shooter_x
        #    tempy=alien.y-self.shooter_y
        #    distance=math.sqrt(tempx**2+tempy**2)
        #    if(mins==0):
        #        mins=distance
        #    if(distance<mins):
        #        mins=distance
        #self.current_node=mins


        for alien in f:
            distance=alien.y
            if(distance>mins):
                mins=distance
                samerows=[]
            if(distance==mins):
                samerows.append(alien)
        self.current_node=samerows
            
        



        return (reward, state_changed)

    def current_successrate(self):
        self.successrate.append(float(self.success)/self.visits)
        return self.success/self.visits

    def print_successrate(self):
        
        print('RA success: %s' %str(self.successrate))



class SpaceInvaderSRA(SpaceInvaderS):

    def __init__(self, alien_rows=3, alien_cols=3, trainsessionname='test'):
        SpaceInvaderS.__init__(self,alien_rows, alien_cols, trainsessionname)
        self.RA = RewardAutoma(alien_cols, alien_rows,alien_cols*alien_rows, False)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':-1,
            'shooterNotMoving':0,
            'Scores':0,    # alien removed
            'Hit':0,       # shooter hit
            'Goal':0,      # level completed
        }
        self.RA_exploration_enabled = False  # Use options to speed-up learning process
        self.report_str = ''


    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.RA.visits, self.RA.success, self.agent.SA_failure, random.getstate(),np.random.get_state()]

         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        self.RA.visits = data[4]
        self.RA.success = data[5]
        if (len(data)>6):
            self.agent.SA_failure = data[6]
        if (len(data)>7):
            print('Set random generator state from file.')
            random.setstate(data[7])
            np.random.set_state(data[8])       


    def setStateActionSpace(self):
        super(SpaceInvaderSRA, self).setStateActionSpace()
        self.nstates *= self.RA.nRAstates
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getstate(self):
        x = super(SpaceInvaderSRA, self).getstate()
        if(self.RA.current_node!=[]):
            return x + random.choice(self.RA.current_node).x-self.shooter_x
        else: 
            return x

    def reset(self):
        super(SpaceInvaderSRA, self).reset()
        self.RA.reset()
        self.RA_exploration()

    def update(self, a):
        super(SpaceInvaderSRA, self).update(a)
        (RAr, state_changed) = self.RA.update()
        if (state_changed):
            print(self.current_reward)
            self.RA_exploration()
        self.current_reward += RAr
      
     
    def goal_reached(self):
        return self.RA.goal
       
    def getreward(self):
        r = self.current_reward
        #for b in self.last_alienremoved:
        #    if b.i == self.RA.current_node:
        #         r += self.STATES['GoodAlien']
                #print 'Hit right alien for next RA state'
        #if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
            #r *= (self.RA.current_node+1)
            #print "MAXI REWARD ",r
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma

        return r

    def print_report(self, printall=False):
        toprint = printall
        ch = ' '
        if (self.agent.optimal):
            ch = '*'
            toprint = True
            
        RAnode = len(self.RA.current_node)

        s = 'Iter %6d, b_hit: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d, mem: %d/%d  %c' %(self.iteration, self.score, self.shooter_hit_count,self.numactions, self.cumreward, RAnode, len(self.agent.Q), len(self.agent.SA_failure), ch)

        if self.score > self.hiscore:
            self.hiscore = self.score
            s += ' HISCORE '
            toprint = True
        if self.cumreward > self.hireward:
            self.hireward = self.cumreward
            s += ' HIREWARD '
            toprint = True

        if (toprint):
            print(s)

        self.cumreward100 += self.cumreward
        self.cumscore100 += RAnode
        numiter = 100
        if (self.iteration%numiter==0):
            #self.doSave()
            self.report_str = "%s %6d/%4d avg last 100: reward %.1f | RA %.2f | p goals %.1f %% <<<" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100/100), float(self.cumscore100)/100, float(self.RA.goalreached*100)/numiter)
            print('-----------------------------------------------------------------------')
            print(self.report_str)
            self.RA.print_successrate()
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.RA.goalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        #self.resfile.write("%d,%d,%d,%d,%d\n" % (RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.write("%d,%d,%d,%d,%d,%d,%d\n" % (self.iteration, self.elapsedtime, RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.flush()


    def RA_exploration(self):
        if not self.RA_exploration_enabled:
            return
        #print("RA state: ",self.RA.current_node)
        success_rate = max(min(self.RA.current_successrate(),0.9),0.1)
        #print("RA exploration policy: current state success rate ",success_rate)
        er = random.random()
        self.agent.option_enabled = (er<success_rate)
        #print("RA exploration policy: optimal ",self.agent.partialoptimal, "\n")



