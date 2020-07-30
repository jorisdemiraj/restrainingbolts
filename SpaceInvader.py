

import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs


black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]
orange = [180,100,20]
red = [180,0,0]
alienColor = [180,217,147]
shooterColor=[227,185,34]

# game's constant variables

shooter_width = 10
shooter_height = 10

block_width = 10
block_height = 10
block_xdistance = 20
            
resolutionx = 20
resolutiony = 10



class Alien(object):

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.x = (block_width+block_xdistance)*i
        self.y = 70+(block_height+8)*j
        self.mover=1
        self.rect = pygame.Rect(self.x, self.y, block_width, block_height)

    def move(self):
        
        self.x+=self.mover
        print("moving")
        if(self.x==self.win_width):
            self.y+=1
            self.mover=-self.mover
        if(self.x==0):
            self.y+=1
            self.mover=-self.mover
        self.rect = pygame.Rect(self.x, self.y, block_width, block_height)


class SpaceInvader(object):

    def __init__(self, alien_rows=3, alien_cols=3, trainsessionname='test'):

        self.agent = None
        self.RA = None
        self.isAuto = True
        self.gui_visible = False
        self.sound_enabled = True
        self.fire_enabled = False
        self.userquit = False
        self.optimalPolicyUser = False  # optimal policy set by user
        self.evalBestPolicy = False
        self.trainsessionname = trainsessionname
        self.alien_rows = alien_rows
        self.alien_cols = alien_cols
        if (self.alien_cols<5):
            self.block_xdistance = 50

        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':0,
            'shooterNotMoving':0,
            'Scores':10,    # alien removed
            'Hit':0,        # shooter hit
            'Goal':100,     # level completed
        }
        
        # Configuration
        self.deterministic = True   # deterministic 
        self.simple_state = False   # simple = do not consider shooter x
        self.shooter_normal_bump = True  # only left/right bounces
        self.shooter_complex_bump = False  # straigth/left/right complex bounces
        self.pause = False # game is paused
        self.debug = False
        
        self.sleeptime = 0.0
 
        self.accy = 1.00
        self.score = 0
  
        self.alien_hit_count = 0
        self.shooter_hit_count = 0
        self.command = 0
        self.iteration = 0
        self.cumreward = 0
        self.cumscore100 = 0 # cumulative score for statistics
        self.cumreward100 = 0 # cumulative reward for statistics
        self.ngoalreached = 0 # number of goals reached for stats
        self.numactions = 0 # number of actions in this run
        self.mover=1
        self.dropper=0
        self.action_names = ['--','<-','->','o'] # stay, left, right, fire

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = 0 # 0 = not firing, <0 firing up
        
        self.hiscore = 0
        self.hireward = -1000000
        self.vscores = []
        self.resfile = open("data/"+self.trainsessionname +".dat","a+")
        self.elapsedtime = 0 # elapsed time for this experiment

        self.win_width = int((block_width+block_xdistance) * self.alien_cols + block_xdistance )
        self.win_height = 480

        pygame.init()
        pygame.display.set_caption('SpaceInvader')
        
        #allows for holding of key
        pygame.key.set_repeat(1,0)

        self.screen = pygame.display.set_mode([self.win_width,self.win_height])
        self.myfont = pygame.font.SysFont("Arial",  30)

        self.se_alien = None
        self.se_wall = None
        self.se_shooter = None

        
        
    def init(self, agent):  # init after creation (uses args set from cli)

        print('init ', self.sound_enabled)
        if (self.sound_enabled):
            self.se_alien = pygame.mixer.Sound('alien_hit.wav')
            print('self.se_alien loaded')
            self.se_wall = pygame.mixer.Sound('sound/wall_hit.wav')
            self.se_shooter = pygame.mixer.Sound('sound/shooter_hit.wav')
        if (not self.gui_visible):
            pygame.display.iconify()

        self.agent = agent
        self.setStateActionSpace()
        self.agent.init(self.nstates, self.nactions)
        self.agent.set_action_names(self.action_names)


    def setRandomSeed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.agent.SA_failure]
         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        try:
            self.agent.SA_failure = data[4]
        except:
            print('WARNING: Cannot load SA_failure data')
    
    def initAliens(self):
        self.aliens = []
        self.aliensgrid = np.zeros((self.alien_cols,self.alien_rows))
        for i in range(0,self.alien_cols):
            for j in range(0,self.alien_rows):
                temp = Alien(i,j)
                self.aliens.append(temp)
                self.aliensgrid[i][j]=1

        
    def reset(self):


        self.shooter_x = 0
        self.shooter_y = self.win_height-20
        self.shooter_speed = 10 # same as resolution
        #self.shooter_vec = 0
        self.com_vec = 0

        self.score = 0
  
        self.shooter_hit_count = 0
        self.alien_hit_count = 0
        self.cumreward = 0
        self.gamman = 1.0 # cumulative gamma over time

        self.shooter_hit_without_alien = 0
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        
        self.prev_state = None # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this run
        self.iteration += 1

        self.agent.optimal = self.optimalPolicyUser or (self.iteration%100)==0 # False #(random.random() < 0.5)  # choose greedy action selection for the entire episode
        
        self.initAliens()

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = -10 # 0 = not firing, <0 firing up




    def goal_reached(self):
        #print("you freeze it")
        return len(self.aliens) == 0
        
        
    def update(self, a):
        
        self.command = a

        self.prev_state = self.getstate() # remember previous state
        
        #print(" == Update start %d" %self.prev_state)
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        #print('self.current_reward = 0')
        self.numactions += 1
        self.last_alienremoved = []
        
        while (self.prev_state == self.getstate()):
            #print(self.prev_state, self.getstate())
            if (self.firstAction):
                self.current_reward += self.STATES['Init']
                self.firstAction = False
            
            if self.command == 0:  # not moving
                # do nothing
                self.current_reward += self.STATES['shooterNotMoving']
                pass
            elif self.command == 1:  # moving left
                self.shooter_x -= self.shooter_speed
            elif self.command == 2:  # moving right
                self.shooter_x += self.shooter_speed                

            if self.shooter_x < 0:
                self.shooter_x = 0
            if self.shooter_x > self.screen.get_width() - shooter_width:
                self.shooter_x = self.screen.get_width() - shooter_width

            if self.command == 3:  # fire
                if (self.fire_speedy==0):
                    self.fire_posx = self.shooter_x + shooter_width/2
                    self.fire_posy = self.shooter_y
                    self.fire_speedy = -10


            self.current_reward += self.STATES['Alive']
   
            #move the aliens
            self.dropper=0.02
            for alien in self.aliens:
                alien.x+=self.mover
                alien.y+=self.dropper
                if(alien.x>self.win_width-2 or alien.x<2):
                    
                    self.mover=-self.mover
            
                alien.rect = pygame.Rect(alien.x, alien.y, block_width, block_height)

            # firing
            if (self.fire_speedy < 0):
                #self.fire_posy =0
                self.fire_posy  += self.fire_speedy
                #print(self.fire_posy)
            self.hitDetect()

        #print(" ** Update end - state: %d prev: %d" %(self.getstate(),self.prev_state))

   
            
    def hitDetect(self):
        ##COLLISION DETECTION
        shooter_rect = pygame.Rect(self.shooter_x, self.shooter_y, shooter_width, shooter_height)

        fire_rect = pygame.Rect(self.fire_posx-1, self.fire_posy-1, 3, 3)

        # print("fire pos %d %d - spd %d" %(self.fire_posx, self.fire_posy, self.fire_speedy))

        # TERMINATION OF EPISODE
        if (not self.finished):
            end1= False
            for alien in self.aliens:
                
                if(alien.y > self.screen.get_height() - 10):
                    end1= True
                    break
                else: end1= False
            end2 = self.goal_reached()
            end3 = self.shooter_hit_without_alien == 100
            end3b = self.numactions > 1000 * self.alien_cols
            end4 = len(self.aliens) == 0
            if (end1 or end2 or end3 or end3b or end4):
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_wall.play()
                if (end1):    
                    print("dead")
                    self.current_reward += self.STATES['Dead']
                if (end2):
                    print("goal reached")
                    self.ngoalreached += 1
                    self.current_reward += self.STATES['Goal']
                    
                    self.initAliens()
                if (end4):
                    self.finished=True
                self.finished = True # game will be reset at the beginning of next iteration
                return 
        

 
        #for shooter
 
    
                
       
        #firing
        if (self.fire_posy < 5):
            #reset
           
            self.fire_posx = 0
            self.fire_posy = 0
            self.fire_speedy = 0

        for alien in self.aliens:
            if alien.rect.colliderect(fire_rect):
                #print ('alien hit with fire ',alien.i,alien.j)
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_alien.play()
                self.score = self.score + 1
                self.aliens.remove(alien)
                self.last_alienremoved.append(alien)
                self.aliensgrid[(alien.i,alien.j)] = 0
                self.current_reward += self.STATES['Scores']
                self.shooter_hit_without_alien = 0
                #print("aliens left: %d" %len(self.aliens))
                # reset firing
                self.fire_posx = 0
                self.fire_posy = 0
                self.fire_speedy = 0
                break

        if self.alien_hit_count > 0:
            
            self.alien_hit_count = 0


    def input(self):
        self.isPressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('pygame quit event')
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.command = 1
                    self.isPressed = True
                elif event.key == pygame.K_RIGHT:
                    self.command = 2
                    self.isPressed = True
                elif event.key == pygame.K_x: # Fire
                    self.command = 3
                    self.isPressed = True
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print("Game paused: %d" %self.pause)
                elif event.key == pygame.K_a:
                    self.isAuto = not self.isAuto
                elif event.key == pygame.K_s:
                    self.sleeptime = 1.0
                    #self.agent.debug = False
                elif event.key == pygame.K_d:
                    self.sleeptime = 0.07
                    #self.agent.debug = False
                elif event.key == pygame.K_f:
                    self.sleeptime = 0.005
                    self.agent.debug = False
                elif event.key == pygame.K_g:
                    self.sleeptime = 0.0
                    self.agent.debug = False
                elif event.key == pygame.K_o:
                    self.optimalPolicyUser = not self.optimalPolicyUser
                    print("Best policy: %d" %self.optimalPolicyUser)
                elif event.key == pygame.K_q:
                    self.userquit = True
                    print("User quit !!!")
                    
        if not self.isPressed:
            self.command = 0

        return True

    def getUserAction(self):
        return self.command

    def getreward(self):
        r = self.current_reward
        failed = self.RA is not None and self.RA.current_node==self.RA.RAFail  # FAIL RA state
        if (self.current_reward>0 and failed):  
            r = 0
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma
        return r


    def print_report(self, printall=False):
        toprint = printall
        ch = ' '
        if (self.agent.optimal):
            ch = '*'
            toprint = True
            
        s = 'Iter %6d, sc: %3d, p_hit: %3d, na: %4d, r: %5d  %c' %(self.iteration, self.score, self.shooter_hit_count,self.numactions, self.cumreward, ch)

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
        self.cumscore100 += self.score
        numiter = 10
        pgoal = 0
        if (self.iteration%numiter==0):
            #self.doSave()
            pgoal = float(self.ngoalreached*100)/numiter
            print('-----------------------------------------------------------------------')
            print("%s %6d/%4d avg last 100: reward %.1f | score %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100/100), float(self.cumscore100)/100, pgoal))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.ngoalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        self.resfile.write("%d,%d,%d,%d\n" % (self.score, self.cumreward, self.goal_reached(),self.numactions))
        self.resfile.flush()


    def draw(self):
        self.screen.fill(black)

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['white'])
        self.screen.blit(score_label, (20, 10))

        #count_label = self.myfont.render(str(self.shooter_hit_count), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (70, 10))

        #x = self.getstate()
        cmd = ' '
        if self.command==1:
            cmd = '<'
        elif self.command==2:
            cmd = '>'
        elif self.command==3:
            cmd = 'o'
        #s = '%d %s' %(x,cmd)
        s = '%s' %(cmd)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))
        #self.screen.blit(count_label, (160, 10))

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))
        if (self.agent.optimal):
            opt_label = self.myfont.render("Best", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-100, 10))
            
        for alien in self.aliens:
            pygame.draw.rect(self.screen,alienColor,alien.rect,0)
        pygame.draw.rect(self.screen, shooterColor, [self.shooter_x, self.shooter_y, shooter_width, shooter_height], 0)

        # print("fire %d %d %d" %(self.fire_posx, self.fire_posy,self.fire_speedy))
        if (self.fire_speedy<0):
            pygame.draw.rect(self.screen, red, [self.fire_posx, self.fire_posy, 5, 5], 0)

        pygame.display.update()


    def quit(self):
        self.resfile.close()
        pygame.quit()

    # To be implemented by sub-classes    
    def setStateActionSpace(self):
        print('ERROR: this function must be overwritten by subclasses')
        sys.exit(1)
        
    def getstate(self):
        print('ERROR: this function must be overwritten by subclasses')
        sys.exit(1)


#
# SpaceInvader with standard definition of states
#
class SpaceInvaderN(SpaceInvader):

    def __init__(self, alien_rows=3, alien_cols=3, trainsessionname='test'):
        SpaceInvader.__init__(self,alien_rows, alien_cols, trainsessionname)
        
    def setStateActionSpace(self):
        
        self.n_shooter_x = int(self.win_width/resolutionx)+1

        self.nactions = 3  # 0: not moving, 1: left, 2: right
        if (self.fire_enabled):
            self.nactions = 4  # 3: fire
        
        self.nstates = self.n_shooter_x
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)
 
    def getstate(self):
        #diff_shooter_alien = (int(self._x)-self.shooter_x+self.win_width)/resolution
        resx = resolutionx # highest resolution
        resy = resolutiony # highest resolution
          

        if self.simple_state:
            shooter_x = 0 
        else:
            shooter_x = int(self.shooter_x)/resx
        
        x = shooter_x
        
        return int(x)


#
# SpaceInvader with simplified definition of states
#

class SpaceInvaderS(SpaceInvader):

    def __init__(self, alien_rows=3, alien_cols=3, trainsessionname='test'):
        SpaceInvader.__init__(self,alien_rows, alien_cols, trainsessionname)

    def setStateActionSpace(self):
        self.n_diff_shooter_alien = int(2*self.win_width/resolutionx)+1

        self.nactions = 3  # 0: not moving, 1: left, 2: right
        if (self.fire_enabled):
            self.nactions = 4  # 3: fire
        
        self.nstates = self.n_diff_shooter_alien
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

        
    def getstate(self):
        resx = resolutionx 
        temp=0
        mins=0
        for alien in self.aliens:
                temp= alien.x-self.shooter_x
                if(mins==0):
                    mins=temp
                #if(temp<0):
                 #   temp=temp+10
                #elif (temp>0):
                #    temp=temp-10
                if(temp<mins):
                    mins=temp
        diff_shooter_alien = int((mins+self.win_width)/resx)
     
        if(diff_shooter_alien<-33): 
            
            self.reset()
          
       
        return diff_shooter_alien


