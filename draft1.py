# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:47:11 2019

@author: wbyse
"""
import pandas as pd 
import numpy as np


#create a player dictionary
#map player to index of the column vector
#playbyplay_Data = pd.read_csv('/Users/baiyangwang/Box Sync/academics/maching learning/final-project/NBA-PBP-2018-2019.csv')
player_Data = pd.read_csv('/Users/baiyangwang/Box Sync/academics/maching learning/final-project/Players.csv')
game_Data = pd.read_csv('/Users/baiyangwang/Box Sync/academics/maching learning/final-project/nba.games.stats.csv')
num_players = len(player_Data.keys())
Synergy = np.zeros([num_players,num_players])
Antisynergy = np.zeros([num_players,num_players])

#create player name to index dictionary
player_ind = {}
ind = 0
for i in player_Data.keys():
    player_ind[i] = ind
    ind += 1

#load (x,y)
#game has chronical order and Y shows score differential, X plus is team 1 payer
#X minus is team 2 player
game_results = game_Data['TeamPoints']-game_Data['OpponentPoints']
Teams = [game_Data['Team'],game_Data['Opponent']]
Dates = game_Data['Date']
inds = Dates.argsort()
Teams = Teams[inds]
game_results = game_results[inds]
Dates = np.sort(Dates)

#get rid of duplicate games
i = 0
while i < len(game_results):
    sameDay = np.where(Dates == Dates[i])
    for j in sameDay:
        if Teams[j][0] == Teams[i][1] and Teams[j][1] == Teams[i][0] and game_results[i] == -1*game_results[j]:
            np.delete(game_results, j)
            np.delete(Teams, j)
            np.delete(Dates, j)
    i += 1
    

#create a class for linear regression
class NBA_LR:
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
    def predict(self, lineup):
        
        