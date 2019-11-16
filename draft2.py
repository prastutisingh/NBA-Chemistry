# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:47:11 2019

@author: wbyse
"""
import pandas as pd 
import numpy as np
import os

curr_directory = os.getcwd()

# Player Roster Information (Copied over from player_roster.ipynb)
teams = ['BOS','BRK','NYK','PHI','TOR','CHI','CLE','DET','IND','MIL','ATL','CHO','MIA','ORL','WAS',
         'DEN','MIN','OKC','POR','UTA','GSW','LAC','LAL','PHO','SAC','DAL','HOU','MEM','NOP','SAS']
        
# Dictionary of roster
# Ex. The roster of Boston Celtics players for the 2019-2020 season can be accessed using roster['BOS']['2019']
# It does not include any players/rookies for which there is no season data
roster = {}
    
for team in teams: 
    roster[team] = {}

# Initialize set for list of all players (with no repeats)
all_players = set()
    
for filename in os.listdir(os.path.join(curr_directory, 'data_sets/player_roster')):
    data = pd.read_csv(os.path.join('data_sets/player_roster', filename))
    year = filename[0:4]
    
    for team in teams:
        roster[team][year] = []
        
        players = data.loc[data['Tm'] == team]
        for ind in players.index: 
            player_name = players['Player'][ind].split('\\', 1)[0]
            if player_name not in roster[team][year]: 
                roster[team][year].append(player_name)
            
        all_players.update(roster[team][year])

# Player dictionary that maps all players to index
player_index = dict(zip(list(all_players), range(len(all_players))))

# playbyplay_Data = pd.read_csv('/Users/baiyangwang/Box Sync/academics/maching learning/final-project/NBA-PBP-2018-2019.csv')

### What is the first column being loaded here? 
game_data = pd.read_csv(os.path.join(curr_directory,'data_sets/nba.games.stats.csv'))

# Sort all values by the Date
game_data = game_data.sort_values(by=['Date'])

# game has chronical order and Y shows score differential, X plus is team 1 payer
# X minus is team 2 player
game_results = np.array(list(game_data['TeamPoints'] - game_data['OpponentPoints']))
teams = np.array(list(zip(game_data.Team, game_data.Opponent)))
dates = np.array(list(game_data['Date']))

unique_dates = list(set(dates))

# Makes an index of all games that are repeated
repeat_indexes = []

for date in unique_dates: 
    same_day = np.where(dates == date)
    # suppose same_day = [0, 1, 2, 3, 4, 5]
    for i in same_day[0]: 
        # start with i = 0
        for j in same_day[0]: 
            # j = 0, 1, 2, 3, 4, 5
            if j > i: 
                if np.array_equal(np.flip(teams[j], axis=0) , teams[i]): 
                    repeat_indexes.append(j)

# Make new unique game results, teams and dates arrays
unique_game_results = game_results[repeat_indexes]
unique_teams = teams[repeat_indexes]
unique_dates = dates[repeat_indexes]

# Initialize synergy and anti-synergy parameters
synergy = np.zeros([num_players,num_players])
antisynergy = np.zeros([num_players,num_players])

# create a class for linear regression
class LinearRegression:
    def __init__(self, synergy, antisynergy, step_size=0.01, max_iter=1000000, eps=1e-5,
                  verbose=True):
        
        self.synergy = synergy
        self.antisynergy = antisynergy
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps

    def predict(self, teams, date):
        x, z = x_for_game(teams, date)
        team_1_S = np.matmul(np.transpose(x), np.matmul(self.synergy, x))
        team_2_S = np.matmul(np.transpose(z), np.matmul(self.synergy, z))
        cross_term = np.matmul(np.transpose(x), np.matmul(self.antisynergy, z))
        
        return team_1_S - team_2_S + 2*cross_term
        
    def x_for_game(self, teams, date): 
        x_1 = np.zeros(num_players)
        x_2 = np.zeros(num_players)

        if int(date[5:7]) < 9: 
            year = str(int(date[0:4]) - 1)
        else: 
            year = date[0:4]

        game_players = roster[team][year]
        for item in game_players: 
            x_1[player_index[item]] = 1
                
        for team in teams: 
            game_players = roster[team][year]
            for item in game_players: 
                x_2[player_index[item]] = -1

        return x_1, x_2
    
    def gradLossFunction(theta, x, y):
        update = 0
        for i in range(x.shape[0]):
            x = np.matrix(x[i,:])
            y = np.matrix(y[i])
            theta = np.matrix(theta)   
            update += x*x.T*theta*x*x.T + x*y*x.T
            
        return update
        
    
    def fit(self, x, y):
        
        update = None
        while update == None or update >= self.eps:
            update = self.step_size*LinearRegression.gradLossFuction(self.theta, x, y)
            self.theta = self.theta - update
            


            
        