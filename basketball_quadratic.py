import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import time

curr_directory = os.getcwd()

def preprocess_data():
    # Game data from 2014 - 2015 season to 2017-2018 season
    game_data = pd.read_csv(os.path.join(curr_directory, 'data_sets/nba.games.stats.csv'))

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
                    if np.array_equal(np.flip(teams[j], axis=0), teams[i]):
                        repeat_indexes.append(j)

    # Make new unique game results, teams and dates arrays
    unique_game_results = game_results[repeat_indexes]
    unique_teams = teams[repeat_indexes]
    unique_dates = dates[repeat_indexes]

    # Game data from the 2018-2019 season and the 2019-2020 season
    game_data_2018 = pd.read_csv(os.path.join(curr_directory, 'data_sets/game_data_2018_2019.csv'))
    game_data_2019 = pd.read_csv(os.path.join(curr_directory, 'data_sets/game_data_2019_2020.csv'))

    # Strip the day of week abbreviation from Date
    game_data_2018['Date'] = game_data_2018['Date'].str[4:]
    game_data_2019['Date'] = game_data_2019['Date'].str[4:]

    # Date conversion functions
    def monthToNum(shortMonth):
        return {
            'Jan': '01',
            'Feb': '02',
            'Mar': '03',
            'Apr': '04',
            'May': '05',
            'Jun': '06',
            'Jul': '07',
            'Aug': '08',
            'Sep': '09',
            'Oct': '10',
            'Nov': '11',
            'Dec': '12'
        }[shortMonth]

    def convert_dates(dataframe):
        for i in range(dataframe['Date'].shape[0]):
            if len(dataframe['Date'][i]) == 10:
                year = dataframe['Date'][i][6:10]
                date = '0' + dataframe['Date'][i][4]
                month = monthToNum(dataframe['Date'][i][0:3])
                dataframe.loc[i, 'Date'] = year + '-' + month + '-' + date
            else:
                year = dataframe['Date'][i][7:11]
                date = dataframe['Date'][i][4:6]
                month = monthToNum(dataframe['Date'][i][0:3])
                dataframe.loc[i, 'Date'] = year + '-' + month + '-' + date

    convert_dates(game_data_2018)
    convert_dates(game_data_2019)

    game_results_2018 = np.array(list(game_data_2018['Visitor PTS'] - game_data_2018['Home PTS']))
    teams_2018 = np.array(list(zip(game_data_2018.Visitor, game_data_2018.Home)))
    dates_2018 = np.array(list(game_data_2018['Date']))

    game_results_2019 = np.array(list(game_data_2019['Visitor PTS'] - game_data_2019['Home PTS']))
    teams_2019 = np.array(list(zip(game_data_2019.Visitor, game_data_2019.Home)))
    dates_2019 = np.array(list(game_data_2019['Date']))

    # Combine all data into one dataset
    teams_all = np.concatenate((unique_teams, teams_2018, teams_2019), axis=0)
    dates_all = np.concatenate((unique_dates, dates_2018, dates_2019), axis=0)
    results_all = np.concatenate((unique_game_results, game_results_2018, game_results_2019), axis=0)

    return teams_all, dates_all, results_all


def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1))
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def x_for_game(roster, player_index, teams, date):
    num_players = len(player_index)

    x_1 = np.zeros((num_players, 1))
    x_2 = np.zeros((num_players, 1))

    if int(date[5:7]) < 9:
        year = str(int(date[0:4]) - 1)
    else:
        year = date[0:4]

    team_1_players = roster[teams[0]][year]
    for item in team_1_players:
        x_1[player_index[item]] = 1

    team_2_players = roster[teams[1]][year]
    for item in team_2_players:
        x_2[player_index[item]] = 1

    return x_1, x_2


def create_x_and_y(roster, player_index, teams, dates, results):
    num_games = teams.shape[0]
    num_players = len(player_index)

    # Create x for all games
    # To access x for 0th game -- x[0, :]
    x_without_intercept = np.zeros((num_games, 2 * num_players))

    for i in range(num_games):
        z, t = x_for_game(roster, player_index, teams[i], dates[i])
        combined = np.vstack((z, t))
        x_without_intercept[i, :] = combined[:, 0]

    x = x_without_intercept

    # Create y for all games (if team A wins, y = 1; if team B wins, y = 0)
    y = np.zeros((num_games, 1))
    for i in range(num_games):
        if results[i] > 0:
            y[i] = 1
        else:
            y[i] = 0

    return x, y


# def playoff_prediction(playoff_filename, playoff_date, theta):
#     # Load playoff data
#     playoff_data = pd.read_csv(os.path.join(curr_directory, playoff_filename))
#
#     # Extract features of interest
#     raw_playoff_results = np.array(list(playoff_data['PTS'] - playoff_data['PTS.1']))
#     raw_playoff_team_pairs = np.array(list(zip(playoff_data['Visitor/Neutral'], playoff_data['Home/Neutral'])))
#     raw_playoff_dates = np.array(list(playoff_data['Date']))
#
#     playoff_pairs = {}
#
#     for i in range(len(raw_playoff_team_pairs)):
#         team_1 = raw_playoff_team_pairs[i][0]
#         team_2 = raw_playoff_team_pairs[i][1]
#         if (team_1, team_2) in playoff_pairs.keys():
#             # if results > 0 --> team A won --> +1
#             # if results < 0 --> team B won --> -1
#             if raw_playoff_results[i] > 0:
#                 playoff_pairs[team_1, team_2] += 1
#             else:
#                 playoff_pairs[team_1, team_2] += -1
#         elif (team_2, team_1) in playoff_pairs.keys():
#             # if results > 0 --> team B won --> -1
#             # if results < 0 --> team A won --> +1
#             if raw_playoff_results[i] > 0:
#                 playoff_pairs[team_2, team_1] += -1
#             else:
#                 playoff_pairs[team_2, team_1] += 1
#         else:
#             if raw_playoff_results[i] > 0:
#                 playoff_pairs[team_1, team_2] = 1
#             else:
#                 playoff_pairs[team_1, team_2] = -1
#
#     playoff_teams = []
#     playoff_results = []
#     playoff_dates = []
#
#     for key in playoff_pairs:
#         playoff_teams.append([key[0], key[1]])
#         playoff_results.append(playoff_pairs[key])
#         playoff_dates.append(playoff_date)
#
#     playoff_teams = np.array(playoff_teams)
#     playoff_results = np.array(playoff_results)
#     playoff_dates = np.array(playoff_dates)
#
#     playoff_x, playoff_y = create_x_and_y(playoff_teams, playoff_dates, playoff_results)
#
#     predicted_y =
#
#     return prediction_accuracy


class QuadraticRegression:
    def __init__(self, player_index, step_size=1e-5, max_iter=200, eps=1e-3, batch_size=32, theta=None):

        self.theta = theta
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.num_players = len(player_index)
        self.error_list = []
        self.training_acc = []
        self.dev_acc = []

    def getSAS(self):
        # Top left
        S = np.array(self.theta[0:962, 0:962])
        # Bottom right
        S2 = np.array(self.theta[963:1925, 963:1925])
        # Top right
        A = np.array(self.theta[0:962, 963:1925])
        # Bottom left
        A2 = np.array(self.theta[963:1925, 0:962])

        return S, S2, A, A2

    def predict(self, x):
        z = x @ self.theta @ x.T
        return self.sigmoid(z)

    def sigmoid(self, z):
        return 1.0 / (1. + np.exp(-z))

    def loss_function_t(self, theta_t, x, y):
        eps = 1e-8
        x = torch.tensor(x)
        y = torch.tensor(y)
        p = torch.sigmoid(x @ theta_t @ x.T)
        return -1. * ((y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).sum())

    def pytorch_gradient(self, x, y):
        theta_t = torch.tensor(self.theta, requires_grad=True)
        self.loss_function_t(theta_t, x, y).backward()
        return theta_t.grad.numpy()

    def pytorch_batch_gradient(self, x_teams, y_teams, index):
        x = x_teams[index::self.batch_size]
        y = y_teams[index::self.batch_size]

        theta_t = torch.tensor(self.theta, requires_grad=True)
        self.loss_function_t(theta_t, x, y).backward()
        return theta_t.grad.numpy()

    def grad_loss_function(self, x_teams, y_teams):
        update = 0
        theta = np.matrix(self.theta)

        for i in range(x_teams.shape[0]):
            x = np.matrix(x_teams[i, :])
            y = np.asscalar(y_teams[i])
            update += x.T @ x @ theta @ x.T @ x - y * x.T @ x

        return update

    def grad_batch_loss_function(self, x_teams, y_teams, batch_size, index):
        update = 0
        theta = np.matrix(self.theta)

        for i in range(batch_size):
            x = np.matrix(x_teams[int((i + index) % x_teams.shape[0]), :])
            y = np.asscalar(y_teams[int((i + index) % x_teams.shape[0])])
            update += x.T @ x @ theta @ x.T @ x - y * x.T @ x

        return update

    def symmetrize(self, m):
        m1 = np.array(m)
        m2 = np.array(m).T
        m_out = 0.5 * (m1 + m2)

        return m_out

    def antisymmetrize(self, m):
        m1 = np.array(m)
        m2 = np.array(m).T
        m_out = 0.5 * (m1 - m2)

        return m_out

    def project(self, m):
        m = np.array(m)
        side = m.shape[0]
        S = self.symmetrize(m[0:int(side / 2 - 1), 0:int(side / 2 - 1)])
        S_minus = self.symmetrize(m[int(side / 2):int(side - 1), int(side / 2):int(side - 1)])

        A = self.antisymmetrize(m[0:int(side / 2 - 1), int(side / 2):int(side - 1)])
        A_minus = self.antisymmetrize(m[int(side / 2):int(side - 1), 0:int(side / 2 - 1)])
        S_new = (S - S_minus) / 2
        S_minus_new = (S_minus - S) / 2

        if np.allclose(A, -1 * A_minus, 1e-10, 1e-10):
            A_new = A
            A_minus_new = A_minus
        elif np.linalg.norm(A.T - A_minus, 2) < np.linalg.norm(A - A_minus, 2):
            A_new = 0.5 * (A + A_minus)
            A_minus_new = A_new.T
        else:
            A_new = 0.5 * (A + A_minus.T)
            A_minus_new = A_new.T

        M = np.zeros(m.shape)
        M[0:int(side / 2 - 1), 0:int(side / 2 - 1)] = S_new
        M[int(side / 2):int(side - 1), int(side / 2):int(side - 1)] = S_minus_new
        M[0:int(side / 2 - 1), int(side / 2):int(side - 1)] = A_new
        M[int(side / 2):int(side - 1), 0:int(side / 2 - 1)] = A_minus_new

        return M

    def general_predict(self, test_x, test_y):
        predicted_y = []
        for i in range(test_x.shape[0]):
            x = test_x[i, :]
            prediction = self.predict(x)
            if np.asscalar(prediction) > 0.5:
                predicted_y.append(1)
            else:
                predicted_y.append(0)

        predicted_y = np.array(predicted_y)
        return np.mean(np.array(predicted_y) == np.array(test_y.T))

    def fit(self, x, y, dev_x, dev_y, mini=False):
        iterations = 0
        abs_error = 1
        ind = 0

        if self.theta is None:
            self.theta = np.zeros((2 * self.num_players, 2 * self.num_players))

        if not mini:
            while iterations < self.max_iter and self.eps < abs_error < 1000000:
                error = self.step_size * self.pytorch_gradient(x, y)
                abs_error = np.linalg.norm(error, 2)
                self.error_list.append(abs_error)

                theta_new = self.theta - error
                self.theta = self.project(theta_new)

                iterations += 1

                train_accuracy = self.general_predict(x, y)
                self.training_acc.append(train_accuracy)
                dev_accuracy = self.general_predict(dev_x, dev_y)
                self.dev_acc.append(dev_accuracy)

                print('Error {}: {}'.format(iterations, abs_error))
                print('Training Accuracy: {}'.format(train_accuracy))
                print('Dev Accuracy: {}'.format(dev_accuracy))
        else:
            ind = 0
            while iterations < self.max_iter and self.eps < abs_error < 1000000:
                error = self.step_size * self.pytorch_batch_gradient(x, y, ind)
                abs_error = np.linalg.norm(error, 2)
                self.error_list.append(abs_error)

                theta_new = self.theta - error
                self.theta = self.project(theta_new)

                iterations += 1
                ind += 1

                print('Error {}: {}'.format(iterations, abs_error))

        print('Convergence!')
        plt.style.use('seaborn-darkgrid')
        plt.plot(self.training_acc, color='firebrick', label='Training Accuracy')
        plt.plot(self.dev_acc, color='teal', label='Dev Accuracy')
        plt.legend(loc='lower right')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('{}.png'.format(self.step_size), dpi=300)
        plt.close()

        return self.theta

def main():
    # Player Roster Information
    teams = ['BOS', 'BRK', 'NYK', 'PHI', 'TOR', 'CHI', 'CLE', 'DET', 'IND', 'MIL', 'ATL', 'CHO', 'MIA', 'ORL', 'WAS',
             'DEN', 'MIN', 'OKC', 'POR', 'UTA', 'GSW', 'LAC', 'LAL', 'PHO', 'SAC', 'DAL', 'HOU', 'MEM', 'NOP', 'SAS']

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

    teams, dates, results = preprocess_data()

    # x, y processing
    x, y = create_x_and_y(roster, player_index, teams, dates, results)

    # Train, test split
    x_train_all, x_test, y_train_all, y_test = \
        train_test_split(x, y, test_size=0.2, random_state=10)

    # Train, dev split
    x_train, x_dev, y_train, y_dev = \
        train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=10)

    # Different learning rates with projection
    print('lr = 1E-6')
    model_lr_1e6 = QuadraticRegression(player_index, step_size=1e-6, max_iter=50)
    theta_1e6 = model_lr_1e6.fit(x_train, y_train, x_dev, y_dev)

    train_accuracy_1e6 = model_lr_1e6.general_predict(x_train, y_train)
    dev_accuracy_1e6 = model_lr_1e6.general_predict(x_dev, y_dev)

    print('Train Accuracy: {}'.format(train_accuracy_1e6))
    print('Dev Accuracy: {}'.format(dev_accuracy_1e6))

    np.savetxt('training_acc_lr_1e6.txt', np.array(model_lr_1e6.training_acc), delimiter=',')
    np.savetxt('dev_acc_lr_1e6.txt', np.array(model_lr_1e6.dev_acc), delimiter=',')
    np.savetxt('theta_1e6.txt', model_lr_1e6.theta, delimiter=',')

    print('lr = 5E-7')
    model_lr_5e7 = QuadraticRegression(player_index, step_size=5e-7, max_iter=50)
    theta_5e7 = model_lr_5e7.fit(x_train, y_train, x_dev, y_dev)

    train_accuracy_5e7 = model_lr_5e7.general_predict(x_train, y_train)
    dev_accuracy_5e7 = model_lr_5e7.general_predict(x_dev, y_dev)

    print('Train Accuracy: {}'.format(train_accuracy_5e7))
    print('Dev Accuracy: {}'.format(dev_accuracy_5e7))

    np.savetxt('training_acc_lr_5e7.txt', np.array(model_lr_5e7.training_acc), delimiter=',')
    np.savetxt('dev_acc_lr_5e7.txt', np.array(model_lr_5e7.dev_acc), delimiter=',')
    np.savetxt('theta_5e7.txt', model_lr_1e6.theta, delimiter=',')

    print('lr = 1E-7')
    model_lr_1e7 = QuadraticRegression(player_index, step_size=1e-7, max_iter=50)
    theta_1e7 = model_lr_1e7.fit(x_train, y_train, x_dev, y_dev)

    train_accuracy_1e7 = model_lr_1e7.general_predict(x_train, y_train)
    dev_accuracy_1e7 = model_lr_1e7.general_predict(x_dev, y_dev)

    print('Train Accuracy: {}'.format(train_accuracy_1e7))
    print('Dev Accuracy: {}'.format(dev_accuracy_1e7))

    np.savetxt('training_acc_lr_1e7.txt', np.array(model_lr_1e7.training_acc), delimiter=',')
    np.savetxt('dev_acc_lr_1e7.txt', np.array(model_lr_1e7.dev_acc), delimiter=',')
    np.savetxt('theta_1e7.txt', model_lr_1e6.theta, delimiter=',')

    plt.style.use('seaborn-darkgrid')
    plt.plot(model_lr_1e6.dev_acc, color='firebrick', label='1E-6')
    plt.plot(model_lr_5e7.dev_acc, color='darkgreen', label='5E-7')
    plt.plot(model_lr_1e7.dev_acc,color='navy', label='1E-7')
    plt.legend(loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Dev Accuracy')
    plt.savefig('learning_rates_dev_accuracy.png', dpi=300)

    # Regularization


if __name__ == '__main__':
    main()
