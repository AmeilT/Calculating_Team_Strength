import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATADIR = r"C:\Users\ameil\Documents\GitHub\Sports Analytics Course (Harvard Online)\L2 Poisson"


def load_PL_data():
    """Loads in PL match result data from CSV file located in your DATADIR
    :return: a pandas dataframe containing the scores of every regular season PL game in the given season
    """
    PL_season = pd.read_csv(DATADIR + f'/PL Results Historical.csv')
    PL_season.sort_values(by=["Season","Gameweek ID"], inplace=True)

    return PL_season


def get_X(teams, PL_season):
    """Creates the X matrix for the normal model calculation
    :param teams: numpy array of all unique teams in the match result dataset
    :param PL_season: pandas dataframe containing PL match result data
    :return: A numpy array where each row is a unique game of the dataset, and each column represents a team
    """
    # calculate number of teams 32
    nteams = teams.size
    # calculate number of games 256
    ngames = PL_season.shape[0]
    # assign each game a unique id #0-255 list
    game_id = np.arange(ngames)
    # initalize X matrix
    X = np.zeros((ngames, nteams))  # zeros 256,32 (games x teams)
    # populate home teams, for each of the 256 games tell me which is playing according to team number
    home_team_idx = np.searchsorted(teams, PL_season["Home"])
    X[game_id, home_team_idx] = 1
    # populate away teams
    away_team_idx = np.searchsorted(teams, PL_season["Away"])
    X[game_id, away_team_idx] = -1
    return X


def get_W(nteams):
    """Creates the W matrix for the normal model calculation, with the constraint that team strengths must sum to 0
    :param nteams: An integer representing the number of teams in our dataset for generating the W matrix
    :return: W matrix for estimating team strength
    """
    # set up an nteams x nteams identify matrix
    W = np.eye(nteams)
    # set all elements in the last row = -1
    W[-1, :] = -1
    # remove the last column
    W = W[:, :-1]
    return W


def team_strengths_create(df):
    """
    :param df: Results data for the PL season
    :return: tuple of paramaters in list or dict form
    """
    # find all the team names
    teams = np.unique((df['Home'], df["Away"]))
    # calculate number of teams
    nteams = teams.size
    # calculate number of games
    ngames = df.shape[0]
    # calculate design matrix
    X = get_X(teams, df)
    # get observed score differentials
    y = df["Home Goals"] - df["Away Goals"]
    # construct linear constraint matrix
    W = get_W(nteams)
    # calculate matrix product of X and W
    Xs = np.matmul(X, W)

    # here is the solution! (the @ symbol is another way of doing matrix multiplication in python)
    theta_n1 = np.linalg.inv(Xs.T @ Xs) @ Xs.T @ y

    # calculate the nth team strength
    theta_n = np.append(theta_n1, -theta_n1.sum())
    # expected score differences
    yhat = X @ theta_n

    # variance of residuals
    svar = np.dot(y - yhat, y - yhat) / (ngames - nteams)
    # covariance matrix of coefficients
    theta_covar = svar * W @ np.linalg.inv(Xs.T @ Xs) @ W.T
    # standard errors of  coefficients
    theta_sigma = np.sqrt(np.diagonal(theta_covar))

    team_params = []
    for team, theta, sigma in zip(teams, theta_n, theta_sigma):
        team_params.append((team, theta.round(3), sigma.round(3)))

    team_params_dict = dict(zip(teams, theta_n.round(3)))

    team_params = sorted(team_params, key=lambda x: x[0], reverse=False)
    for tp in team_params[::-1]:
        print("%s,%1.3f,%1.3f" % tuple(tp))
    print("Residual standard deviation,%1.3f," % (np.sqrt(svar)))
    return team_params, team_params_dict


def create_param_dict(SEASON_RANGE, PL_season, WINDOW):
    """

    :param SEASON_RANGE: min and max of the seasons in the results dataframe
    :param PL_season: dataframe of results
    :param WINDOW: lookback window to calculate team strength
    :return: dictionary of form {(SEASON,GAMEWEEK):{team:strength}
    """
    param_dict = {}
    for season in range(SEASON_RANGE[0], SEASON_RANGE[1] + 1):
        df = PL_season[PL_season["Season"] == season]
        END_GAMEWEEK = df["Gameweek ID"].max()
        START_GAMEWEEK = df["Gameweek ID"].min()

        for gameweek in range(START_GAMEWEEK + 1, END_GAMEWEEK + 1):
            if gameweek <= WINDOW:
                results_window = df[(df["Gameweek ID"] <= gameweek)]
                team_params, team_dict = team_strengths_create(results_window)
                print(season, "Up to", gameweek, team_params)
            else:
                results_window = df[(df["Gameweek ID"] >= min(gameweek, 33))
                                    & (PL_season["Gameweek ID"] <= min(END_GAMEWEEK, gameweek + WINDOW))]
                team_params, team_dict = team_strengths_create(results_window)
                print(season, min(gameweek, 33), ":", min(END_GAMEWEEK, gameweek + WINDOW), team_params)

            data_dict = {(season, gameweek): team_dict}
            param_dict.update(data_dict)
    return param_dict


def create_team_df(PL_season):
    """

    :param PL_season: dataframe of PL results
    :return: Dataframe for every team in the dataset by gamweek and season
    """
    home = PL_season.copy()
    home.drop("Away", inplace=True, axis=1)
    home = home.rename(columns={"Home": "Team"})
    home = home[["Season", "Gameweek ID", "Team"]]
    away = PL_season.copy()
    away.drop("Home", inplace=True, axis=1)
    away = away.rename(columns={"Away": "Team"})
    away = away[["Season", "Gameweek ID", "Team"]]
    teams_df = home.append(away)
    teams_df.sort_values(by=["Season", "Gameweek ID", "Team"], inplace=True)
    return teams_df


def apply_team_strength(teams_df, param_dict):
    for index, row in teams_df.iterrows():
        team = row["Team"]
        gameweek = row["Gameweek ID"]
        season = row["Season"]
        if gameweek == 1:
            teams_df.loc[index, 'Strength'] = "NA"
        else:
            strength = param_dict[(season, gameweek)][team]
            print(season, gameweek, team, strength)
            teams_df.loc[index, 'Strength'] = strength
    return teams_df


def make_team_strength_plot(season, gameweek, param_dict):
    """Creates horizontal bar chart for team strength estimates based on the ouptut of the normal team strength model model

    :param season: Season that we are generating plot for
    :param gameweek: gameweek that we are generating plot for (note this is the last gameweek of the range, eg enter "20"
    to view strenghts from gameweek 10-20 (assuming window =10)
    :param param_dict: list or numpy array of tuples containing information, where the first element of each tuple is a team name and the second element is their strength estimate
    """
    team_params = param_dict[season, gameweek]
    teamnames = [t for t in list(team_params.keys())]
    theta_ = [v for v in list(team_params.values())]

    # plot outcomes
    fig, ax = plt.subplots(figsize=(6, 8))
    dict_param = dict(zip(teamnames, theta_))
    list_param = sorted(dict_param.items(), key=lambda x: x[1], reverse=False)

    teams = [t[0] for t in list_param]
    values = [t[1] for t in list_param]

    ax.barh(teams, values, facecolor='red', alpha=0.3, height=0.5)
    fig.suptitle(f'         Team strength estimates, PL {season, gameweek}', y=0.92)
    ax.set_xlabel('Team strength')
    ax.xaxis.grid(True)
    fig.subplots_adjust(left=0.3)
