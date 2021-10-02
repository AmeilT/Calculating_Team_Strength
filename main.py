import pandas as pd

from strength_functions import team_strengths_create,load_PL_data,create_param_dict,create_team_df,apply_team_strength
PL_season=load_PL_data()

# model distribution of scorelines with a histogram of the data
# score_diff = PL_season["Home Goals"] - PL_season["Away Goals"]
# (mu, sigma) = norm.fit(score_diff)
# fig, ax = plt.subplots()
# n, bins, patches = ax.hist(score_diff, np.arange(-10, 10, 1), facecolor='green', alpha=0.75)

# add a 'best fit' line
# y = norm.pdf(bins, mu, sigma)
# print(mu, sigma)
# ax.plot(bins, y * score_diff.size, 'r--', linewidth=1)

SEASON_RANGE=[2011,2021]
GAMEWEEK_RANGE=[1,38]
WINDOW = 10

param_dict=create_param_dict(SEASON_RANGE,PL_season,WINDOW)

teams_df=create_team_df(PL_season)

teams_df=apply_team_strength(teams_df,param_dict)



#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
