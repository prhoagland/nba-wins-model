# Project-4
### From Courtside to Spreadsheet: Unveiling Basketball Analytics
#### Team Members: Peter Hoagland, Alena Matusevich, and Lillian Ruelas-Thompson (All code and analysis from all members)
#### Data source: https://www.kaggle.com/datasets/nathanlauga/nba-games?select=ranking.csv,
####              https://www.kaggle.com/datasets/drgilermo/nba-players-stats?resource=download,
####              season_id_lookup.csv, Season_Stats.csv, team_abrev_lookup.csv, year_team_abrev_lookup.csv 
#### Summary:
Basketball season is in full effect and we are trying to train a model that will predict the number of wins a team will reach in a season using the columns below by combining the csv files above and creating a team_stats_final.csv file. The csv files are all stored in the Resources folder and the code used to clean the data is in the file Project4_data_cleanup.ipynb. 

##### team_name	abv	pts_per_min	2fg_pct	3fg_pct	ts_pct	dbpm_tot	obpm_tot	wins_tot

The description of the columns above are broken down below:

##### team_name – name of team from ranking.csv (TEAM column)

##### abv – team abbreviation from Seasons_Stats.csv (Tm column)

pts_per_min (points per minute) – total points the team scored divided by minutes played (dividing by minutes played should help account for when some teams go into overtime more than others). Should be calculated:
##### pts_per_min = SUM(PTS)/SUM(MP)

2fg_pct (2-point field goal percentage) – percentage of two-point shots the team made by how many they attempted. Should be calculated:
##### 2fg_pct = SUM(2P)/SUM(2PA)

3fg_pct (3-point field goal percentage) – percentage of three-point shots the team made by how many they attempted. Should be calculated:
##### 3fg_pct = SUM(3P)/SUM(3PA)

ts_pct (true shooting percentage) – this is pretty commonly used to compare shooting stats because it takes into account the fact that some players shoot more three-point shots than others. Here’s a link to the Wikipedia article: https://en.wikipedia.org/wiki/True_shooting_percentage
Should be calculated:
##### ts_pct = SUM(PTS)/(2*(SUM(FGA) + 0.44*SUM(FTA)))

dbpm_tot (Defense Box Plus/Minus Total) – DBPM is a number created to compare each player’s contributions to defense to each other no matter what team they play for. Here’s a more in-depth explainer of it: https://www.sports-reference.com/blog/2014/10/introducing-box-plusminus-bpm-2/
Basically the idea is to add up each team’s player’s DBPM to see which teams have the better defensive players. Since the number doesn’t take into account how many minutes each player plays, we have to weight it by each player’s minutes. Should be calculated like this:
##### dbpm_tot = SUM(DBPM*MP)

obpm_tot (Offensive Box Plus/Minus Total) – Basically the same as DBPM but for offense. Calculated similarly:
##### obpm_tot = SUM(OBPM*MP)

##### wins_tot (Total Wins) – Taken directly from the total wins (W column) in ranking.csv. This is the outcome variable we’re going to train the model to predict.

Based on the calculated columns and training a model, we hope to answer these questions: 

* #### What type of model generates the most accurate predictions for how a basketball team will perform (win total) in a given season?
* #### What statistics fed into the model inform the model the most? Common statistics include a team’s previous record, offensive rating, defensive rating, and the team’s player’s individual statistics?

#### Machine Learning Models Used:
For this project, we used a Decision Tree model and a Neural Network model and we optimized the models to get the most accurate score by updating and removing columns, increasing epochs, using different activations, as well as creating visualizations such as the importance_features to determine the optimal columns to be used for the models. The code used to create the the decision tree model is in the file model_optimization_decision_tree_colab.ipynb. The code used for the Neural Network model is in separate files due to Keras-Tuner compatibility. The file with the six main stats is model_optimization_colab_neural.ipynb, the file using the six main stats with the Year column is model_optimization_colab_neural_year.ipynb, and the file using all the stats is model_optimization_colab_neural_year.ipynb. 

#### Results:
Using a Decision Tree model, we were able to reach a r2 score of 0.759, which is really close the the 0.80 score that is required. In order to reach this score, we had to drop many columns, and normalize the dbpm_tot and obpm_tot values by dividing their weighted sums by the team's total minutes.  We also added the Year column and dropped the categorical columns and columns pertaining to team wins/losses/games played. The removed categorical columns included the team name and abbreviation. Using the Decision Tree model, we were also able to analyze the importance of the features, which gave us more of an idea of which features were more valuable for the model.

The neural network model was originally run to confirm our choices in features to train the model. The Keras-Tuner was run only going to 20 epochs (to save time) with three separate combinations of features: the six main team stats, the six stats in addition to the year, and then all the numerical stats. These initial runs showed that including the year was important for training the model, while including stats other than the six major stats did not noticeably improve the model. The Keras-Tuner was then run to 100 epochs and an RMSE value of 3.71 and r2 value of 0.895 was obtained.

From our models, we were able to answer the questions we had proposed.

* #### What type of model generates the most accurate predictions for how a basketball team will perform (win total) in a given season?
  * Based on the data that was used the best type of model that gave the most accurate predictions was the Neural Network model, however, in using the decision tree model, we were able to see which features were more important than others, and how we needed to trim the columns to run both models.

* #### What statistics fed into the model inform the model the most? Common statistics include a team’s previous record, offensive rating, defensive rating, and the team’s player’s individual statistics?
  *  We used mostly the same columns and data points for each ML model, and we were able to determine that the obpm_norm (Normalized Offensive Box Plus/Minus) and the dbpm_norm (Normalized Defense Box Plus/Minus) informed both models the most. We also found that, due to shooting trends across the time period analyzed, including the Year improved the accuracy of both models.​