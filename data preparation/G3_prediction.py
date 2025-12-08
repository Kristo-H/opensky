import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv("filtered_final_data.csv")

# features and target
features = ["velocity", "vertical_rate", "longitude", "latitude", "on_ground"] # features to predict target
target = "baro_altitude" # predict baro_altitude

X = data[features]
Y = data[target]

# training , testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# train decision tree model
decision_tree = DecisionTreeRegressor(random_state=10)
decision_tree.fit(X_train, Y_train)

#prediction
y_prediction_decision_tree = decision_tree.predict(X_test)

# MSE
def MSE(y_target, y_pred):
    n=len(y_target)
    sum = 0
    for i in range(n):
        delta = y_target[i] - y_pred[i]
        sum += delta ** 2
    mean = sum / n
    return mean

# RMSE
def RMSE(y_target, y_pred):
    return MSE(y_target, y_pred) ** 0.5


print("Decision tree RMSE: ", RMSE(Y_test.values, y_prediction_decision_tree))
print("Decision tree R2: ", r2_score(Y_test, y_prediction_decision_tree))


# train random forest model
random_forest = RandomForestRegressor(n_estimators=100, random_state=10)
random_forest.fit(X_train, Y_train)

y_prediction_random_forest = random_forest.predict(X_test)

print("Random Forest RMSE: ", RMSE(Y_test.values, y_prediction_random_forest))
print("Random Forest R2: ", r2_score(Y_test, y_prediction_random_forest))

"""
Decision tree RMSE:  1698.5754400795804
Decision tree R2:  0.8344769292731583
Random Forest RMSE:  1244.8482226502606
Random Forest R2:  0.9110959263309708
"""
# conclusion: 
# Random Forest performs better for predicting baro_altitude. It has higher R^2(0.91), indicating that
# it explains more of the variance in the data. 
# It also has lower RMSE(1244), meaning its predictions are closer to the actual baro_altitude values.
# R^2 explains the variance in the target variable. Value closer to 1 is better (model explains the data better)
# RMSE measures the average prediction error in units of baro_altitude, smaller RMSE is better(predictions are closer to actual values)
# Random Forest provides better accuracy, generalization.

# feature importance for Random Forest
# feature importance show how much each input variable contributes to the model's predictions. Higher value- bigger impact
feature_importances = random_forest.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance for Baro Altitude")
plt.show()

print("feature importances")
print(features, feature_importances)
#feature importance values:
"""
on ground: 0.00572
latitude: 0.03
longitude: 0.033
vertical_rate: 0.063
velocity: 0.868
"""
# conclusion: the most important value in Random Forest to predict baro_altitude is velocity


#plot Random Forest predicted baro_altitude vs actual baro_altitude to show model's performance
plt.figure(figsize=(10,6))
plt.scatter(Y_test, y_prediction_random_forest, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("Actual baro_altitude (m)")
plt.ylabel("Predicted baro_altitude (m)")
plt.title("Random Forest Prediction vs Actual baro_altitude")
plt.show()
# blue circles are predicted baro_altitude values, red diagonal line is actual baro_altitude values.


# plot, tune Random Forest to be more accurate
random_forest_tree_tuned = RandomForestRegressor(
    n_estimators = 300, # more trees for stable predictions
    max_depth= 20, # limits tree depth to prevent overfitting
    min_samples_leaf=5, # ensures each leaf has enough samples for reliability
    random_state=10,
    n_jobs = 1
)

random_forest_tree_tuned.fit(X_train, Y_train)
random_forest_tree_tuned = random_forest_tree_tuned.predict(X_test)

#plot
plt.figure(figsize=(10,6))
plt.scatter(Y_test, random_forest_tree_tuned, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("Actual baro_altitude")
plt.ylabel("Predicted baro_altitude")
plt.title("Tuned Random Forest Prediction vs Actual")
plt.show()


# Also check Decision Tree feature importance and tune it
# feature importance for Decision Tree
feature_importances2 = decision_tree.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, feature_importances2)
plt.xlabel("Feature Importance")
plt.title("Decision Tree Feature Importance for Baro Altitude")
plt.show()

print("feature importances")
print(features, feature_importances2)
#feature importance values for Decision Tree:
"""
on ground: 0.00569
latitude: 0.03
longitude: 0.031
vertical_rate: 0.067
velocity: 0.866
"""
# conclusion: the most important value in Decision Tree to predict baro_altitude is  also velocity

# plot Decision Tree predicted baro_altitude vs actual baro_altitude
plt.figure(figsize=(10,6))
plt.scatter(Y_test, y_prediction_decision_tree, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("Actual baro_altitude")
plt.ylabel("Predicted baro_altitude")
plt.title("Decision Tree Prediction vs Actual")
plt.show()

# plot, tune Decision Tree to be more accurate
decision_tree_tuned = DecisionTreeRegressor(
    max_depth = 10, # limits tree depth to reduce overfitting
    min_samples_leaf=10, # ensures leavesa have enough samples
    min_samples_split=20, # requires minimum samples to split a node
    random_state=10
)

decision_tree_tuned.fit(X_train, Y_train)
decision_tree_tuned_predict = decision_tree_tuned.predict(X_test)

#plot
plt.figure(figsize=(10,6))
plt.scatter(Y_test, decision_tree_tuned_predict, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("Actual baro_altitude")
plt.ylabel("Predicted baro_altitude")
plt.title("Tuned Decision Tree Prediction vs Actual")
plt.show()

# tuned values
print("Tuned Decision tree RMSE: ", RMSE(Y_test.values, decision_tree_tuned_predict))
print("Tuned Decision tree R2: ", r2_score(Y_test, decision_tree_tuned_predict))
print("Tuned Random Forest RMSE: ", RMSE(Y_test.values, random_forest_tree_tuned))
print("Tuned Random Forest R2: ", r2_score(Y_test, random_forest_tree_tuned))
# After tuning, Decision Tree improved: RMSE - 1698.58 -> 1387.81. R2 - 0.83 -> 0.89
# Random Forest did not improve: RMSE - 1244.84 -> 1283.49. R2 - 0.911 -> 0.905
# Random Forest is still the best for predicting baro_altitude.
# Most accurate model to predict baro_altitude is Random Forest before tuning.


# To improve model even more: add new features that may improve model
# check all columns as features, and look for important features
irrelevant_features = ["baro_altitude", "geo_altitude", "altitude_diff_percent", "altitude_diff"]
check_features = data.select_dtypes(include=["float64", "int64", "bool"]).columns.tolist() # features to predict target
check_features = [f for f in check_features if f not in irrelevant_features]

print("columns used for feature importance check: ", check_features)
X_all = data[check_features]
Y_all = data[target]
X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(X_all, Y_all, test_size=0.2, random_state=10)

# train new random forest model with new features
random_forest_all = RandomForestRegressor(n_estimators=100, random_state=10)
random_forest_all.fit(X_train_all, Y_train_all)

check_features_importances = random_forest_all.feature_importances_
feature_series = pd.Series(check_features_importances, index=check_features)

#sort descending by importance
feature_series_sorted = feature_series.sort_values(ascending=False)

#decimals
pd.set_option("display.float_format", "{:.6f}".format)
print(feature_series_sorted)

# found that true_track is better feature than on_ground, train predict model using true_track instead.
"""
velocity          0.863166
vertical_rate     0.060528
longitude         0.025796
latitude          0.023955
true_track        0.018616
on_ground         0.005666
time_position     0.001311
last_contact      0.000924
spi               0.000038
position_source   0.000000
"""

y_prediction_random_forest_all = random_forest_all.predict(X_test_all)

print("New Random Forest RMSE: ", RMSE(Y_test_all.values, y_prediction_random_forest_all))
print("New Random Forest R2: ", r2_score(Y_test_all, y_prediction_random_forest_all))

#plot Random Forest predicted baro_altitude vs actual baro_altitude to show model's performance
plt.figure(figsize=(10,6))
plt.scatter(Y_test_all, y_prediction_random_forest_all, alpha=0.5)
plt.plot([Y_test_all.min(), Y_test_all.max()], [Y_test_all.min(), Y_test_all.max()], "r--")
plt.xlabel("Actual baro_altitude")
plt.ylabel("Predicted baro_altitude")
plt.title("Random Forest Prediction vs Actual")
plt.show()

# new random forest with all features is slightly better: 1237 vs 1244 RMSE. 0.912 vs 0.911 R2
# New Random Forest RMSE:  1237.3603465899212
# New Random Forest R2:  0.9121622419532451



# Random Forest with top features:
top_features = ["velocity", "vertical_rate", "longitude", "latitude", "true_track"]

X_top = data[top_features]
Y_top = data[target]
X_train_top, X_test_top, Y_train_top, Y_test_top = train_test_split(X_top, Y_top, test_size=0.2, random_state=10)

# train top random forest model with top features
random_forest_top = RandomForestRegressor(n_estimators=100, random_state=10)
random_forest_top.fit(X_train_top, Y_train_top)

top_features_importances = random_forest_top.feature_importances_
feature_series_top = pd.Series(top_features_importances, index=top_features)

#sort descending by importance
feature_series_top_sorted = feature_series_top.sort_values(ascending=False)

#decimals
pd.set_option("display.float_format", "{:.6f}".format)
print(feature_series_top_sorted)

# found that true_track is better feature than on_ground, train predict model using true_track instead.
"""
velocity        0.865347
vertical_rate   0.061609
longitude       0.027186
latitude        0.026385
true_track      0.019473
"""

y_prediction_random_forest_top = random_forest_top.predict(X_test_top)

print("Top features Random Forest RMSE: ", RMSE(Y_test_top.values, y_prediction_random_forest_top))
print("Top features Random Forest R2: ", r2_score(Y_test_top, y_prediction_random_forest_top))

#plot Random Forest predicted baro_altitude vs actual baro_altitude to show model's performance
plt.figure(figsize=(10,6))
plt.scatter(Y_test_top, y_prediction_random_forest_top, alpha=0.5)
plt.plot([Y_test_top.min(), Y_test_top.max()], [Y_test_top.min(), Y_test_top.max()], "r--")
plt.xlabel("Actual baro_altitude")
plt.ylabel("Predicted baro_altitude")
plt.title("Random Forest Prediction vs Actual")
plt.show()

# Top Random Rorest with top features is worse than original Random Forest: 1278 vs 1244 RMSE. 0.906 vs 0.911 R2
# Top Random Forest RMSE:  1278.7112290729776
# Top Random Forest R2:  0.9061933099989871
# conclusion: 
# Random Forest with top features (velocity, vertical_rate, longitude, latitude, true_track) performs worse than
# Random Forest with original features(velocity, vertical_rate, longitude, latitude, on_ground).
# Even though true_track has higher importance than on_ground, the combination of all original Random Forest features are still better.
# In the original Random Forest, on_ground gave the Random Forest more information to reduce error.

# Random Forest with best prediction is with features velocity, vertical_rate, longitude, latitude, on_ground, and no tuning:
# Random Forest RMSE:  1244.8482226502606
# Random Forest R2:  0.9110959263309708