import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# loading dataset, exploring the dataset
data = pd.read_csv("opensky_states_snapshot.csv")
print(data.head())
print(data.info()) # -> dataset has Dtype-s of object, float64, int64, bool,
print(data.describe())
print(data.isnull().sum()) # sum in each column that have missing values

# Columns with missing values before cleaning:
""" 
[8 rows x 12 columns]
icao24                0
callsign            136
origin_country        0
time_position        61
last_contact          0
longitude            61
latitude             61
baro_altitude       605
on_ground             0
velocity              2
true_track            0
vertical_rate       584
sensors            7420
geo_altitude        701
squawk             2870
spi                   0
position_source       0
"""



# Handling missing and inconsistent values
# drop column with all missing values
data.drop(columns=["sensors"], inplace=True)

# filling missing categorical columns
data["callsign"].fillna("Unknown", inplace=True)
data["squawk"].fillna("Unknown", inplace=True)

# filling missing numerical columns with median
fill_median_list = ["time_position", "longitude", "latitude", "baro_altitude", "velocity", "vertical_rate", "geo_altitude"]
for column in fill_median_list:
    data[column].fillna(data[column].median(), inplace=True)


# after preparing the data, check the data for missing values again
print(data.isnull().sum())
print(data.head())
print(data.info())
print(data.describe())


# Columns with missing values after cleaning:
"""
icao24             0
callsign           0
origin_country     0
time_position      0
last_contact       0
longitude          0
latitude           0
baro_altitude      0
on_ground          0
velocity           0
true_track         0
vertical_rate      0
geo_altitude       0
squawk             0
spi                0
position_source    0
"""


# Handle inconsistent values
#checking unique values in each column
for column in data.columns:
    print(f"{column}: {data[column].nunique()} unique values")

#columns that can be used to explore: origin_country, longitude, latitude, baro_altitude, on_ground, velocity, vertical_rate, geo_altitude, spi


#histograms for quick distribution view on numerical columns
numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
for column in numerical_columns:
    plt.figure(figsize=(8,5))
    plt.hist(data[column], bins=30, color="skyblue", edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f'Distribution of {column}')
    plt.show()


# view categorical columns
categorical_columns = data.select_dtypes(include=["object", "bool"]).columns
print(categorical_columns)
# plot to check for possible columns to explore
"""
for column in categorical_columns:
    plt.figure(figsize=(15,10))
    data[column].value_counts().plot(kind="bar")
    plt.title(f'Counts for {column}')
    plt.ylabel("Column")
    plt.xticks(rotation=45)
    plt.show()
"""
# no patterns to explore in icao24, callsign, squawk


# plot for patterns: origin_country, on_ground, spi
#plot origin_country
all_countries = data["origin_country"].unique()
country_counts = data["origin_country"].value_counts()
country_counts = country_counts[country_counts > 0] # countries with more than 0 flights
top_countries = data["origin_country"].value_counts().head(10) # top 10 countries
missing_countries = [c for c in all_countries if c not in country_counts.index]

print("Countries with zero entries: ", missing_countries)
plt.figure(figsize=(15,10))
top_countries.plot(kind="bar")
plt.title("Top 10 origin countries by number of flights")
plt.ylabel("Number of flights")
plt.xticks(rotation=45)
plt.show()

#spi(special purpose indicator flag) true
print("spi true:")
print(data[data["spi"] == True])


# plot on_ground, spi
for column in ["on_ground", "spi"]:
    plt.figure(figsize=(6,4))
    data[column].value_counts().plot(kind="bar")
    plt.title(f'Counts for {column}')
    plt.ylabel("Number of flights")
    plt.xticks(rotation=0)
    plt.show()


# rows where on_ground is true
on_ground_data = data[data["on_ground"] == True][["velocity", "geo_altitude", "baro_altitude"]]
on_ground_data_sorted = on_ground_data.sort_values(by="velocity", ascending=False)
print("on_ground data sorted")
print(on_ground_data_sorted)
# some of these rows are anomalies - baro_altitude and geo_altitude can not be that high when plane is not flying
# for example:
on_ground_data_sorted_rows = on_ground_data.sort_values(by="velocity", ascending=True).head(20)
print("on_ground data sorted (rows with anomalies)")
print(on_ground_data_sorted_rows)


# sanity check - check count, mean, std, min, max, quartiles for these numerical columns
print("sanity check")
print(data[["baro_altitude", "geo_altitude", "velocity"]].describe())
# explanation: 
# mean values: average baro_altitude-7402m, geo_altitude-7834m, velocity 172m/s
# std: altitudes vary around 4100m-4200m, velocity 84m/s
# min: smallest baro_altitude- -122m , geo_altitude- -31m, velocity-0.0m/s
# max: largest baro_altitude-21062m, geo_altitude-21046m, velocity 1241m/s
# 50% of baro_altitudes are <= 9106m, geo_altitudes <= 9601m, velocity <= 211m/s


# check rows for impossible values (shows where geo_altitude, baro_altitude, velocity is negative)
negative_values = data[(data["geo_altitude"] < 0) | 
           (data["baro_altitude"] < 0) | 
           (data["velocity"] < 0)]

# show negative values for each column
print("negative values:")
print(negative_values[["geo_altitude", "baro_altitude", "velocity"]])
# conclusion -> values are within possible values, no need to remove



#check that rows don't have too high positive values
print("top 5 geo_altitude: ")
print(data["geo_altitude"].sort_values(ascending=False).head(5))

# check top 1 geo_altitude 21046.44m:
print("top 1 geo_altitude values:")
print(data.loc[data["geo_altitude"].idxmax(), 
               ["geo_altitude", "baro_altitude", "velocity"]])
# conclusion -> top 1 geo_altitude 21046.44m with baro_altitude 20086.32m and velocity 2.06 m/s is possible for an aircraft

# check top 20 geo_altitude values
print("top 20 geo_altitude values with velocity and baro_altitude")
print(data.sort_values("geo_altitude", ascending=False).head(20)[["geo_altitude", "baro_altitude", "velocity"]])
#conclusion: geo_altitudes are in normal ranges, no handling needed

#top 20 geo_altitude:
""" geo_altitude  baro_altitude  velocity
2512      21046.44       20086.32      2.06
1361      19644.36       18684.24      5.25
5233      16329.66       15544.80      2.77
3750      15026.64       14325.60    251.21
2030      14394.18       13716.00    212.03
1050      14348.46       13693.14    236.38
2220      14272.26       13716.00    230.01
6918      14234.16       13716.00    191.38
4254      14196.06       13716.00    196.87
4819      14173.20       13708.38    240.82
2681      14142.72       13708.38    252.04
7147      14127.48       13716.00    248.03
7291      14119.86       13716.00    269.26
7062      14104.62       13716.00    231.76
3530      14043.66       13106.40    267.36
1136      13952.22       13098.78    277.34
3396      13929.36       13106.40    267.57
1625      13906.50       13106.40    243.03
1881      13853.16       13106.40    264.94
5341      13837.92       13106.40    228.51"""



print("top 5 baro_altitude: ")
print(data["baro_altitude"].sort_values(ascending=False).head(5))

# check top 1 baro_altitude 21061.68m:
print("top 1 baro_altitude:")
print(data.loc[data["baro_altitude"].idxmax(), 
               ["baro_altitude", "geo_altitude", "velocity"]])
# conclusion -> top 1 baro_altitude 21061.68m with geo_altitude 121.92m and velocity 51.51 m/s is possible for an aircraft
print()

# check top 20 baro_altitude values
print("top 20 baro_altitude values with velocity and geo_altitude")
print(data.sort_values("baro_altitude", ascending=False).head(20)[["baro_altitude", "geo_altitude", "velocity"]])
#conclusion: values with row indexes 6125 and 1003 have high difference between baro_altitude and geo_altitude -> apply contradictions.

# top 20 baro_altitude:
""" baro_altitude  geo_altitude  velocity
6125       21061.68        121.92     51.51
2512       20086.32      21046.44      2.06
1003       18714.72        137.16     51.98
1361       18684.24      19644.36      5.25
5233       15544.80      16329.66      2.77
3750       14325.60      15026.64    251.21
2030       13716.00      14394.18    212.03
2220       13716.00      14272.26    230.01
7291       13716.00      14119.86    269.26
7147       13716.00      14127.48    248.03
7062       13716.00      14104.62    231.76
4254       13716.00      14196.06    196.87
6918       13716.00      14234.16    191.38
2681       13708.38      14142.72    252.04
4819       13708.38      14173.20    240.82
1050       13693.14      14348.46    236.38
6763       13114.02      13822.68    243.38
2608       13114.02      13616.94    241.82
1625       13106.40      13906.50    243.03
2238       13106.40      13182.60    244.70"""




#altitude contradiction:
#calculating geo_altitude and baro_altitude differences
data["altitude_diff"] = data["geo_altitude"] - data["baro_altitude"]
data["altitude_diff_percent"] = (data["altitude_diff"] / data["geo_altitude"]) * 100
print("altitude_diff percent")
print(data["altitude_diff_percent"].describe())


print("altitude contradiction")
print(data[["geo_altitude","baro_altitude","altitude_diff", "altitude_diff_percent"]].sort_values(by="altitude_diff_percent", ascending=False))
# Top 20 rows with largest altitude difference percentage
print(data[["geo_altitude", "baro_altitude", "altitude_diff", "altitude_diff_percent"]]
      .sort_values(by="altitude_diff_percent", ascending=False).head(20))

# show all rows with geo_altitude, baro_altitude, and altitude_diff
print("altitude diff")
print(data[["geo_altitude", "baro_altitude", "altitude_diff"]].sort_values(by="altitude_diff", ascending=False))
# large values for altitude_diff(9601.2m and -20939m). geo_altitude and baro_altitude need to be in correlation -> remove large values

# export to CSV to view all altitude_diff
data[["geo_altitude", "baro_altitude", "altitude_diff"]].sort_values(by="altitude_diff", ascending=False).to_csv("altitude_diff_full.csv", index=False)
# conclusion: filter out altitude differences greater than 1000m positive and 500m negative, to remove extreme data errors and keep most valid flight data

#filtering out by altitude difference:
data_filtered = data[(data["altitude_diff"] <= 1000) & (data["altitude_diff"] >= -500)]

# export to CSV to view all filtered altitude_diff
data_filtered[["geo_altitude", "baro_altitude", "altitude_diff"]].sort_values(by="altitude_diff", ascending=False).to_csv("altitude_diff_full_filtered.csv", index=False)


print()
print("top 20 velocity: ")
print(data["velocity"].sort_values(ascending=False).head(20))
print("top 20 velocity values with icao24, callsign, geo_altitude and baro_altitude")
print(data_filtered.sort_values("velocity", ascending=False).head(20)[["icao24", "callsign", "velocity", "baro_altitude", "geo_altitude"]])


# filter out planes that are above 340m/s per second (1 Mach is the speed of sound, these planes(by icao24) don't reach that speed)
# Filter out rows with high velocity (>340 m/s)
data_filtered = data_filtered[data_filtered["velocity"] <= 340]

#check top velocities after filtering
print("top 20 velocities after filtering")
print(data_filtered.sort_values("velocity", ascending=False).head(20)[["icao24", "callsign","velocity","baro_altitude","geo_altitude"]])


# Also need to filter out extremely low velocity (<20 m/s)
print("velocity 20")
print(data_filtered.sort_values("velocity")[["callsign","velocity","baro_altitude","geo_altitude"]].head(20))


# export to CSV to view all velocities
data_filtered.sort_values("velocity", ascending=False)[["callsign","velocity","baro_altitude","geo_altitude"]].to_csv("velocity.csv", index=False)

# remove duplicates: baro_altitude=9105.9, geo_altitude=9601.2, velocity < 25 . (parked aircraft can not be at 9000m altitude)
data_filtered = data_filtered[~((data_filtered["velocity"] < 25) & 
                                (data_filtered["baro_altitude"] == 9105.9) &
                                (data_filtered["geo_altitude"] == 9601.2))]

# export to CSV after filtering to view all filtered velocities
data_filtered.sort_values("velocity", ascending=False)[["callsign","velocity","baro_altitude","geo_altitude"]].to_csv("velocity_filtered.csv", index=False)

# Check lowest velocities after filtering
print(data_filtered.sort_values("velocity")[["callsign","velocity","baro_altitude","geo_altitude"]].head(20))


# check vertical_rate values
print("vertical rate")
print(data_filtered["vertical_rate"].describe())

# export vertical_rate values to CSV
data_filtered.sort_values("vertical_rate", ascending=False)[["icao24", "callsign","velocity","baro_altitude","geo_altitude", "vertical_rate"]].to_csv("vertical_rate.csv", index=False)
# mean -0.012 shows that total ascent and descent are balanced (aircraft is slightly descending)
# 25% to 75% is +-0.33 . 50% of data is clustered near level flight
# min and max are high, but they are actual climbs or descents, not wrong data, no filtering needed.


# after removing anomalies identified in Goal 2, clusters are now denser and more consistent
# Export fully filtered dataset to CSV
data_filtered.to_csv("filtered_final_data.csv", index=False)


# histograms for some numerical features, to spot patterns and outliers
numerical_features = {
    "baro_altitude": "Altitude (m)", 
    "geo_altitude": "Altitude (m)", 
    "velocity": "Velocity (m/s)",
    "vertical_rate": "Vertical rate (m/s)"
}
for column, item in numerical_features.items():
    plt.figure(figsize=(8,5))
    plt.hist(data_filtered[column], bins=50, color="skyblue", edgecolor="black")
    plt.title(f'Distribution of {column}')
    plt.xlabel(item)
    plt.ylabel("Count")
    plt.show()



# linear correlation between numerical columns (values -1 to 1)
print("correlations:")
print(data_filtered[list(numerical_features.keys())].corr())
# geo_altitude and baro_altitude are almost perfectly correlated : 0.999
# altitudes and velocity have strong correlation : 0.89
# vertical_rate is not correlated well with altitudes and velocity : 0.03-0.04



# Maps
# visualizing longitude, latitude on the World map
world_map = gpd.read_file("C:/Users/Kristo/Desktop/andme/project/data preparation/ne_110m_admin_0_countries.shp")

fig, ax = plt.subplots(figsize=(16, 8))
world_map.plot(ax=ax, color="lightgray", edgecolor="black")

ax.scatter(data_filtered["longitude"], data_filtered["latitude"], s=2, alpha=0.5)
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("Flight positions on the map")
plt.show()
# shows world map with flight positions on the map


# airplane positions, colored with geo altitude
fig, ax = plt.subplots(figsize=(16, 8))
world_map.plot(ax=ax, color="lightgray", edgecolor="black")

scatter = ax.scatter(data_filtered["longitude"], data_filtered["latitude"], c=data_filtered["geo_altitude"], cmap="viridis", s=1, alpha=1)

fig.colorbar(scatter, ax=ax, label="Geo Altitude (m)")
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("Air traffic flight positions on the map colored by geo_altitude")
plt.show()
# shows that flights in: 
# for example West Europe countries have geo_altitude concentrated between 10000-12500m
# Central Europe countries have geo_altitude concentrated between 0-2500m


# airplane positions, colored with velocity
fig, ax = plt.subplots(figsize=(16, 8))
world_map.plot(ax=ax, color="lightgray", edgecolor="black")

scatter = ax.scatter(data_filtered["longitude"], data_filtered["latitude"], c=data_filtered["velocity"], cmap="viridis", s=1, alpha=1)

fig.colorbar(scatter, ax=ax, label="Velocity (m/s)")
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("Air traffic flight positions on the map colored by velocity")
plt.show()
# shows that flights in:
# for example East Coast of the United States has velocity concentrated between 200-250 m/s
# Japan has velocity concentrated between 250-280 m/s




