# KAGGLE-OpenSky Aircraft States Snapshot
---
**Authors:** Kristo Hark

Global air traffic generates large volumes of real-time data that can be used to understand flight behavior, detect anomalies, and build predictive models. Analyzing this data helps reveal traffic patterns, identify unusual flights, and evaluate how well simple flight features explain aircraft altitude.

## Project's goals
* Handle missing or inconsistent values, visualize global air traffic patterns using maps and charts showing position (longitude, latitude), altitude, velocity.
* Group flights into clusters to identify traffic patterns and detect unusual or anomalous flights.
* Build a simple model (Decision Tree or Random Forest) to predict baro_altitude using other features like velocity, vertical_rate, longitude, latitude, on_ground.

---

## Guide to the contents of the repository

**All datasets**

**data/**

├── **altitude_diff_full_filtered.csv**

├── **altitude_diff_full.csv**

├── **filtered_final_data.csv** - Cleaned dataset

├── **opensky_states_snapshot.csv** - Original dataset

├── **velocity_filtered.csv** 

├── **velocity.csv**

├── **vertical_rate.csv**__




**images/**

├── **exploration/** - Additional figures produced during analysis

├── **poster/** - Only figures used in the final poster



**Code scripts for the project**

**src/**

├── **worldmap/** - Helper files for map visualizations in Goal 1

├── **G1_data_preparation.py** - Goal 1

├── **G2_clustering.py** - Goal 2

├── **G3_prediction.py** - Goal 3



**README.md**

---

## Replicating the analysis

1. **Install dependencies**
2. **Run the scripts in order G1 -> G2 -> G3.**
3. **Each script is documented with section headers.**

<summary><strong> G1_data_preparation.py </strong></summary>

* **Section 1** - Data understanding & exploration
* **Section 2** - Handling missing & inconsistent values
* **Section 3** - Feature validation
* **Section 4** - Sanity checks
* **Section 5** - Advanced visualizations (maps, altitude/velocity charts)


<summary><strong> G2_clustering.py </strong></summary>

* **Section 1** - Feature selection
* **Section 2** - Data normalization & preprocessing 
* **Section 3** - K-means clustering exploration
* **Section 4** - DBSCAN clustering exploration
* **Section 5** - Cluster analysis & reports


<summary><strong> G3_prediction.py </strong></summary>

* **Section 1** - Data preparation
* **Section 2** - Model selection (Decision Tree, Random Forest)
* **Section 3** - Feature engineering
* **Section 4** - Tuning the models
* **Section 5** - Improving Random Forest with new features
* **Section 6** - Final reporting & documentation
