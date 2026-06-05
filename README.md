# Predicting High-Impact Power Outages

**Name:** David Yang

This project analyzes major power outage events in the continental United States from 2000 to 2016. I focus on customer impact: which outages affected unusually many customers, and which location, climate, cause, and infrastructure characteristics are associated with those outages.

## Introduction

The dataset contains **1,534 cleaned outage observations** from **2000-2016**. My main question is:

**What factors are associated with major power outages that affect unusually many customers?**

For prediction, I define `HIGH_IMPACT` as whether `CUSTOMERS.AFFECTED` is above the median observed customer impact of **70,135** customers.

| Column | Description |
|---|---|
| `CUSTOMERS.AFFECTED` | Number of customers affected by the outage; used to define `HIGH_IMPACT`. |
| `CAUSE.CATEGORY` | Broad cause of the outage, such as severe weather or intentional attack. |
| `U.S._STATE` | State where the outage occurred. |
| `MONTH` | Month of the outage, used to study seasonality. |
| `CLIMATE.CATEGORY` | Whether climate conditions were cold, normal, or warm. |
| `TOTAL.CUSTOMERS`, `POPULATION` | Area size and exposure variables used in the final model. |

## Data Cleaning and Exploratory Data Analysis

The raw data is an Excel workbook. The first five rows are metadata, so I loaded the sheet with `header=5`. The first loaded row contains units rather than an outage observation, so I removed it by keeping rows with numeric `YEAR` values. I converted relevant numeric columns, combined `OUTAGE.START.DATE` and `OUTAGE.START.TIME` into `OUTAGE.START`, and combined the restoration date and time columns into `OUTAGE.RESTORATION`.

### Head of Cleaned DataFrame

| YEAR | MONTH | U.S._STATE | CAUSE.CATEGORY | CUSTOMERS.AFFECTED | OUTAGE.DURATION |
|---:|---:|---|---|---:|---:|
| 2011 | 7 | Minnesota | severe weather | 70,000 | 3,060 |
| 2014 | 5 | Minnesota | intentional attack | NaN | 1 |
| 2010 | 10 | Minnesota | severe weather | 70,000 | 3,000 |
| 2012 | 6 | Minnesota | severe weather | 68,200 | 2,550 |
| 2015 | 7 | Minnesota | severe weather | 250,000 | 1,740 |

The `NaN` value in 2014 is kept because it is genuine missing data in the original dataset, and missingness is part of the analysis.

### Univariate Analysis

<iframe src="assets/cause_counts.html" width="100%" height="520" frameborder="0"></iframe>

Severe weather is the most common cause category by a wide margin, followed by intentional attacks. This shows that many major outages are weather-related, but the dataset also includes many human-caused events.

<iframe src="assets/month_counts.html" width="100%" height="520" frameborder="0"></iframe>

Outages are most common during summer months, especially June and July. This suggests seasonality is important when studying power outage risk.

### Bivariate Analysis

<iframe src="assets/median_by_cause.html" width="100%" height="520" frameborder="0"></iframe>

Severe weather outages have the highest median customer impact. This motivates using cause category in the later hypothesis test and predictive model.

<iframe src="assets/state_map.html" width="100%" height="520" frameborder="0"></iframe>

California and Texas have the largest number of recorded major outages in this dataset. Geography is therefore relevant to understanding outage frequency and severity.

### Grouped Summary by Cause Category

| Cause Category | Outage Count | Median Customers Affected | High-Impact Rate |
|---|---:|---:|---:|
| severe weather | 763 | 110,433 | 0.688 |
| system operability disruption | 127 | 69,000 | 0.482 |
| equipment failure | 60 | 45,452 | 0.333 |
| islanding | 46 | 2,343 | 0.000 |
| fuel supply emergency | 51 | 0 | 0.000 |
| intentional attack | 418 | 0 | 0.010 |
| public appeal | 69 | 0 | 0.000 |

This grouped table shows that severe weather is not only common, but also associated with larger customer impacts.

## Assessment of Missingness

I chose `CUSTOMERS.AFFECTED` for missingness analysis because it is my main severity variable and is missing in about **28.9%** of rows.

I believe this column could be **NMAR**: agencies may be less likely to report customer counts when the count itself is unknown, extremely small, or difficult to estimate during certain outage types. Additional data about utility reporting rules, whether customer impact was required in the incident report, and how estimates were collected could help explain the missingness and make it closer to MAR.

<iframe src="assets/missing_by_cause.html" width="100%" height="520" frameborder="0"></iframe>

I used total variation distance to compare the distribution of `CAUSE.CATEGORY` for rows where `CUSTOMERS.AFFECTED` was missing versus observed. The permutation test produced a very small p-value, so I found evidence that missingness depends on cause category. I also tested whether missingness depends on `CLIMATE.CATEGORY`; that p-value was large, so I did not find evidence that missingness depends on climate category.

## Hypothesis Testing

I tested whether severe-weather outages affect more customers than non-severe-weather outages.

| Component | Choice |
|---|---|
| Null Hypothesis | Severe-weather and non-severe-weather outages have the same distribution of `CUSTOMERS.AFFECTED`. |
| Alternative Hypothesis | Severe-weather outages tend to affect more customers than non-severe-weather outages. |
| Test Statistic | Difference in mean customers affected: severe weather minus non-severe weather. |
| Significance Level | 0.05 |
| Result | Observed difference: about 131,616 customers. Permutation test p-value: approximately 0.0000. |

I rejected the null hypothesis. This suggests severe-weather outages tend to affect more customers in this dataset, though the test does not prove a causal relationship.

## Framing a Prediction Problem

The prediction task is **binary classification**: predict `HIGH_IMPACT`, where an outage is high-impact if `CUSTOMERS.AFFECTED` is above the median observed customer impact of 70,135 customers.

I use F1-score as the main metric because both false positives and false negatives matter, and the task is more informative when precision and recall are balanced than when accuracy alone is optimized.

At the time of prediction, I assume an energy company would know the outage month, location, climate and regional context, electricity consumption/economic characteristics, and the reported cause category. I do not use `CUSTOMERS.AFFECTED` itself or restoration-time information as predictors, because those are outcomes known after the outage impact is measured.

## Baseline Model

The baseline model is a logistic regression pipeline using two original features: `MONTH` and `CLIMATE.CATEGORY`. It contains **0 quantitative features, 0 ordinal features, and 2 nominal features**. Both features are one-hot encoded, with missing categories imputed using the most frequent value.

Its test F1-score is **0.498** and accuracy is **0.535**, so I do not consider it a strong model. It captures seasonality and broad climate conditions, but it misses important differences in outage cause, location, grid region, and area size.

## Final Model

The final model is still logistic regression, but it uses a richer sklearn pipeline. I add original features such as cause category, state, NERC region, climate region, climate anomaly, population, electricity sales, and electricity prices.

I also engineer `SEASON`, `LOG_TOTAL_CUSTOMERS`, `LOG_POPULATION`, and `CUSTOMERS.PER.PERSON`. These features are useful because outage impact should depend on when the outage occurs, how many customers are exposed, and the scale of the local population and electric system.

I tuned logistic regression regularization strength `C` and `class_weight` using 5-fold `GridSearchCV` with F1-score. The best hyperparameters were `C=10` and `class_weight="balanced"`.

| Model | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline Model | 0.535 | 0.498 | 0.538 | 0.463 |
| Final Model | 0.736 | 0.750 | 0.711 | 0.794 |

The final model improves the F1-score from 0.498 to 0.750.

## Fairness Analysis

I compared accuracy for Group X, outages during `warm` climate periods, against Group Y, outages during non-warm climate periods.

| Component | Choice |
|---|---|
| Group X | `CLIMATE.CATEGORY == "warm"` |
| Group Y | `CLIMATE.CATEGORY != "warm"` |
| Metric | Accuracy parity |
| Test Statistic | Absolute difference in accuracy between the two groups |
| Null Hypothesis | The final model is fair: its accuracy for warm and non-warm climate-category outages is roughly the same, and any observed difference is due to chance. |
| Alternative Hypothesis | The final model is unfair: its accuracy differs between warm and non-warm climate-category outages. |
| Significance Level | 0.05 |
| Result | Observed absolute difference: about 0.085. Permutation test p-value: about 0.2312. |

Because the p-value is larger than 0.05, I failed to reject the null hypothesis. I did not find strong evidence that the final model is unfair between warm and non-warm climate-category outages using accuracy parity.
