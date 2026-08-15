# solar-flare-prediction

\# Solar Flare Prediction Using Machine Learning



A machine learning system for predicting whether a significant solar flare (M/X class) is likely to occur within the next 24 hours using solar magnetic-field observations from NASA/JSOC HMI SHARP data and flare observations from the GOES satellite dataset.



\## Project Overview



Solar flares are sudden releases of energy from the Sun that can affect space weather and potentially disrupt satellite communication, navigation systems, and other technological infrastructure.



This project aims to develop a data-driven forecasting pipeline that:



1\. Collects solar magnetic-field observations.

2\. Integrates them with historical GOES flare observations.

3\. Creates 24-hour flare prediction labels.

4\. Engineers machine-learning features.

5\. Trains and evaluates classification models.

6\. Provides interpretable predictions through a future dashboard.

7\. Supports periodic retraining using a rolling historical window.



\## Current Pipeline



```text

HMI SHARP Data

&#x20;     |

&#x20;     v

Data Collection

&#x20;     |

&#x20;     v

GOES Flare Data

&#x20;     |

&#x20;     v

Temporal + Active Region Alignment

&#x20;     |

&#x20;     v

24-Hour M/X Flare Labels

&#x20;     |

&#x20;     v

Data Quality Control

&#x20;     |

&#x20;     v

Feature Engineering

&#x20;     |

&#x20;     v

Leakage-Aware Evaluation

&#x20;     |

&#x20;     v

XGBoost Classification

&#x20;     |

&#x20;     v

Prediction + Future Dashboard

Data Sources

HMI SHARP



Solar magnetic-field parameters are obtained from the HMI SHARP CEA data series through the Joint Science Operations Center (JSOC).



Important parameters currently used include:



USFLUX

TOTUSJH

TOTPOT

MEANPOT

MEANSHR

GOES



GOES X-ray flare observations are used to generate the prediction target.



The current target is:



target\_24h = 1



when an M-class or X-class flare begins within 24 hours after a SHARP observation.



Otherwise:



target\_24h = 0

Feature Engineering



The current baseline uses:



Magnetic Features

USFLUX

TOTUSJH

TOTPOT

MEANPOT

MEANSHR

Log-Transformed Features

LOG\_USFLUX

LOG\_TOTUSJH

LOG\_TOTPOT

LOG\_MEANPOT

Temporal Features

observation\_hour

day\_of\_year



Missing magnetic values are handled using median imputation during model training. The imputer is fitted only on the training data to prevent data leakage.



30-Day Development Dataset



The current development dataset covers March 2012.



After alignment and quality control:



21,523 observations

24 active regions

20,010 negative observations

1,513 positive observations

Positive rate: 7.03%



The dataset is highly imbalanced, so accuracy is not considered the primary evaluation metric.



The main evaluation metrics are:



PR-AUC

ROC-AUC

Precision

Recall

F1-score

Evaluation Strategy



A major focus of the project is preventing data leakage.



Initial experiments using random train/test splitting produced unrealistically high performance because observations from the same solar active regions could appear in both training and testing sets.



Therefore, the project is moving toward:



Chronological evaluation

Active-region-aware evaluation

Unseen active-region testing



The 30-day dataset also demonstrated that a naive chronological split can produce a test period containing no positive flare events. This is being addressed by expanding the historical dataset and designing a more robust evaluation strategy.



Machine Learning



The current baseline model is:



XGBoost Classifier



Class imbalance is handled using:



scale\_pos\_weight



The final model architecture and hyperparameters will be refined after the larger historical dataset has been prepared.



Current Development Status

Environment Setup              ✅

GOES Data Collection           ✅

SHARP Data Collection          ✅

30-Day SHARP Dataset           ✅

GOES/SHARP Alignment           ✅

Data Quality Control           ✅

Data Cleaning                  ✅

Feature Engineering            ✅

Initial XGBoost Baseline       ✅

Leakage Investigation          ✅





90-Day Historical Dataset      🚧

Robust Model Evaluation        🚧

Model Optimization             🚧

Prediction Interface           🚧

Dashboard                      🚧

Rolling 90-Day Retraining      🚧

Human-Readable Reports         🚧

Planned System



The final system is planned to include:



Prediction Engine



Predict the probability of an M/X-class flare occurring within the next 24 hours.



Dashboard



A visualization dashboard will display:



Current solar active-region information

Magnetic-field features

Predicted flare probability

Risk classification

Historical predictions

Model performance

Data freshness

Retraining status

Rolling Retraining



The planned production architecture will use a rolling historical window.



Conceptually:



Current Time

&#x20;    |

&#x20;    v

Previous 90 Days

&#x20;    |

&#x20;    v

Data Update

&#x20;    |

&#x20;    v

Model Retraining

&#x20;    |

&#x20;    v

Validation

&#x20;    |

&#x20;    v

Latest Model

&#x20;    |

&#x20;    v

Dashboard

Human-Readable Reports



The final system is also intended to generate human-readable prediction reports summarizing:



Predicted risk

Probability

Important contributing features

Active-region information

Model version

Data period used for training

Prediction timestamp

Project Structure

solar-flare-prediction/

│

├── scripts/

│   ├── data collection

│   ├── GOES processing

│   ├── SHARP processing

│   ├── alignment

│   ├── quality control

│   ├── feature engineering

│   └── model training/evaluation

│

├── src/

│   └── solarflare/

│

├── tests/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│

├── reports/

│

├── requirements.txt

└── README.md

Development



Create the virtual environment:



python -m venv .venv



Activate it on Windows:



.venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt

Project Goal



The long-term goal is to develop a reproducible solar-flare forecasting system that combines physical solar observations with machine learning and presents the resulting predictions through an accessible dashboard and human-readable reporting system.

