# Solar Flare Prediction Report

---

## Observation

**Active Region:** 11579
**Observation Time:** 2012-09-26 00:48:00
**Prediction Window:** Next 24 Hours

## Model Prediction

**Flare Probability:** 99.95%
**Risk Level:** **VERY HIGH RISK**
**Model Decision:** FLARE RISK DETECTED

## Interpretation

The model indicates a very high likelihood of an M/X-class solar flare within the next 24 hours. The active region warrants close monitoring.

The prediction threshold was exceeded, so the model classified this observation as a positive flare-risk prediction.

## Important Model Inputs

The following features had the highest global importance in the trained XGBoost model for this prediction system:

- **Harpnum**: `2059.0000` (model importance: 0.1455)
- **Usflux**: `21690100000000000917504.0000` (model importance: 0.1180)
- **Totusjh Change 6H**: `88.7060` (model importance: 0.0562)
- **Totusjh**: `973.1270` (model importance: 0.0394)
- **Noaa Ar**: `11579.0000` (model importance: 0.0391)

## Method

The prediction was generated using the trained XGBoost model using the project's temporal magnetic field features. Missing values were handled using the median imputer fitted during model training.

> **Note:** Feature importance indicates which inputs were influential to the trained model overall. It should not be interpreted as a causal relationship.

---

Report generated: 2026-08-24 23:46:06