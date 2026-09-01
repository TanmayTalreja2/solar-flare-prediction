# ☀️ Solar Flare Prediction Report

---

## 🔭 Observation

**Active Region:** 11579
**Observation Time:** 2012-09-26 00:48:00
**Prediction Window:** Next 24 Hours

## 🚨 Model Prediction

### **99.95% predicted probability**

**Risk Level:** **VERY HIGH RISK**
**Model Decision:** FLARE RISK DETECTED

## 🧠 Interpretation

The model indicates a very high predicted risk of an M/X-class solar flare within the next 24 hours. The active region warrants close monitoring.

The reported probability is the probability predicted by the trained XGBoost model; it should not be interpreted as a guaranteed physical probability of a flare occurring.

## 🧲 Key Magnetic Indicators

The following features had the highest global importance among the model inputs for this prediction system:

- **Total Unsigned Magnetic Flux**    Value: `2.169e+22`    Model importance: `0.1180`
- **Totusjh Change 6H**    Value: `88.706`    Model importance: `0.0562`
- **Total Current Helicity**    Value: `973.127`    Model importance: `0.0394`
- **Meanshr**    Value: `30.128`    Model importance: `0.0326`
- **Cmask**    Value: `31307.000`    Model importance: `0.0313`

## ⚙️ Method

The prediction was generated using an XGBoost model trained on 71 magnetic and temporal features derived from SHARP observations.

Missing values were handled using the median imputer fitted on the training data.

> **Important:** Model feature importance indicates which inputs were influential to the trained model overall. It does not imply a causal relationship.

---

Report generated: 2026-08-25 09:37:38