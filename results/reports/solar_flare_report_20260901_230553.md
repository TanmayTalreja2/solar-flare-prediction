# ☀️ Solar Flare Prediction Report

---

## 🔭 Observation

**Active Region:** 11548
**Observation Time:** 2012-08-18 04:24:00
**Prediction Window:** Next 24 Hours

## 🚨 Model Prediction

### **81.80% predicted probability**

**Risk Level:** **VERY HIGH RISK**
**Model Decision:** FLARE RISK DETECTED

## 🧠 Interpretation

The model indicates a very high predicted risk of an M/X-class solar flare within the next 24 hours. The active region warrants close monitoring.

The reported probability is the probability predicted by the trained XGBoost model; it should not be interpreted as a guaranteed physical probability of a flare occurring.

## 🧲 Key Magnetic Indicators

The following features had the highest global importance among the model inputs for this prediction system:

- **Totpot Change 12H**    Value: `6.022e+22`    Model importance: `0.1335`
- **Usflux Relchange 6H**    Value: `57.153`    Model importance: `0.0740`
- **T Rec**    Value: `2012.08.18_04:24:00_TAI`    Model importance: `0.0612`
- **Meanpot Change 1H**    Value: `2695.743`    Model importance: `0.0411`
- **Target 24H Mag**    Value: `1.000`    Model importance: `0.0328`

## ⚙️ Method

The prediction was generated using an XGBoost model trained on 71 magnetic and temporal features derived from SHARP observations.

Missing values were handled using the median imputer fitted on the training data.

> **Important:** Model feature importance indicates which inputs were influential to the trained model overall. It does not imply a causal relationship.

---

Report generated: 2026-09-01 23:05:53