# 🌞 Hybrid Solar Flare Prediction System

A hybrid machine learning system for predicting solar flare activity using both **solar active-region features** and **solar magnetogram images**.

The project combines:

- XGBoost for temporal and tabular solar features
- ResNet18 CNN for solar magnetogram images
- A weighted hybrid ensemble
- A real-time prediction pipeline
- Human-readable solar flare risk classification

---

# 📌 Project Overview

Solar flares are sudden releases of energy from the Sun that can affect space weather, satellite systems, communication systems, and other technological infrastructure.

This project develops a hybrid prediction system that analyzes two different types of solar observations:

1. Numerical active-region features
2. Magnetogram images of solar active regions

The outputs of two machine learning models are combined to produce a final solar flare probability and risk level.

## Hybrid Architecture

```text
                         SOLAR OBSERVATION
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        SHARP / Temporal Features        Magnetogram Image
                 │                             │
                 ▼                             ▼
              XGBoost                      ResNet18 CNN
                 │                             │
                 ▼                             ▼
        Tabular Probability           Image Probability
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                         HYBRID ENSEMBLE
                                │
                                ▼
                    SOLAR FLARE RISK PREDICTION
