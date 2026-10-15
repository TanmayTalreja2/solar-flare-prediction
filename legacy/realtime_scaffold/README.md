# Retired real-time scaffold

These files were an early simulated real-time prototype. They are kept for reference only and are not used.

Why they were retired:
- They load the original 5-parameter XGBoost model instead of the current 17-parameter one.
- They classify risk at 0.20 / 0.40 / 0.70, which disagrees with the production bands (0.01 / 0.05 / 0.20).
- They hardcode a 0.2 / 0.8 blend instead of the trained fusion model.
- Their feature math differs slightly from the training pipeline.

The current live pipeline is src/solarflare/realtime/live_pipeline.py.