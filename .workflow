name: Preprocessing Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  preprocess:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pandas numpy scikit-learn

      - name: Run preprocessing
        run: |
          python preprocessing/automate_Mhd_Rifan_Baihaqie.py

      - name: Commit preprocessed data
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add california_housing_raw/ preprocessing/california_housing_preprocessing/
          git diff --cached --quiet || git commit -m "Auto: update preprocessed dataset"
          git push
