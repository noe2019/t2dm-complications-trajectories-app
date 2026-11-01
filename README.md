# Type 2 Diabetes Complication Trajectories

Graph analysis of the order in which complications appear after a type 2 diabetes diagnosis.

**Live app:** https://noe-t2dm-trajectories-094c80ceabf2.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- Directed transition network between complications
- Sankey diagram of the first three complications
- Most frequent trajectories and time between complications
- Filters by sex and age at diagnosis

## Tech stack
Python, NetworkX, Streamlit, Plotly

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
