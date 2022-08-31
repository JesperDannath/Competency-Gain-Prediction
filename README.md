# Knowledge-Growth-Prediction

Die aktuelle Funktionalität kann über simulation_experiments.ipynb nachvollzogen werden

Bekannte Probleme:
- EM-Algorithmus ist langsam
- Newton-Raphson Methode für Q_0 noch nicht implementiert
- Kovarianzmatrix wird zwangsläufig symmetrisch geschätzt
- Tests laufen wegen import-fehlern aktuell nicht mehr durch


## Optimzation Roadmap

- Add Q-Matrix (less parameters, Identification)
- Add better Parameter Initialization (smaller search-space)
- Add Newton-Raphson Method for 
- Add vectorization in Monte Carlo Integral calculation (better utilization of compute ressources)
- Add multiprocessing for the m_step in q_item optimzation (possible J-fold performance increase)

## Update Roadmap

- Add Q-Matrix Constraint (Identification, Modelling)
- Add MIRT-Gain Model
- Add M-Step and E-Step for MIRT-Gain
