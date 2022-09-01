# Knowledge-Growth-Prediction

Die aktuelle Funktionalität kann über simulation_experiments.ipynb nachvollzogen werden

## Bekannte Probleme
- EM-Algorithmus ist langsam
- Newton-Raphson Methode für Q_0 noch nicht implementiert
- Tests laufen wegen import-fehlern aktuell nicht mehr durch


## Optimization Roadmap

- Add better Parameter Initialization (smaller search-space, Q*U-Methode wäre okay oder Q^T*Q)
- Add Newton-Raphson Method for 
- Add vectorization in Monte Carlo Integral calculation (better utilization of compute ressources)
- cython für das Integral benutzen
- Add multiprocessing for the m_step in q_item optimzation (possible J-fold performance increase) (eher nicht)

## Update Roadmap

- Add MIRT-Gain Model
- Add M-Step and E-Step for MIRT-Gain
