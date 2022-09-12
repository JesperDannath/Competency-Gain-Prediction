# Knowledge-Growth-Prediction

Die aktuelle Funktionalität kann über simulation_experiments.ipynb nachvollzogen werden

## Bekannte Probleme
- Newton-Raphson Methode für Q_0 noch nicht implementiert
- Für hohe Fallzahlen ist die Methode zu langsam und konvergiert nicht gut


## Optimization Roadmap
- Add Newton-Raphson Method for 
- Vecorization for Q-Item such that monte carlo interation is paralell for different parameter-sets (Better scaling with ga pop-size)
- cython für das Integral benutzen
- Add multiprocessing for the m_step in q_item optimzation (possible J-fold performance increase) (eher nicht)

## Update Roadmap

- Add MIRT-Gain Model
- Add M-Step and E-Step for MIRT-Gain
