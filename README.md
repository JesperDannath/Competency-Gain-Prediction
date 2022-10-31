# Competency-Gain-Prediction

Die aktuelle Funktionalität kann über simulation_experiments.ipynb nachvollzogen werden

## Bekannte Probleme
- Langsamer E-step für das Gain Modell
- Convergence Probleme bei Newton-Raphson Methode
- Schlechte Ergebnisse für Gain-Prediction


## Optimization Roadmap
- Implementierung von switch zwischen constrained und unconstrained optimization (done)
- Check Quality of competency fitting, switch between real competency and estimated or baseline (done)
- Improve Callbacks + switch
- Implementierung von Regularisierungen für Newton-Raphson Methode
- Implementierung von close spd matrix für reset-projection method
- Go through simulation framework, especially correct answer rate is too high

## Update Roadmap
- Beselines Framework (done)
- Trying real theta as upper baseline (done)
- Repeating simulation Experiments (done)
- Gain Prediction

