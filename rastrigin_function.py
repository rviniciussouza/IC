
# Equipe: Vinicius Rodrigues
#         Marcus Magalhaes
#         Daniel Veloso Braga

from scipy.optimize import differential_evolution
import numpy as np

def rastrigin(x):
        somatorio = 10 * len(x)
        for i in range(len(x)):
            somatorio += (x[i] ** 2) - (10 * np.cos(2 * np.pi * x[i]))    
        return somatorio

print(rastrigin([0,0,0,0]))

limites = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
result = differential_evolution(rastrigin, limites)
print(result.x)
print(result.fun)