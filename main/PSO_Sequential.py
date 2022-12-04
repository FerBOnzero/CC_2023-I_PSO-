import numpy as np

# Función objetivo
def Rosenbrock(x):
    d = len(x) # Dimensión de nuestro espacio
    total_sum = 0
    for i in range(d-1):
        total_sum += 100*(-x[i]**2 + x[i+1])**2 + (x[i]-1)**2
    return total_sum

# Función objetivo extra para pruebas con mínimo global en (0,0) y f = 0
def f_x2y2(x): # función definida sólo para R^2 (dimensión 2)
    return x[0]**2 + x[1]**2

# Implementación secuencial
def PSO_sequential(n_P, d, l, u, f):
    # Inicialización 
    X = np.zeros((n_P, d))
    V = np.zeros((n_P, d))
    
    for i in range(n_P):
        for j in range(d):
            X[i][j] += l + np.random.rand(u-l)[0]
    
    for i in range(n_P):
        for j in range(d):
            V[i][j] += np.random.rand(1)
            
    # Evaluar la población en la función objetivo y elegir la mejor partícula
    pbest = X # local
    gbest_f = f(X[0]) # global
    gbest = X[0]
    
    for i in range(1,len(X)):
        new_val_f = f(X[i])
        if new_val_f < f(X[0]):
            gbest_f = new_val_f
            gbest = X[i]
            
    # while (condición de paro)
    stop_condition = 500 
    while stop_condition > 0: 
        for i in range(len(X)):
            # Nueva velocidad  
            V = V + np.random.rand(1)[0]*(pbest - X) + np.random.rand(1)[0]*(gbest - X)
            # Nueva posición
            X = X + V
                
            if f(pbest[i]) > f(X[i]):
                pbest[i] = X[i]
                
        for i in pbest:
            # Evaluar función objetivo y elegir mejor partícula
            if f(i) < f(gbest):
                gbest = i
                
        stop_condition -= 1

    return pbest, gbest, f(gbest) 

# test con función de Rosenbrock
pbest1, gbest1, f1 = PSO_sequential(20, 2, -5, 10, Rosenbrock)
print(f'Primer test con mínimo en [1,1] f=0, con algoritmo nos dice que la aproximación es {gbest1} y función valuada = {Rosenbrock(gbest1)}')

# test con la función f = x^2 + y^2
pbest2, gbest2, f2 = PSO_sequential(20, 2, -5, 10, f_x2y2)
print(f'Segundo test con mínimo en [0,0] f=0, con algoritmo nos dice que la aproximación es {gbest2} y función valua = {f_x2y2(gbest2)}')