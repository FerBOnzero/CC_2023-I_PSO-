from numba import cuda
import numpy as np

def Rosenbrock(x):
    d = len(x) # Dimensión de nuestro espacio
    total_sum = 0
    for i in range(d-1):
        total_sum += 100*(-x[i]**2 + x[i+1])**2 + (x[i]-1)**2
    return total_sum

def f_x2y2(x): # función definida sólo para R^2 (dimensión 2)
    return x[0]**2 + x[1]**2

def PSO_GPU(n_P, d, l, u, f):
    # Inicialización de partículas y velocidades
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
    gbest = X[0] # toda la población

    for i in range(1, len(X)):
        new_val_f = f(X[i])
        if new_val_f < f(gbest):
            gbest = X[i] # Modificar el mejor de toda la población


    device = cuda.get_current_device()
    
    # El decorador @cuda.jit es usado para definir funciones que correran en la GPU 
    @cuda.jit('void(float64[:], float64[:], float64[:])')
    def cu_add1(a, b, c):
        # Esta función kernel será ejecutada por un thread
        bx = cuda.blockIdx.x # which block in the grid?
        bw = cuda.blockDim.x # what is the size of a block?
        tx = cuda.threadIdx.x # unique thread ID within a blcok
        i = tx + bx * bw

        if i > c.size:
            return

        c[i] = a[i] + b[i]

    stop_condition = 100 # Condición de paro
    while stop_condition > 0: 
        for i in range(len(X)):
            # Nueva velocidad
            V = V + np.random.rand(1)[0]*(pbest - X) + np.random.rand(1)[0]*(gbest - X) # X + V
            # mandar input vector a un device (GPU)
            X_temp = []
            for k in range(len(X)):
                # Assign equivalent storage on device
                a = X[k]
                b = V[k]
                # mandar input vector a un device (GPU)
                da = cuda.to_device(a)
                db = cuda.to_device(b)

                # Asignar storage al device para el outpu
                dc = cuda.device_array_like(a)

                # Set up suficientes threads para el kernel
                tpb = device.WARP_SIZE
                bpg = int(np.ceil(float(2)/tpb))

                # Correr el kernel
                cu_add1[bpg, tpb](da, db, dc)

                # Transfefirir salir de device a host
                c = dc.copy_to_host()
                X_temp.append(c)
            X = np.array(X_temp)
    
            if f(pbest[i]) > f(X[i]):
                pbest[i] = X[i]

        for i in pbest:
            # Evaluar función objetivo y elegir mejor partícula
            if f(i) < f(gbest):
                gbest = i
                
        stop_condition -= 1  
    return pbest, gbest, f(gbest)

pbest1, gbest1, f1 = PSO_GPU(20, 2, -5, 10, Rosenbrock)
print(f'Primer test con mínimo en [1,1] f=0, con algoritmo nos dice que la aproximación es {gbest1} y función valuada = {Rosenbrock(gbest1)}')

pbest2, gbest2, f2 = PSO_GPU(20, 2, -5, 10, f_x2y2)
print(f'Segundo test con mínimo en [0,0] f=0, con algoritmo nos dice que la aproximación es {gbest2} y función valua = {f_x2y2(gbest2)}')