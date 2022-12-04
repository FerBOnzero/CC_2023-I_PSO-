# Descripción de la carpeta

En la presente carpeta se incluye el código con extensión .ipynb de la implementación PSO secuencial con Python, al igual que el código con extensión .py de la implemetación en paralelo con GPU's. Cada código tiene dos pruebas de ejemplo, en ambas son las mismas funciones a evaluar.

### Consideraciones a correr el código
- Se requiere tener instalada la librería **Numpy** de Python en una versión 1.21 o inferior, esto para poder correr Cuda con Numba.
- Se requiere tarjeta gráfica nvidia en tu equipo para correr código.

### Consideración propias respecto al algoritmo PSO
- Para elegir la mejor partícula, se declaró que la mejor sería aquella que minimice la función objetivo, así que la implementación de la mejor partícula como el máximo es similar.
- Se eligieron dos funciones objetivos con mínimos globales conocidos para las pruebas.

### Observaciones acerca de la implementación en paralelo
- Se decidió paralelizar la actualización de cada partícula, es decir, cuál es la nueva posición de cada partícula en cada iteración.
- Podemos correr Cuda con Numba.
- Cuda tiene un decorador, el cual es @cuda.jit que es usado para definir funciones que correran en la GPU.
- Dicho decorador se pone a la "función kernel" la cuál se define de tal manera ya que es la que se espera correr en GPU, o bien, se piensa como aquella función que tiende a repetirse con "mucha" frecuencia. Además dicha función no presenta un return.
- En la función kernel inicializamos los threads, tales que se correran en paralelo. Esto ya nos lo facilita Numba Cuda simplemente definiéndolo.
- Ya que queremos hacer uso de la función kernel, necesitamos decir qué recursos, o bien, variables mandaremos a la función kernel y la salida del mismo que con posteriedad la usaremos. Esto lo hacemos con la sentencia "cuda.to_device(<source>)" para mandar a los GPU's y "cuda.device_array_like(<source>)" para almacenar la salida de la función kernel.

#### Referencias utlizadas y para más información: 
- https://people.duke.edu/~ccc14/sta-663/CUDAPython.html
- https://numba.pydata.org/numba-doc/dev/cuda/overview.html
- https://towardsdatascience.com/cuda-by-numba-examples-1-4-e0d06651612f
- https://thedatafrog.com/en/articles/boost-python-gpu/
- https://developer.nvidia.com/how-to-cuda-python
- https://medium.com/@gbadahamza18/exploit-your-gpu-by-parallelizing-your-codes-using-python-2dd8e2215aa8
- https://www.geeksforgeeks.org/running-python-script-on-gpu/
