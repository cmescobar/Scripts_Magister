import soundfile as sf
import numpy as np
import cvxpy as cp
import gurobipy as gp
import time


# Lectura de audio
audio_total, samplerate = sf.read('Comp_tests/Drums_zanark2.wav')
audio_drums, samplerate = sf.read('Comp_tests/Drums_zanark2_drums.wav')
audio_piano, samplerate = sf.read('Comp_tests/Drums_zanark2_piano.wav')

# Abriendo archivos de separación
n_comp = 400
data = np.load(f'Comp_tests/Separation_{n_comp}.npz')

# Recuperando las variables de interés
comps = data['comps']

# Se define la matriz A cuyas columnas son todas las componentes obtenidas a 
# partir de NMF
A = comps.T[:len(audio_total),:]

# Se define el límite de variables a activar
L = 25

'''# Se construye la variable
x = cp.Variable(A.shape[1], boolean=True)
z = cp.Variable(1)

# Se plantean las restricciones
print('Adding restrictions...')
constraints = [sum(x) <= L,
               A*x - audio_piano <= z,
               A*x - audio_piano >= -z]
print('Restrictions completed!')

# Declaración de la función objetivo
obj = cp.Minimize(z)

# Declaración del problema de optimización
opt_prob = cp.Problem(obj, constraints)

# Resolviendo...
print('Solving the problem...')
opt_prob.solve(solver=cp.GUROBI, verbose=True)

# Imprimiendo el resultado
print("\nThe optimal value is", opt_prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(opt_prob.constraints[0].dual_value)
print(f"Suma de x: {sum(x.value)}")'''


try:
    # Definir el modelo
    m1 = gp.Model("Compressed Sensing para Audio percusión con piano")
    
    # Definición de indices
    I, J = range(A.shape[0]), range(A.shape[1])
    
    # Creación de variables
    x = m1.addVars(J, vtype=gp.GRB.BINARY, name='x')
    z = m1.addVar(vtype=gp.GRB.CONTINUOUS, name='z')

    # Definición de la función objetivo
    m1.setObjective(z, gp.GRB.MINIMIZE)
    
    # Definición de las restricciones
    print(f"Escribiendo las restricciones...")
    begin_time = time.time()
    m1.addConstr(gp.quicksum(x) <= L, "Rest. Cantidad Activación")
    m1.addConstrs((gp.quicksum(A[i,j] * x[j] for j in J) - audio_piano[i] <= z
                  for i in I), name=f'Rest. Límite Sup')
    m1.addConstrs((gp.quicksum(A[i,j] * x[j] for j in J) - audio_piano[i] >= -z
                  for i in I), name=f'Rest. Límite Inf')
    end_time = time.time()
    print(f"¡Restricciones completadas!")
    print(f'Tiempo total de carga de restricciones = {end_time - begin_time}')
    
    
    # Partiendo rutina de optimización
    print("Optimizando...")
    m1.optimize()

#except gp.GurobiError as e:
#    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
    
for i in range(A.shape[1]):
    print(f'x[{i}] = {x[i].x}')

print('Obj: %g' % m1.objVal)