# Resolución del problema del Agente viajero por medio de Algoritmos Genéticos, distribuyendo la carga de trabajo en varias computadoras con MPI y paralelizandolos con OMP.


## Algoritmos Genéticos
Son algoritmos metaheuristicos de busqueda inspirados en lo que sabemos acerca del proceso de la evolución humana. Se hicieron muy populares ya que hacen posible abordar problemas en los que es muy difícil aplicar procedimientos matemáticos tradicionales.

En la actualidad, los AG son preferentemente utilizados como métodos de búsqueda de soluciones óptimas que simulan la evolución natural y han sido usados con éxito en la solución de problemas de optimización combinatoria, optimización de funciones reales y como mecanismos de aprendizaje de maquina (machine learning).

“La selección natural obra solamente mediante la conservación y acumulación de pequeñas modificaciones heredadas, provechosas todas al ser conservado” escribio Darwin y dio nombre a este proceso (El origen de las especies, selección natural). Estas “modificaciones heredadas”, señaladas por Darwin como las generadoras de organismos mejores, son llamadas mutaciones hoy en día y constituyen el motor de la
evolución.

Un organismo mutante ha sufrido una modificación que lo hace diferente al resto de sus congéneres. Esta modificación puede ser un inconveniente para él (la falta de un miembro útil de su cuerpo, por ejemplo), pero puede ocurrir también que le confiera algúna cualidadque le permita sobrevivir más facilmente que al resto de los individuos de su especie. Este organismo tendrá mayor probabilidad de reproducirse y heredará a sus descendientes la caracteristica que le dió ventaja. Con el tiempo los organismos que al principio eran raros se volverán comunes a costa de la desaparición del “modelo anterior”. Se habrá dado entonces un paso en el proceso evolutivo.

Para imitar el proceso de evolución, se parte de una población inicial de la cual se seleccionan los individuos más capacitados para luego reproducirlos y mutarlos para finalmente obtener la siguiente generación.

##Problema del Agente viajero (TSP)
En el Problema del Agente Viajero - TSP (Travelling Salesman Problem), el objetivo es visitar todas las ciudades de un conjunto de ciudades, pasando por cada ciudad solamente una vez, volviendo al punto de partida, y que además minimice el costo total de la ruta.

### Compilación
	mpic++ viajero_mpi_final_float.cpp -o viajero_mpi_final_float

### Ejecución
	mpirun -np 32 -hostfile ./hostfile.txt ./viajero_mpi_final_float < config.txt

#### Notas
 - config.txt es el archivo que contiene la matriz de distancias
 - Los parametros de configuracion se cambian desde el codigo, es importante cambiar el valor de la constante NUM_CIUDADES cuando se quiera leer una matriz de distancias de tamaño diferente al establecido en el código.
