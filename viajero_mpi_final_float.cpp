/**
 * Autor: Martín Alejandro Pérez Güendulain
 * Para compilar el programa:
 * 		mpic++ viajero_mpi_final_float.cpp -o viajero_mpi_final_float
 * Para ejecutarlo:
 * 		mpirun -np 32 -hostfile ./hostfile.txt ./viajero_mpi_final_float < config.txt
 * 	config.txt es el archivo que contiene la matriz de distancias
 * 	Los parametros de configuracion se cambian desde el codigo, es importante cambiar 
 * 	NUM_CIUDADES cuando se quiera leer una matriz de distancias de tamaño diferente 
 * 	al establecido en el codigo.
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

using namespace std;

enum EstrategiaDeSeleccion { aleatoria, rank, torneo };
#define INFINITO 2147483647
#define NUM_MAX_CIUDADES 50
#define NUM_CIUDADES 6
#define NUM_GENERACIONES 10
#define NUM_HILOS 32
char nombre_archivo_reporte[] = "graficas.m";
int TAM_GENERACION = 1600;
float NUM_ELEM_A_MUTAR = 5;
float distancias[NUM_MAX_CIUDADES][NUM_MAX_CIUDADES];
EstrategiaDeSeleccion ESTRATEGIA_DE_SELECCION = aleatoria;

/**
 * Estructura que representa una posible solcion del problema del agente viajero
 */
typedef struct
{
	int		ruta[NUM_MAX_CIUDADES];
	float 	costo_de_ruta;
} Solucion;

/**
 * printConfig
 * Imprime la configuracion con la que va trabajar el programa, como lo es:
 * - Numero de ciudades
 * - Numero de Generaciones
 * - Numero de Hilos que se ocuparan para evaluar una generacion
 * - etc.
 */
void printConfig();

/**
 * readConfig
 * Lee la matriz de distancias.
 */
void readConfig();

/**
 * printGeneracion
 * Imprime la Generacion que se le pasa como pasametro.
 */
void printGeneracion(Solucion* generacion);

/**
 * printSolucion
 * Imprime una posible solucion mandada como parametro.
 */
void printSolucion(Solucion sol);

/**
 * compara
 * Funcion que compara dos soluciones (se ocupa para ordenar desendentemente un arreglo de Soluciones).
 */
int compara(const void *s1, const void *s2);
/**
 * comparaIntMayorAMenor
 * Funcion que compara dos enteros (se ocupa para ordenar desendentemente un vector de enteros).
 */
bool comparaIntMayorAMenor(int a, int b);

/**
 * inicializar
 * Retorna un arreglo con la generacion inicial.
 */
Solucion* inicializar();

/**
 * getMemoria
 * Reserva memoria para una generacion de tamaño TAM_GENERACION.
 */
Solucion* getMemoria();

/**
 * getRutaRandom
 * Retorna una ruta aleatoria.
 */
int* getRutaRandom();
/**
 * evaluaSolucion
 * Retorna el costo de una ruta.
 */
float evaluaSolucion(int* ruta);
/**
 * evaluaGeneracion
 * Evalua la generacion pasada como parametro.
 */
void evaluaGeneracion(Solucion* generacion, int tam_generacion);
/**
 * cruzaADNs
 * Funcion de cruza de los Algoritmos Geneticos.
 */
void cruzaADNs(int* ADN1, int* ADN2, Solucion* nueva_generacion, int pos_insert_nuevos_adns);
/**
 * cruzaGeneracion
 * Aplica la cruza a la generacion que se le pasa como parametro y retorna una nueva generacion.
 */
void cruzaGeneracion(Solucion* generacion);
/**
 * mutarGeneracion
 * Muta NUM_ELEM_A_MUTAR elementos de la generacion pasada como parametro.
 */
void mutarGeneracion(Solucion* generacion);
/**
 * mutarADN
 * Muta un individuo de una generacion.
 */
void mutarADN(int* ADN);
/**
 * seleccionAleatoria
 * Aplica la estrategia de seleccion Aleatoria.
 */
pair< int, int > seleccionAleatoria(int limite_gen);
/**
 * seleccionAleatoria
 * Aplica la estrategia de seleccion Rank.
 */
pair< int, int > seleccionRank();
/**
 * seleccionAleatoria
 * Aplica la estrategia de seleccion por Torneo.
 */
pair< int, int > seleccionTorneo();
/**
 * getPosition
 * Funcion auxiliar para la seleccion Rank que retorna una posicion en la generacion.
 */
int getPosition(int t, vector< float > ini_rebanada, int tam_gen);
/**
 * getIndicesMaxDeGrupos
 * Funcion auxiliar que retorna un pair con los indices más grandes de cada grupo pasado como parametro.
 */
pair< int, int > getIndicesMaxDeGrupos(vector< int > grupo1, vector< int > grupo2);
/**
 * sumatoria
 * Calcula la sumatoria de n.
 */
int sumatoria(int n);
/**
 * generaArchivoMejoresSol
 * Genera un arhico de Matlab con el codigo para graficar las mejores soluciones en cada generacion.
 */
void generaArchivoMejoresSol(vector<float> mejores_sol);

int main(int argc, char** argv)
{
	double starttime, endtime;
	starttime = MPI_Wtime();

	srand(time(NULL));

	MPI_Init(&argc, &argv);
	int  size, rank;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, & size);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);


	// char s[50] ;
	// sprintf(s,"salida%02d.txt", rank );
	// freopen(s, "w", stdout);

	readConfig();
	Solucion* generacion = inicializar();
	vector<float> mejores_sol;

	if(rank == 0)
	{
		// printConfig();
		for(int hilo = 1; hilo < NUM_HILOS; hilo++)
		{
			// Enviando matriz de distancias
			MPI_Send(
				(int*)distancias, // ptr a los datos
				NUM_MAX_CIUDADES*NUM_MAX_CIUDADES, //cant de datos a enviar  T/4
				MPI_INT, //tipo de dato
				hilo, //rank del destino 
				2,
				MPI_COMM_WORLD
			);
		}
		for(int gen = 0; gen < NUM_GENERACIONES; gen++)
		{
			// printf("\n\nGeneracion %d\n", gen+1);
			for(int hilo = 1; hilo < NUM_HILOS; hilo++)
			{
				// Enviando parte de la generacion a cada hilo
				MPI_Send(
					(int*)(generacion+TAM_GENERACION/NUM_HILOS*hilo), // ptr a los datos
					TAM_GENERACION/NUM_HILOS*51, //cant de datos a enviar  T/4
					MPI_INT, //tipo de dato
					hilo, //rank del destino 
					7,
					MPI_COMM_WORLD
				);
				// printf("Datos Enviados a rank %d:\n", hilo);
				// for(int i = 0; i < TAM_GENERACION/NUM_HILOS; i++)
				// 	printGeneracion(generacion+TAM_GENERACION/NUM_HILOS*i);
			}
			// printf("Datos enviados a cada hilo.\n");
			evaluaGeneracion(generacion, TAM_GENERACION/NUM_HILOS);

			for(int hilo = 1; hilo < NUM_HILOS; hilo++)
			{
				MPI_Recv(
					(int*)(generacion+TAM_GENERACION/NUM_HILOS*hilo), // ptr a los datos
					TAM_GENERACION/NUM_HILOS*51, //cant de datos a recibir
					MPI_INT, //tipo de dato
					hilo,  //rank de quien envio los datos
					17, //etiqueta
					MPI_COMM_WORLD, //mundo
					&status //status de la comunicacion
				);
			}
			// printf("Recibio datos de los esclavos\n");
			// printGeneracion(generacion);
			qsort(generacion, TAM_GENERACION, sizeof(Solucion), compara);
			mejores_sol.push_back(generacion[0].costo_de_ruta);
			if(gen == NUM_GENERACIONES-1)
				break;
			cruzaGeneracion(generacion);
			mutarGeneracion(generacion);
		}
		// printf("Ultima Generacion:\n");
		// printGeneracion(generacion);
		endtime = MPI_Wtime();
		generaArchivoMejoresSol(mejores_sol);
		printf("Mejor Solucion:\n");
		printSolucion(generacion[0]);
		printf("Tiempo de ejecución total usando: %f seg.\n", endtime-starttime);
	}
	else
	{
		// Recibiendo matriz de distancias
		MPI_Recv(
			(int*)distancias, // ptr a los datos
			NUM_MAX_CIUDADES*NUM_MAX_CIUDADES, //cant de datos a enviar
			MPI_INT, //tipo de dato
			0,  //rank de quien envio los datos
			2, //etiqueta
			MPI_COMM_WORLD, //mundo
			&status //status de la comunicacion
		);
		// printConfig();
		for(int gen = 0; gen < NUM_GENERACIONES; gen++)
		{
			// printf("\n\nGeneracion %d\n", gen+1);
			// Recibiendo parte de la generacion
			MPI_Recv(
				(int*)generacion, // ptr a los datos
				TAM_GENERACION/NUM_HILOS*51, //cant de datos a enviar
				MPI_INT, //tipo de dato
				0,  //rank de quien envio los datos
				7, //etiqueta
				MPI_COMM_WORLD, //mundo
				&status //status de la comunicacion
			);
			// printf("Datos recibidos de maestro: \n");
			// printGeneracion(generacion);
			evaluaGeneracion(generacion, TAM_GENERACION/NUM_HILOS);
			// printf("Despues de Evaluarlos: \n");
			// printGeneracion(generacion);

			MPI_Send(
				(int*)generacion , // ptr a los datos
				TAM_GENERACION/NUM_HILOS*51, //cant de datos a enviar  T/4
				MPI_INT, //tipo de dato
				0, //rank del destino 
				17, 
				MPI_COMM_WORLD
			);
			// printf("Enviados los datos hacia master (%d)\n", rank);
		}
	}
	MPI_Finalize();
}

void generaArchivoMejoresSol(vector<float> mejores_sol)
{
	FILE *fp;
	fp = fopen(nombre_archivo_reporte, "w");

	int tam_mejores_sol = mejores_sol.size();
	for(int i = 0; i < tam_mejores_sol; i++)
		fprintf(fp, "Generacion(%d)=%f;\n", i+1, mejores_sol[i]);
	fprintf(fp, "figure,plot(1:%d,Generacion);\n", tam_mejores_sol);
	
	fclose(fp);
}


/**
 * Funciones del Algoritmo Genetico
 */

void mutarGeneracion(Solucion* generacion)
{
	int pos;
	for(int i = 0; i < (int)NUM_ELEM_A_MUTAR; i++)
	{
		mutarADN(generacion[rand()%TAM_GENERACION].ruta);
		NUM_ELEM_A_MUTAR *= 0.95;
	}
}

void mutarADN(int* ADN)
{
	int pos1, pos2;
	do {
		pos1 = rand()%NUM_CIUDADES;
		pos2 = rand()%NUM_CIUDADES;
	} while(pos1 == pos2);
	int aux = ADN[pos1];
	ADN[pos1] = ADN[pos2];
	ADN[pos2] = aux;
}

void cruzaGeneracion(Solucion* generacion)
{
	int mitad_gen = TAM_GENERACION/2;
	int pos_insert_nuevos_adns = 0;
	Solucion* nueva_generacion = (Solucion*)malloc(TAM_GENERACION*sizeof(Solucion));
	pair< int, int > posiciones_a_cruzar;
	for(int i = 0; i < mitad_gen; i++)
	{
		switch(ESTRATEGIA_DE_SELECCION)
		{
			case aleatoria:
				posiciones_a_cruzar = seleccionAleatoria(mitad_gen);
				break;
			case rank:
				posiciones_a_cruzar = seleccionRank();
				break;
			case torneo:
				posiciones_a_cruzar = seleccionTorneo();
				break;
		}
		cruzaADNs(generacion[posiciones_a_cruzar.first].ruta, generacion[posiciones_a_cruzar.second].ruta, nueva_generacion, pos_insert_nuevos_adns);
		pos_insert_nuevos_adns += 2;
	}
	for(int i = 0; i < TAM_GENERACION; i++)
		generacion[i] = nueva_generacion[i];
	// memcpy(generacion, nueva_generacion, sizeof(nueva_generacion));	//¿Porque no funciona con memcpy??
}

void cruzaADNs(int* ADN1, int* ADN2, Solucion* nueva_generacion, int pos_insert_nuevos_adns)
{
	int pos_corte = (rand()%(NUM_CIUDADES-1))+1;
	int ADN1_aux[NUM_MAX_CIUDADES];
	int ADN2_aux[NUM_MAX_CIUDADES];
	vector< int* > indicesADN1;
	vector< int* > indicesADN2;
	int* pos;

	for(int i = 0; i < pos_corte; i++)
		ADN1_aux[i] = ADN1[i];
	for(int i = 0; i < pos_corte; i++)
		ADN2_aux[i] = ADN2[i];

	// Cruzamos
	for(int i = pos_corte; i < NUM_CIUDADES; i++)
		ADN1_aux[i] = ADN2[i];
	for(int i = pos_corte; i < NUM_CIUDADES; i++)
		ADN2_aux[i] = ADN1[i];

	// Cambiamos las ciudades repetidas
	for(int i = 0; i < pos_corte; i++)
	{
		pos = find(ADN1_aux + pos_corte, ADN1_aux+NUM_CIUDADES, ADN1_aux[i]);
		if(pos != ADN1_aux+NUM_CIUDADES)
			indicesADN1.push_back(pos);
		pos = find(ADN2_aux + pos_corte, ADN2_aux+NUM_CIUDADES, ADN2_aux[i]);
		if(pos != ADN2_aux+NUM_CIUDADES)
			indicesADN2.push_back(pos);
	}

	int temp;
	for(int i = 0; i < indicesADN1.size(); i++)
	{
		temp = *(indicesADN1[i]);
		*(indicesADN1[i]) = *(indicesADN2[i]);
		*(indicesADN2[i]) = temp;
	}

	Solucion sol1;
	memcpy(sol1.ruta, ADN1_aux, NUM_MAX_CIUDADES*sizeof(int));
	Solucion sol2;
	memcpy(sol2.ruta, ADN2_aux, NUM_MAX_CIUDADES*sizeof(int));
	nueva_generacion[pos_insert_nuevos_adns] = sol1;
	nueva_generacion[pos_insert_nuevos_adns+1] = sol2;
}

pair< int, int > seleccionAleatoria(int limite_gen)
{
	pair< int, int > posiciones_a_cruzar;
	do {
		posiciones_a_cruzar.first = rand()%limite_gen;
		posiciones_a_cruzar.second = rand()%limite_gen;
	} while(posiciones_a_cruzar.first == posiciones_a_cruzar.second);
	return posiciones_a_cruzar;
}

pair< int, int > seleccionRank()
{
	int sumatoria_rank = sumatoria(TAM_GENERACION);
	vector< float > ini_rebanada;
	pair< int, int > posiciones_a_cruzar;
	float porcion;
	float ini = 0;

	for(int i = 0; i < TAM_GENERACION; i++)
	{
		porcion = (TAM_GENERACION - i)/sumatoria_rank;
		ini_rebanada.push_back(ini);
		ini += porcion;
	}

	do {
		posiciones_a_cruzar.first = getPosition(rand()%TAM_GENERACION, ini_rebanada, TAM_GENERACION);
		posiciones_a_cruzar.second = getPosition(rand()%TAM_GENERACION, ini_rebanada, TAM_GENERACION);
	} while(posiciones_a_cruzar.first == posiciones_a_cruzar.second);
	return posiciones_a_cruzar;
}

pair< int, int > seleccionTorneo()
{
	vector< int > grupo1;
	vector< int > grupo2;

	for(int i = 0; i < 3; i++)
		grupo1.push_back(rand()%TAM_GENERACION);
	for(int i = 0; i < 3; i++)
		grupo2.push_back(rand()%TAM_GENERACION);

	sort(grupo1.begin(), grupo1.end(), comparaIntMayorAMenor);
	sort(grupo2.begin(), grupo2.end(), comparaIntMayorAMenor);
	
	return getIndicesMaxDeGrupos(grupo1, grupo2);
}

pair< int, int > getIndicesMaxDeGrupos(vector< int > grupo1, vector< int > grupo2)
{
	if(grupo1[0] != grupo2[0])
		return make_pair(grupo1[0], grupo2[0]);
	else
	{
		if(grupo1[1] < grupo2[1])
			return make_pair(grupo1[0], grupo1[1]);
		return make_pair(grupo1[0], grupo2[1]);
	}
}

int getPosition(int t, vector< float > ini_rebanada, int TAM_GENERACION)
{
	for(int i = 0; i < TAM_GENERACION; i++)
	{
		if(ini_rebanada[i] > t)
			return i-1;
	}
	return TAM_GENERACION-1;
}

int sumatoria(int n)
{
	int sumatoria = 0;
	while(n > 0)
	{
		sumatoria += n;
		n--;
	}
	return sumatoria;
}

void evaluaGeneracion(Solucion* generacion, int tam_generacion)
{
	//#pragma omp parallel for
	for(int i = 0; i < tam_generacion; i++)
		generacion[i].costo_de_ruta = evaluaSolucion(generacion[i].ruta);
}


float evaluaSolucion(int* ruta)
{
	float costo_de_ruta = distancias[ruta[0]][ruta[NUM_CIUDADES-1]];
	if(costo_de_ruta == INFINITO)
			return INFINITO;
	for(int i = 1; i < NUM_CIUDADES; i++)
	{
		if(distancias[ruta[i-1]][ruta[i]] == INFINITO)
			return INFINITO;
		costo_de_ruta += distancias[ruta[i-1]][ruta[i]];
	}
	return costo_de_ruta;
}

Solucion* inicializar()
{
	Solucion* generacion = getMemoria();
	for(int i = 0; i < TAM_GENERACION; i++)
		memcpy(generacion[i].ruta, getRutaRandom(), NUM_MAX_CIUDADES*sizeof(int));

	return generacion;
}

Solucion* getMemoria()
{
	Solucion* generacion;
	generacion = (Solucion*)malloc(TAM_GENERACION*sizeof(Solucion));
	// for(int i = 0; i < TAM_GENERACION; i++)
	// 	generacion[i].ruta = (int*)malloc(NUM_CIUDADES*sizeof(int));
	return generacion;
}


int* getRutaRandom()
{
	int* ruta;
	vector< int > ciudades;

	ruta = (int*)malloc(NUM_MAX_CIUDADES*sizeof(int));
	for(int i = 0; i < NUM_CIUDADES; i++)
		ciudades.push_back(i);

	int num;
	int pos = 0;
	while(ciudades.size())
	{
		num = rand() % ciudades.size();
		ruta[pos++] = ciudades[num];
		ciudades.erase(ciudades.begin() + num);
	}

	return ruta;
}

int compara(const void *s1, const void *s2)
{
	Solucion* sol1 = (Solucion*)s1; 
	Solucion* sol2 = (Solucion*)s2;

	if( sol1->costo_de_ruta == sol2->costo_de_ruta)
		return 0;
	else if( sol1->costo_de_ruta > sol2->costo_de_ruta)
		return 1;

	return -1;
}

bool comparaIntMayorAMenor(int a, int b)
{
	if(a > b)
		return true;
	return false;
}

// /**
//  * Funciones para Imprimir
//  */

void printGeneracion(Solucion* generacion)
{
	int cant_elem_a_imprimir = TAM_GENERACION < 50 ? TAM_GENERACION : 50;
	for(int i = 0; i < cant_elem_a_imprimir; i++)
	{
		printf("%d ) ", i);
		printSolucion(generacion[i]);
	}
}

void printSolucion(Solucion sol)
{
	for(int i = 0; i < NUM_CIUDADES; i++)
		printf("%d\t", sol.ruta[i]+1);
	printf("%.3f\n", sol.costo_de_ruta);
}

void printConfig()
{
	printf("# de Ciudades: %d\n", NUM_CIUDADES);
	printf("Estrategia de Seleccion: ");
	if(ESTRATEGIA_DE_SELECCION == aleatoria)
		printf("aleatoria\n");
	else if(ESTRATEGIA_DE_SELECCION == rank)
		printf("rank\n");
	else if(ESTRATEGIA_DE_SELECCION == torneo)
		printf("torneo\n");
	printf("Archivo de Reporte: %s\n", nombre_archivo_reporte);
	printf("# de Elementos a Mutar: %f\n", NUM_ELEM_A_MUTAR);
	printf("Tam de Generacion: %d\n", TAM_GENERACION);
	printf("# de Generaciones: %d\n", NUM_GENERACIONES);
	printf("# de Procesadores: %d\n", NUM_HILOS);
	printf("Matriz de Distancias: \n");
	for(int i = 0; i < NUM_CIUDADES; i++)
	{
		for(int j = 0; j < NUM_CIUDADES; j++)
			printf("%.3f\t", distancias[i][j]);
		printf("\n");
	}
	printf("\n");
}

void readConfig()
{
	float distancia;
	for(int i = 0; i < NUM_CIUDADES; i++)
	{
		for(int j = 0; j < NUM_CIUDADES; j++)
		{
			scanf("%f", &distancia);
			if(distancia == -1)
				distancia = INFINITO;
			distancias[i][j] = distancia;
		}
	}
}