#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <time.h>
#include "../common/book.h"

//Elabora un n�mero aleatorio
__global__ void make_rand(int seed, char* m, int size) {
    float myrandf;
    int num;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state); //Se prepara la ejecuci�n del random de CUDA
    myrandf = curand_uniform(&state);
    myrandf *= (size - 0 + 0.999999);
    num = myrandf;
    if (m[num] == 'O')
    {
        m[num] = 'X';
    }
}
//Se da el valor inicial de las distintas casillas de la matriz
__global__ void prepare_matrix(char* p, int number_columns, int number_rows, int width_block, int situation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valAux = 0;
    if (situation == 0) // Caso en el que el n�mero de filas y el n�mero de columnas son m�ltiplos del width de los bloques creados
    {
        p[idx] = 'O';
    }
    else if ((situation == 1) && (idx % ((number_columns / width_block + 1) * width_block) < number_columns)) 
        // Caso en el que el n�mero de filas es m�ltiplo del width de los bloques creados
        // y que el id del hilo se encuentra en una columna interior al bloque que no es completado
    {
        // Dado que el idx no tiene porque coincidir con la posici�n de la matriz donde se debe colocar el valor se debe modificar el valor de acuerdo al n�mero de columnas
        // Se obtiene el n�mero de bloques en exceso para representar todas las columnas
        valAux = idx / ((number_columns / width_block + 1) * width_block); 
        // Se realiza el producto de lo ya se ha obtenido con la diferencia entre el n�mero de columnas de n�mero de bloques en exceso y el n�merro de columnas que realmente tiene la matriz 
        valAux *= (((number_columns / width_block + 1) * width_block) - number_columns);
        // Se coloca el valor en la posici�n que se obtiene de diferencia entre el id del hilo y el valor previamente obtenido
        p[idx -  valAux] = 'O';
    }
    else if ((situation == 2) && (idx < number_rows*number_columns))
        // Caso en el que el n�mero de columnas es m�ltiplo del width de los bloques creados
        // y que el id es inferior al tama�o de la matriz a partir del n�mero de elementos en cada fila y en cada columna 
    {
        p[idx] = 'O';
    }
    else if ((situation == 3) && (idx % ((number_columns / width_block + 1) * width_block) < number_columns) && (idx - (idx / ((number_columns / width_block + 1) * width_block)) * (((number_columns / width_block + 1) * width_block) - number_columns) < number_rows * number_columns))
        // Caso en el que el n�mero de filas y el n�mero de columnas no son m�ltiplos del width de los bloques creados,
        // que el id del hilo se encuentra en una columna interior al bloque que no es completado 
        // y que el id es inferior reajustado a las dimensiones de la matriz dada es menor al tama�o de la matriz a partir del n�mero de elementos en cada fila y en cada columna
    {
        // Dado que el idx no tiene porque coincidir con la posici�n de la matriz donde se debe colocar el valor se debe modificar el valor de acuerdo al n�mero de columnas
        // Se obtiene el n�mero de bloques en exceso para representar todas las columnas
        valAux = idx / ((number_columns / width_block + 1) * width_block);
        // Se realiza el producto de lo ya se ha obtenido con la diferencia entre el n�mero de columnas de n�mero de bloques en exceso y el n�merro de columnas que realmente tiene la matriz 
        valAux *= (((number_columns / width_block + 1) * width_block) - number_columns);
        // Se coloca el valor en la posici�n que se obtiene de diferencia entre el id del hilo y el valor previamente obtenido
        p[idx - valAux] = 'O';
    }
}

//Se genera una matriz de manera que los elementos bajan una fila
__global__ void matrix_operation(char* m, char* p, int width, int size, int number_columns, int number_rows, int width_block, int situation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int counter = 0;
    int valAux = 0;
    bool verify = false;
    if (situation == 0)// Caso en el que el n�mero de filas y el n�mero de columnas son m�ltiplos del width de los bloques creados
    {
        verify = true;
    }
    else if ((situation == 1) && (idx % ((number_columns / width_block + 1) * width_block) < number_columns))
        // Caso en el que el n�mero de filas es m�ltiplo del width de los bloques creados
        // y que el id del hilo se encuentra en una columna interior al bloque que no es completado
    {
        verify = true;
    }
    else if ((situation == 2) && (idx < number_rows * number_columns))
        // Caso en el que el n�mero de columnas es m�ltiplo del width de los bloques creados
        // y que el id es inferior al tama�o de la matriz a partir del n�mero de elementos en cada fila y en cada columna 
    {
        verify = true;
    }
    else if ((situation == 3) && (idx % ((number_columns / width_block + 1) * width_block) < number_columns) && (idx - (idx / ((number_columns / width_block + 1) * width_block)) * (((number_columns / width_block + 1) * width_block) - number_columns) < number_rows * number_columns))
        // Caso en el que el n�mero de filas y el n�mero de columnas no son m�ltiplos del width de los bloques creados,
        // que el id del hilo se encuentra en una columna interior al bloque que no es completado 
        // y que el id es inferior reajustado a las dimensiones de la matriz dada es menor al tama�o de la matriz a partir del n�mero de elementos en cada fila y en cada columna
    {
        verify = true;
    }
    if (verify) //Solo realizar�n esta operaci�n aquellos hilos que hayan cumplido una de las condiciones anteriores
    {
        if ((situation == 1) || (situation == 3))
        {
            // Dado que el idx no tiene porque coincidir con la posici�n de la matriz donde se debe colocar el valor se debe modificar el valor de acuerdo al n�mero de columnas
            // Se obtiene el n�mero de bloques en exceso para representar todas las columnas
            valAux = idx / ((number_columns / width_block + 1) * width_block);
            // Se realiza el producto de lo ya se ha obtenido con la diferencia entre el n�mero de columnas de n�mero de bloques en exceso y el n�merro de columnas que realmente tiene la matriz 
            valAux *= (((number_columns / width_block + 1) * width_block) - number_columns);
            // Se coloca el valor en la posici�n que se obtiene de diferencia entre el id del hilo y el valor previamente obtenido
            valAux = idx - valAux;

        }
        else if ((situation == 0) || (situation == 2))
        {
            valAux = idx;
        }

        if ((valAux % width != 0) && (valAux - width >= 0) && (m[valAux - width - 1] == 'X')) // Estudia si existe esquina superior izquierda y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux % width != 0) && (m[valAux - 1] == 'X')) //Estudia si existe el casilla en el lateral izquierdo y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux - width >= 0) && (m[valAux - width] == 'X')) //Estudia si existe el casilla en el lateral superior y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux % width != width - 1) && (valAux - width >= 0) && (m[valAux - width + 1] == 'X')) // Estudia si existe esquina superior derecha y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux % width != width - 1) && (m[valAux + 1] == 'X')) //Estudia si existe el casilla en el lateral derecho y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux % width != 0) && (valAux + width < size) && (m[valAux + width - 1] == 'X')) // Estudia si existe esquina inferior izquierda y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux + width < size) && (m[valAux + width] == 'X')) //Estudia si existe el casilla en el lateral inferior y si tiene una c�lula viva
        {
            counter++;
        }
        if ((valAux % width != width - 1) && (valAux + width < size) && (m[valAux + width + 1] == 'X')) // Estudia si existe esquina inferior derecha y si tiene una c�lula viva
        {
            counter++;
        }

        if (situation == 1 || situation == 3)
        {
            // Dado que el idx no tiene porque coincidir con la posici�n de la matriz donde se debe colocar el valor se debe modificar el valor de acuerdo al n�mero de columnas
                // Se obtiene el n�mero de bloques en exceso para representar todas las columnas
            valAux = idx / ((number_columns / width_block + 1) * width_block);
            // Se realiza el producto de lo ya se ha obtenido con la diferencia entre el n�mero de columnas de n�mero de bloques en exceso y el n�merro de columnas que realmente tiene la matriz 
            valAux *= (((number_columns / width_block + 1) * width_block) - number_columns);
            // Se coloca el valor en la posici�n que se obtiene de diferencia entre el id del hilo y el valor previamente obtenido
            if ((counter == 3) && (m[idx - valAux] == 'O')) // Una c�lula muerte se convierte en viva si tiene 3 c�lulas vivas alrededor de ella
            {
                p[idx - valAux] = 'X';
            }
            else if (((counter < 2) || (counter > 3)) && (m[idx - valAux] == 'X')) // Una c�lula viva se convierte en muerte si alrededor de ella hay un n�mero de c�lulas distinto de 2 o 3
            {
                p[idx - valAux] = 'O';
            }
            else //La c�lula mantiene su estado
            {
                p[idx - valAux] = m[idx - valAux];
            }
        }
        else if((situation == 0) || (situation == 2)) // Situaciones 0 o 2
        {
            if ((counter == 3) && (m[idx] == 'O')) // Una c�lula muerte se convierte en viva si tiene 3 c�lulas vivas alrededor de ella
            {
                p[idx] = 'X';
            }
            else if (((counter < 2) || (counter > 3)) && (m[idx] == 'X')) // Una c�lula viva se convierte en muerte si alrededor de ella hay un n�mero de c�lulas distinto de 2 o 3
            {
                p[idx] = 'O';
            }
            else //La c�lula mantiene su estado
            {
                p[idx] = m[idx];
            }
        }
    }
}

void operation(int size, int width, int nBlocks, int nThreads, int number_columns, int number_rows, char execution_mode, int width_block, int situation);
void generate_matrix(char* m, int size, int nBlocks, int nThreads, int number_columns, int number_rows, int width_block, int situation);
int generate_random(int min, int max);
void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads, int number_columns, int number_rows, int width_block, int situation);
void show_info_gpu_card();
int main(int argc, char* argv[])
{
    show_info_gpu_card(); //Muestra las caracter�sticas de la tarjeta gr�fica
    printf("Comienza el juego de la vida:\n");
    int situation = 0;
    int number_blocks = 1;
    int number_threads = 1;
    int number_rows = 32;
    int number_columns = 32;
    int width_block = 1;
    char execution_mode = 'a';
    // Condiciones para los casos en los que se est� pasando por terminal una serie de par�metros
    if (argc == 2)
    {
        execution_mode = argv[1][0];
    }
    else if (argc == 3)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
    }
    else if (argc >= 4)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
        number_columns = atoi(argv[3]);
    }
    int size = number_rows * number_columns;
    int width = number_columns;
    if ((size <= 8*8)||(number_rows < 8) || (number_columns < 8)) 
        // Si el tama�o de la matriz es inferior o igual a 64 o el cualquiera, ya sea el n�mero de elementos por fila o por columna es inferior a 8
    {
        number_blocks = number_columns;
        number_threads = number_rows;
        //number_blocks = 1;
        //number_threads = size;
        operation(size, width, number_blocks, number_threads, number_columns, number_rows, execution_mode, width_block, situation);
    }
    else if ((size <= 16 * 16) || (number_rows < 16) || (number_columns < 16))
        // Si el tama�o de la matriz es inferior o igual a 256 o el cualquiera, ya sea el n�mero de elementos por fila o por columna es inferior a 16
    {
        width_block = 8;
        number_threads = width_block * width_block;
        if ((number_rows % width_block == 0) && (number_columns % width_block == 0)) // N�mero de elementos m�ltiplos de 8 tanto en fila como en columna
        {
            number_blocks = (number_rows / width_block) * (number_columns / width_block);
            situation = 0;
        }
        else if (number_rows % width_block == 0)// N�mero de elementos m�ltiplos de 8 en fila 

        {
            number_blocks = (number_rows / width_block) * ((number_columns / width_block) + 1);
            situation = 1;
        }
        else if (number_columns % width_block == 0)// N�mero de elementos m�ltiplos de 8 en columna
        {
            number_blocks = ((number_rows / width_block) + 1) * (number_columns / width_block);
            situation = 2;
        }
        else
        {
            number_blocks = ((number_rows / width_block) + 1) * ((number_columns / width_block) + 1);
            situation = 3;
        }
        operation(size, width, number_blocks, number_threads, number_columns, number_rows, execution_mode, width_block, situation);
    }
    else if (size <= 32 * 32)
        // Si el tama�o de la matriz es inferior o igual a 1024 
    {
        width_block = 16;
        number_threads = width_block * width_block;
        if ((number_rows % width_block == 0) && (number_columns % width_block == 0))// N�mero de elementos m�ltiplos de 16 tanto en fila como en columna
        {
            number_blocks = (number_rows / width_block) * (number_columns / width_block);
            situation = 0;
        }
        else if (number_rows % width_block == 0)// N�mero de elementos m�ltiplos de 16 en fila 
        {
            number_blocks = (number_rows / width_block) * ((number_columns / width_block) + 1);
            situation = 1;
        }
        else if (number_columns % width_block == 0)// N�mero de elementos m�ltiplos de 16 en columna
        {
            number_blocks = ((number_rows / width_block) + 1) * (number_columns / width_block);
            situation = 2;
        }
        else
        {
            number_blocks = ((number_rows / width_block) + 1) * ((number_columns / width_block) + 1);
            situation = 3;
        }
        operation(size, width, number_blocks, number_threads, number_columns, number_rows, execution_mode, width_block, situation);
    }
    else
    {
        printf("No son v�lidas las dimensiones introducidas para la matriz.\n");
    }


    getchar();
    getchar();
    return 0;
}

void operation(int size, int width, int nBlocks, int nThreads, int number_columns, int number_rows, char execution_mode, int width_block, int situation)
//Realiza todas las operaciones del juego de la vida
{
    int counter = 1;
    char* a = (char*)malloc(size * sizeof(char));
    char* b = (char*)malloc(size * sizeof(char));
    generate_matrix(a, size, nBlocks, nThreads, number_columns, number_rows, width_block, situation);
    printf("Situacion Inicial:\n");
    for (int i = 0; i < size; i++)//Representaci�n matriz inicial
    {
        if (i % width == width - 1)
        {
            printf("%c\n", a[i]);
        }
        else
        {
            printf("%c ", a[i]);
        }
    }
    while (true)
    {
        if (counter % 2 == 1)
        {
            step_life(a, b, width, size, nBlocks, nThreads, number_columns, number_rows, width_block, situation);
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representaci�n matriz inicial
            {
                if (i % width == width - 1)
                {
                    printf("%c\n", b[i]);
                }
                else
                {
                    printf("%c ", b[i]);
                }
            }
        }
        else
        {
            step_life(b, a, width, size, nBlocks, nThreads, number_columns, number_rows, width_block, situation);
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representaci�n matriz inicial
            {
                if (i % width == width - 1)
                {
                    printf("%c\n", a[i]);
                }
                else
                {
                    printf("%c ", a[i]);
                }
            }
        }
        counter++;
        if (execution_mode == 'm') //Si el modo seleccionado no es autom�tico para hasta que el usuario pulse una tecla
        {
            getchar();
        }
    }

    free(a);
    free(b);
}


void generate_matrix(char* m, int size, int nBlocks, int nThreads, int number_columns, int number_rows, int width_block, int situation)
// Genera la matriz con su estado inicial
{
    srand(time(NULL));
    int seed = rand() % 50000;
    char* m_d;
    int numElem = generate_random(1, size*0.25);// Genera un n�mero aleatorio de m�xima n�mero de c�lulas vivas en la etapa inicial siendo el m�ximo un 15% del m�ximo n�mero de casillas
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    prepare_matrix << <nBlocks, nThreads >> > (m_d, number_columns, number_rows, width_block, situation); //Prepara la matriz con todas las casillas con c�lulas muertas
    make_rand << <1, numElem >> > (seed, m_d, size); // Va colocando de forma aleatoria c�lulas vivas en las casillas de la matriz
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
}

void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads, int number_columns, int number_rows, int width_block, int situation)
// Genera la matriz resultado a partir de una matriz inicial con las restricciones marcadas para cada casilla
{
    char* m_d;
    char* p_d;
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMalloc((void**)&p_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, size * sizeof(char), cudaMemcpyHostToDevice);
    matrix_operation << <nBlocks, nThreads >> > (m_d, p_d, width, size, number_columns, number_rows, width_block, situation);// Estudia el cambio o no de valor de las distintas casillas de la matriz
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
    cudaFree(p_d);
}

int generate_random(int min, int max) // Genera un n�mero aleatorio entre un m�nimo y un m�ximo
{
    srand(time(NULL));
    int randNumber = rand() % (max - min) + min;
    return randNumber;
}

void show_info_gpu_card() // Muestra las caracter�sticas de la tarjeta gr�fica usada
{
    cudaDeviceProp prop;

    int count;
    //Obtenci�n n�mero de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Numero de dispositivos compatibles con CUDA: %d.\n", count);

    //Obtenci�n de caracter�sticas relativas a cada dispositivo
    for (int i = 0; i < count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("Informacion general del dispositivo %d compatible con CUDA:\n", i + 1);
        printf("Nombre GPU: %s.\n", prop.name);
        printf("Capacidad de computo: %d,%d.\n", prop.major, prop.minor);
        printf("Velocidad de reloj: %d kHz.\n", prop.clockRate);
        printf("Copia solapada dispositivo: ");
        if (prop.deviceOverlap)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }
        printf("Timeout de ejecucion del Kernel: ");
        if (prop.kernelExecTimeoutEnabled)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }

        printf("\nInformacion de memoria para el dispositivo %d:\n", i + 1);
        printf("Memoria global total: %zu GB.\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Memoria constante total: %zu Bytes.\n", prop.totalConstMem);
        printf("Memoria compartida por bloque: %zu Bytes.\n", prop.sharedMemPerBlock);
        printf("Ancho del bus de memoria global: &d.\n", prop.memoryBusWidth);
        printf("Numero registros compartidos por bloque: %d.\n", prop.regsPerBlock);
        printf("Numero hilos maximos por bloque: %d.\n", prop.maxThreadsPerBlock);
        printf("Memoria compartida por multiprocesador: %zu Bytes.\n", prop.sharedMemPerMultiprocessor);
        printf("Numero registros compartidos por multiprocesador: %d.\n", prop.regsPerMultiprocessor);
        printf("Numero hilos maximos por multiprocesador: %d.\n", prop.maxThreadsPerMultiProcessor);
        printf("Numero de hilos en warp: %d.\n", prop.warpSize);
        printf("Alineamiento maximo de memoria: %zu.\n", prop.memPitch);
        printf("Textura de alineamiento: %zd.\n", prop.textureAlignment);
        printf("Total de multiprocesadores: %d.\n", prop.multiProcessorCount);
        printf("Maximas dimensiones de un hilo: (%d, %d, %d).\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximas dimensiones de grid: (%d, %d, %d).\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    }
    getchar();
}