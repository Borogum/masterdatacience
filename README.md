# Código fuente TFM

El código fuente esta divido en cuatro partes:

##  Generación (generation)

Esta parte es la encargada de simular los datos de las instalaciones. Dentro de este paquete podemos encontrar cuatro módulos:

* simulation.py
* notify.py
* generate_data.py
* generate_all_data.py

### simulation.py

Contiene las clases necesarias para poder realizar las simulaciones de las instalaciones.

* __BrokenMachine:__  Excepción que será lanzada cuando una máquina llegue a un punto de ruptura.
* __Clock:__ Reloj, cada "tick" será un segundo.
* __HealthIndex:__ Permite simular el desgaste de un componente.
* __Machine:__ Simula el funcionamiento de una máquina.
* __Facility:__ Representa una instalación.


### notify.py

Este modulo contiene un conjunto de clases que permiten la comunicación (telemetría) de una máquina con otros dispositivos (archivos, consola, etc.). Se han implementado tres adaptadores:

* __ConsoleNotifier:__ Muestra los datos por consola.
* __ParquetNotifier:__ Almacena los datos en el formato Apache Parquet (https://parquet.apache.org/).
* __CsvNotifier:__ Almacena los datos en formato csv (adaptador usado por defecto).

### generate_data.py

Este módulo realiza una simulación cuyas características se encuentran establecidas en un archivo de configuración. La forma de invocarlo es la siguiente:

```
python generate_data.py plant.cfg
```

Un ejemplo de archivo de configuración podría ser el siguiente:

```
[CONFIGURATION]

name = Plant
; aprox. three months 3*30*24*3600
simulation_time = 7776000
; ambient temperature
temperature = 10
; ambient pressure
pressure = 98
; cycle characteristics
cycle_length_min = 1
cycle_length_max = 3
cycle_duration = 120
; working machines per period
machines_per_batch = 5
; machine speed limits
operational_speed_min = 1000
operational_speed_max = 1100
; period time
batch_time = 3600
; total number of machines
machine_count = 15
; times to failure
ttf1_min = 7000
ttf1_max = 40000
ttf2_min = 1000
ttf2_max = 50000
; max working temperature
temperature_max = 100
; relation between pressure and speed
pressure_factor = 1.5
; output paths
telemetry_path = plant/telemetry
event_path =  plant/event
```

Con la configuración anterior se almacenarían los datos de telemetría (por máquina) en la carpeta 'plant/telemetry' y los datos de mantenimiento (también por máquina) en la carpeta 'plant/event'.

### generate_all_data.py

Este script de conveniencia admite como parámetro de entrada una carpeta. El código escanea esta carpeta en busca de archivos de configuración. Para cada archivo encontrado ejecutará las simulaciones de forma paralela de acuerdo con la configuración.

## Procesado (processing)

Para el procesado se utiliza un único script 'process.py'. Este script procesará los datos obtenidos de la etapa de generación. Las trasformaciones que realizará serán las siguientes:

* Unificar los datos de mantenimiento y telemetría en un único archivo (por máquina).
* Agregar los datos a nivel de ciclo.
* Etiquetar las instancias.
* Calcular las agregaciones temporales.

La forma de invocar este script es la siguiente:

```
python process.py process.cfg

```

El archivo de configuración deberá tener la siguiente estructura:

```
[CONFIGURATION]

; minimum distance between consecutive cycles
gap = 30
; cycles before fail
w = 7
; number of cycles for rolling stats
rolling = 5
; path to load data
path_in = data/generation/
; ath to store processed data
path_out = data/processing/

```

## Preparación (preparation)

Como en el caso anterior se ha utilizado un único script que realizará toda la tarea de preparación de los datos para el proceso de entrenamiento y validación de los modelos de Machine Learning. Mas concretamente las tareas que realizará serán las siguientes:

* Unión de los datos de todas las máquinas en un único archivo.
* División del dataset en dos conjuntos: uno será utilizado para el entrenamiento y otro para la validación.
* Aplicación del método SMOTE para el balanceo de las clases.
* Normalización de los datos.

La forma de invocarlo será la siguiente:

```
python prepare.py preparation.cfg
```

El archivo de configuración tendrá la siguiente estructura:

```
[CONFIGURATION]

; test size percentage
test_size = 0.2
; Look back it must be equal to rolling value in processing stage
lookback = 5
; Proportion of minor classes (for each class)
p = 0.1
; folders
path_in = data/processing/
path_out = data/preparation/

```

## Creación modelo base (single_model)

Este paquete contiene los módulos necesarios para la creación del modelo que, en una fase posterior, se utilizará como modelo base para la construcción del modelo federado. Este paquete esta compuesto por los siguientes módulos:

* datasets.py
* model.py
* utils.py
* workers.py
* start_worker.py
* train_and_validate.py


### datasets.py

Este módulo contiene una única clase 'MachineDataset' que hereda de la clase Dataset (pyTorch). Esta clase facilitará la carga de datos ya que permite abstraernos de la estructura subyacente del dataset.

### model.py

Este modulo contiene dos objetos:

* La clase que describe al modelo de clasificación (Classifier).
* La función de coste que se utilizará para el ajuste del modelo (loss_fn).

## utils.py

LA finalidad de este módulo es proporcionar funciones que sin tener un objetivo específico, serán utilizados de manera transversal por varios paquetes. El módulo contiene dos funciones:

* cm2pred: Esta función transforma una matriz de confusión en dos vectores. Uno de ellos contendrá los etiquetas que suponen son ciertas (y_true) y el otro las etiquetas predichas por el modelo (y_pred).

* show_results: Dado un histórico de matrices de confusión y costes (train y test) crea una representación gráfica de estos. Además calcula una series de estadísticas como: precisión, F1-score, recall, etc.

## workers.py

Dado que la librería no proporcionaba ciertas estadísiticas que se consideraban de interés fué necesario
