# Código fuente TFM

El código fuente esta divido en cinco partes:

* Generación
* Procesado
* Preparado
* Creación del modelo base
* Creación del modelo federado

En las siguientes secciones se describirá detalladamente cada una de las partes.


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

Este módulo contiene un conjunto de clases que permiten la comunicación (telemetría) de una máquina con otros dispositivos (archivos, consola, etc.). Se han implementado tres clases:

* __ConsoleNotifier:__ Muestra los datos por consola.
* __ParquetNotifier:__ Almacena los datos en el formato Apache Parquet (https://parquet.apache.org/).
* __CsvNotifier:__ Almacena los datos en formato csv (clase usada por defecto).

### generate_data.py

Este script realiza una simulación y almacena los resultados (telemetría y mantenimiento) en archivos csv. Las características de la instalación a simular deben ser establecidas en un archivo de configuración. La forma de invocar este script es la siguiente:

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

Con la configuración anterior, se almacenarían los datos de telemetría (por máquina) en la carpeta "plant/telemetry/" y los datos de mantenimiento (también por máquina) en la carpeta "plant/event/".

### generate_all_data.py

Este script de conveniencia admite como parámetro de entrada una carpeta. El código escanea esta carpeta en busca de archivos de configuración. Para cada archivo encontrado se ejecutará la simulación correspondiente. Las simulaciones se realizarán de forma paralelizada lo que acelerará el proceso de generación de datos.

## Procesado (processing)

Para el procesado se utiliza un único script: "process.py". Este script procesará los datos obtenidos de la etapa de generación. Las trasformaciones que realizará sobre los datos serán las siguientes:

* Unificar los datos de mantenimiento y telemetría en un único archivo (por máquina).
* Agregar los datos a nivel de ciclo.
* Etiquetar las instancias.
* Calcular agregaciones temporales.

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

Como en el caso anterior, se ha utilizado un único script que realizará toda la tarea de preparación de los datos: "prepare.py". Esta última fase en la transformación de los datos estará centrada en construir un dataset cuyo objetivo es el proceso de entrenamiento y validación de los modelos de Machine Learning. Concretamente las tareas que realizará serán las siguientes:

* Unión de los datos de todas las máquinas en un único archivo.
* División del dataset en dos conjuntos: uno será utilizado para el entrenamiento y otro para la validación.
* Aplicación del método SMOTE para el balanceo de las clases.
* Normalización de los datos.

La forma de invocarlo será la siguiente:

```
python prepare.py preparation.cfg
```

El archivo de configuración deberá tener la siguiente estructura:

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

## Creación del modelo base (single_model)

Este paquete contiene los módulos necesarios para la creación del modelo que, en una fase posterior, se utilizará como modelo base para la construcción del modelo federado. Este paquete está compuesto por los siguientes módulos:

* datasets.py
* model.py
* utils.py
* workers.py
* start_worker.py
* train_and_validate.py


### datasets.py

Este módulo contiene una única clase llamada __MachineDataset__ que hereda de la clase __Dataset__ (pyTorch). Esta clase facilitará la carga de los datos ya que permitirá la abstracción de la estructura subyacente de los datos.

### model.py

Este módulo contiene dos objetos:

* La clase que describe al modelo de clasificación denominada __Classifier__.
* La función de coste que se utilizará para el ajuste del modelo cuyo nombre será __loss_fn__.

### utils.py

La finalidad de este módulo es proporcionar funciones que, sin tener un objetivo específico, serán utilizados de manera transversal por varios paquetes. El módulo contiene dos funciones:

* __cm2pred__ - Esta función transforma una matriz de confusión en dos vectores. Uno de ellos contendrá las etiquetas que se suponen ciertas (y_true) y el otro las etiquetas predichas por el modelo (y_pred).

* __show_results__ - Dado un histórico de matrices de confusión y costes (train y test) crea una representación gráfica de estas. Además, calcula una serie de estadísticas como: precisión, F1-score, recall, etc.

### workers.py

Dado que el framework utilizado (pySyft) no proporciona ciertas estadísticas que se consideraban de interés a la hora de evaluar los modelos fue necesario añadirlas. Para ello se crearon dos clases __CustomWebsocketClientWorker__ y __CustomWebsocketServerWorker__ que heredan de __WebsocketClientWorker__ y __WebsocketServerWorker__ respectivamente. Estas clases además de todas las estadísticas que proporcionaban las clases originales proporcionan también como método de evaluación la matriz de confusión.

### start_worker.py

El objetivo de este script es poner en marcha un __CustomWebsocketServerWorker__ con los parámetros especificados en línea de comandos. Su sintaxis sería el siguiente:

```
python start_worker.py id host port train_data test_data --verbose
```
Donde:

* id: es el nombre del servidor
* host: ip del servidor
* port: puerto por el que escuchara el servidor
* train_data: archivo que contiene los datos de entrenamiento
* test_data: archivo que contiene los datos de test
* --verbose: Es un parámetro adicional que controla los mensajes que se muestran por consola

Un ejemplo de uso podría ser el siguiente:

```
python start_worker.py  server 127.0.0.1 8777 "data/train.csv" "data/test.csv"
```

### train_and_validate.py

Este script entrena y valida el modelo implementado en __model.py__. Los parámetros de entrenamiento y del worker encargado de realizar la tarea son especificados en un archivo de configuración. A continuación, se muestra un ejemplo de uso:

```
python train_and_validate.py configuration.cfg
```

Un archivo de configuración podría ser el siguiente:

```
[CONFIGURATION]

;Worker config
worker_id = Pilot
host = 127.0.0.1
port = 8777
verbose = 0
; Train config
epochs = 35
batch = 32
optimizer = Adam
lr = 0.002
shuffle = 1
; Data config
train = data/train.csv
test = data/test.csv
```

## Creación del modelo federado (federated_model)

Para la creación del modelo federado se han diseñado dos scripts:

* start_workers.py
* train_and_validate.py

### start_workers.py

Este es el script encargado de iniciar los workers que participarán en la construcción y en la validación del modelo federado. Su invocación se realizaría de la siguiente manera:

```
python start_workers workers.cfg

```

Donde el archivo de configuración debería tener una estructura similar a la siguiente:

```
[WORKER 0]
id = Pilot
host = 127.0.0.1
port = 8800
verbose = 0
train = ../data/preparation/pilot/train.csv
test = ../data/preparation/pilot/test.csv

[WORKER 1]
id = A
host = 127.0.0.1
port = 8801
verbose = 0
train = ../data/preparation/A/train.csv
test = ../data/preparation/A/test.csv

[WORKER 2]
id = B
host = 127.0.0.1
port = 8802
verbose = 0
train = ../data/preparation/B/train.csv
test = ../data/preparation/B/test.csv

[WORKER 3]
id = N
host = 127.0.0.1
port = 8803
verbose = 0
train = ../data/preparation/N/train.csv
test = ../data/preparation/N/test.csv
```

### train_and_validate.py

Este script funciona de forma análoga al descrito en la sección anterior. Su ejecución se realizaría de la siguiente manera:

```
python train_and_validate configuration.cfg
```

Donde el archivo de configuración aunque guarda ciertas similitudes con el descrito anteriormente tiene una estructura propia. Un ejemplo de configuración podría ser la siguiente:

```
[TRAIN]
rounds = 15
epochs = 35
federate_after = 5
batch = 32
optimizer = Adam
lr = 0.002
shuffle = 1

[WORKER 0]
id = Pilot
host = 127.0.0.1
port = 8800
verbose = 0
federation_participant = 1

[WORKER 1]
id = A
host = 127.0.0.1
port = 8801
verbose = 0
federation_participant = 1

[WORKER 2]
id = B
host = 127.0.0.1
port = 8802
verbose = 0
federation_participant = 1

[WORKER 3]
id = N
host = 127.0.0.1
port = 8803
verbose = 0
federation_participant = 0

```

Es importante destacar que el atributo __federation_participant__ de las secciones referidas a los workers permite controlar la participación del worker en la construcción de modelo y en su validación. El valor 1 indica que ese worker participará en la creación del modelo y en su validación mientras que el valor 0 indicará que únicamente participará en la fase de validación.
