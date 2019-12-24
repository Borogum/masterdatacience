# Código fuente TFM

El código fuente esta divido en cuatro partes:

##  Generación

Esta parte es la encargada de generar los datos simulados. Dentro de este paquete podemos encontrar cuatro módulos:

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

Este módulo realiza una simulación cuyas caracterísiticas se encuentran establecidas en un archivo de configuración. La forma de invocarlo es la siguiente:

```
python generate_data.py plant.cfg
```

Un ejemplo de archivo de configuración podría ser el siguiente:

```
[CONFIGURATION]

name = Plant
; aprox. three months 3*30*24*3600
simulation_time = 7776000
temperature = 10
pressure = 98
cycle_length_min = 1
cycle_length_max = 3
cycle_duration = 120
machines_per_batch = 5
operational_speed_min = 1000
operational_speed_max = 1100
batch_time = 3600
machine_count = 15
ttf1_min = 7000
ttf1_max = 40000
ttf2_min = 1000
ttf2_max = 50000
temperature_max = 100
pressure_factor = 1.5
telemetry_path = plant/telemetry
event_path =  plant/event
```

Con la configuración anterior se almacenarían los datos de telemetría (por máquina) en la carpeta 'plant/telemetry' y los datos de mantenimiento (también por máquina) en la carpeta 'plant/event'.

### generate_all_data.py

Este script de conveniencia admite como parámetro de entrada un carpeta. El código escanea esta carpeta en busca de archivos de configuración. Para cada archivo encontrado ejecutará las simulaciones de forma paralela de acuerdo con la configuración.
