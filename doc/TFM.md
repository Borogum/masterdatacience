# Presentación

## Diapositiva 1 (Introducción)

Bienvenidos a la presentación del trabajo fin de master: "Detección de eventos anómalos en un entorno industrial mediante el uso de técnicas de Federated Learning" mi nombre es Darío Martín García Carretero.

A lo largo de esta presentación se va mostrar una de las muchas aplicaciones del uso del aprendizaje federado. En este caso se utilizará como herramienta para la detección de eventos anómalos en un entorno industrial. Sin más preámbulos comencemos con la presentación.


## Diapositiva 2 (Contexto y justificación)

Primero comencemos situando el contexto y justificando la necesidad del trabajo.

### Bloque 1

¿Qué entendemos por un evento anómalo?

Un evento anómalo es aquel que se produce de forma repentina y sin previsión. Estos eventos anómalos pueden ser de muy diversas índoles dependiendo del contexto. Por ejemplo, dentro de la medicina un evento anómalo podría ser una arritmia cardiaca detectada en un paciente o, el en contexto de la seguridad podría ser el sonido del disparo de una pistola que ha sido registrado en las grabaciones de una cámara de seguridad.

## Bloque 2

En un entorno industrial, los eventos anómalos también pueden ser de muy diversa índole, en nuestro trabajo consideraremos como evento anómalo los fallos que, irremediablemente se producen en las máquinas debido al desgaste derivado de su uso continuado.

## Bloque 3

Hay que tener en cuenta que estos fallos pueden provocar grandes daños económicos y personales. Por lo que su detección es algo muy importante que puede ayudar a prevenir situaciones irreversibles.

Después de explicar el contexto y la justificación pasemos ahora a exponer el caso concreto que pretendemos resolver.


## Diapositiva 3 (Escenario I)

Una empresa desea implantar un sistema que detecte las posibles averías de las máquinas de sus factorías antes de que estas puedan provocar daños tanto al personal como a las propias instalaciones.

### Bloque 1

La empresa dispone de plantas repartidas por todo el mundo. A pesar de pertenecer a la misma compañía cada una de las instalaciones tiene sus propias particularidades en cuanto al tipo de productos que fabrican, la manera de fabricarlos, etc. A estas diferencias se han de añadir también las condiciones ambientales de cada lugar: temperatura, humedad, presión atmosférica, etc. A pesar de estas diferencias, las máquinas utilizadas en todas las factorías son similares.


## Bloque 2

A pesar de pertenecer a la misma compañía, las plantas trabajan con un alto grado de independencia y de hecho, suelen competir entre ellas en cuestiones como la cantidad de producción, calidad, etc. Debido a esta competitividad, las factorías son reacias a compartir datos sobre sus técnicas de producción, las configuraciones de sus máquinas, etc. Esto implica que el acceso a sus datos está muy restringido y estos pueden ser únicamente utilizados a nivel interno.

## Bloque 3

La compañía está en constante expansión y es habitual que abra nuevas plantas a lo largo de mundo. Obviamente la empresa desea implantar el sistema de detección de fallos en estas nuevas fábricas lo más rápidamente posible.


## Diapositiva 4 (Escenario II)

### Bloque 1

Hoy en día la mayoría de los componentes dentro de un entorno industrial están monitorizados mediante el uso de dispositivos de medición especializados. Por lo tanto, disponemos de una gran cantidad de datos de los sensores ubicados en las factorías.

### Bloque 2

Como disponemos de esa gran cantidad de datos sobre el funcionamiento de la instalaciones (recordar que los datos solo se pueden usar a nivel interno, solo a nivel local) es totalmente lógico que se opte por el uso de técnicas de Machine Learning para resolver el problema.

### Bloque 3

A pesar de poder aplicar técnicas de Machine Learning tradicional para resolver el problema a nivel local, es importante tener en cuenta que es imprescindible tratar el problema a nivel global por el siguiente motivo, la necesidad de una  implantación rápida del sistema de detección de fallos en las nuevas instalaciones. Abordando el problema de forma tradicional sería necesario repetir el proceso de la creación de modelos que se siguió en el resto de plantas, lo cual podría llevar meses o incluso años. Veremos como nos ayudará el uso del Federated Learning a la hora de crear modelos de calidad y cumplir todas las restricciones de privacidad y rápido despliegue.

Pero antes de nada haremos una breve introducción tanto al Machine Learning como al Federated Learning.


## Diapositiva 5 (Machine Learning I)

### Bloque 1

¿Qué es el Machine Learning?

El Machine Learning (aprendizaje automático) es un subcampo de la computación y una rama de la inteligencia artificial. Su  objetivo es crear programas (llamados comúnmente modelos) capaces de generalizar comportamientos a través de la información suministrada en forma de ejemplos.

### Bloque 2

¿Qué puede hacer por nosotros?,  ¿Qué aplicaciones tiene?

Hoy en día el Machine Learning tiene una gran variedad de aplicaciones, entre las que se incluyen: motores de búsqueda, diagnóstico médico, detección de fraude en el uso de tarjetas de crédito, clasificación de secuencias de ADN, videojuegos, etc.


## Diapositiva 6 (Machine Learning II)

Bien, ¿Y como funciona?

Para explicarlo de una manera sencilla estableceremos una analogía entre nuestro cerebro y un modelo de Machine Learning mediante un simple ejemplo:

### Bloque 1

Cuando somos niños y vemos por primera vez un pez no reconoceremos el objeto porque nunca hemos visto algo similar. Si nos explican de que se trata, la próxima vez que veamos un pez, lo reconoceremos inmediatamente. Esto sucede debido a que, de forma inconsciente, nuestro cerebro ha almacenado las características del animal: que tiene aletas, cola, escamas, etc.

### Bloque 2

El aprendizaje automático funciona, en la mayoría de los casos, de forma análoga. Al modelo (nuestro cerebro) se le suministra un conjunto de datos etiquetados (pez o no pez) y el modelo “aprende” a reconocer patrones en los datos. Posteriormente ese modelo, gracias a la generalización, será capaz de reconocer peces cuando se los encuentre.


## Diapositiva 7 (Machine Learning III)


¿Cómo se aplica el Machine Learning?
La construcción de un modelo de aprendizaje automático puede dividirse, a grandes rasgos, en tres etapas:

### Bloque 1

Etapa 1, adquisición y preparación de los datos. Como ya se menciono anteriormente, los modelos aprenden mediante el uso de ejemplos etiquetados. Es en esta etapa donde se obtienen estos ejemplos. Las fuentes pueden ser variadas y dependen del escenario en el que estemos trabajando. Por ejemplo, en nuestro caso, los datos provendrán de dispositivos instalados con el objetivo de monitorizar el comportamiento de las máquinas y de los registros de mantenimiento de dichas máquinas. Esta parte se correspondería con la parte de adquisición. En muchos casos, los datos no están en un formato adecuado y es por eso que tienen que ser procesados para darles la estructura adecuada para poder entrenar a los modelos. Esta sería la parte de la preparación.

### Bloque 2

Etapa 2, entrenamiento del modelo. Esta es la etapa donde el modelo "aprende" gracias a los datos adquiridos y preparados en la etapa anterior.

### Bloque 3

Etapa 3, validación del modelo. Esta es la etapa donde se "examina" el modelo construido. Es decir, donde se valora si ha aprendido lo suficiente. Aunque existen multitud de técnicas, la forma más habitual es la de suministrarle al modelo un conjunto de datos que no hayan sido utilizados en el entrenamiento y ver que sus resultados en función de una o varias métricas.

### Bloque 4

Es importante destacar que aunque estas fases se han presentado de manera secuencial no necesariamente se aplican de esa manera y es frecuente que se salte de una a otra dependiendo de las características del problema tratado.


## Diapositiva 8 (Federated Learning)






## Diapositiva 21 (Gracias)

Y aquí finaliza la presentación. Gracias por su atención.
