# Presentación

## Diapositiva 1 (Introducción)

Bienvenidos a la presentación del trabajo fin de master: "Detección de eventos anómalos en un entorno industrial mediante el uso de técnicas de Federated Learning" mi nombre es Darío Martín García Carretero. A lo largo de esta presentación se va mostrar una de las muchas aplicaciones del uso del aprendizaje federado. En este caso se utilizará como herramienta para la detección de eventos anómalos en un entorno industrial. Sin más preámbulos comencemos con la presentación.


## Diapositiva 2 (Contexto y justificación)

Empecemos situando el contexto y justificando la necesidad del trabajo.

### Bloque 1

¿Qué entendemos por un evento anómalo?

Un evento anómalo es aquel que se produce de forma repentina y sin previsión. Estos eventos anómalos pueden ser de muy diversas índoles dependiendo del contexto. Por ejemplo, dentro de la medicina un evento anómalo podría ser una arritmia cardiaca detectada en un paciente o, el en contexto de la seguridad podría ser el sonido del disparo de una pistola que ha sido registrado en las grabaciones de una cámara de seguridad.

## Bloque 2

En un entorno industrial, los eventos anómalos también pueden ser de muy diversa índole, en nuestro trabajo consideraremos como evento anómalo los fallos que, irremediablemente se producen en las máquinas debido al desgaste derivado de su uso continuado.

## Bloque 3

Hay que tener en cuenta que estos fallos pueden provocar grandes daños económicos y personales. Por lo que su detección es algo muy importante que puede ayudar a prevenir situaciones irreversibles.

## Diapositiva 3 (Escenario I)

Una vez explicado el contexto y la justificación pasemos ahora a exponer un caso concreto de aplicación.

Supongamos que una empresa desea implantar un sistema que detecte las posibles averías de las máquinas de sus factorías antes de que estas fallen y puedan provocar daños tanto al personal como a las propias instalaciones.

### Bloque 1

Esta empresa dispone de plantas repartidas por todo el mundo. A pesar de pertenecer a la misma compañía cada una de las instalaciones tiene sus propias particularidades en cuanto al tipo de productos que fabrican, la manera de fabricarlos, etc. A estas diferencias se han de añadir también las condiciones ambientales de cada lugar: temperatura, humedad, presión atmosférica, etc. A pesar de estas diferencias, las máquinas utilizadas en todas las factorías son similares.

## Bloque 2

Las plantas trabajan con un alto grado de independencia y de hecho, suelen competir entre ellas en cuestiones como la cantidad de producción, calidad, etc. Debido a esta competitividad, las factorías son reacias a compartir datos sobre sus técnicas de producción, las configuraciones de sus máquinas, etc. Esto implica que el acceso a sus datos está muy restringido y que estos únicamente pueden ser utilizados a nivel interno.

## Bloque 3

La compañía está en constante expansión y es habitual que abra nuevas plantas a lo largo de mundo. Obviamente la empresa desea implantar el sistema de detección de fallos en estas nuevas fábricas lo más rápidamente posible.


## Diapositiva 4 (Escenario II)

### Bloque 1

Hoy en día la mayoría de los componentes dentro de un entorno industrial están monitorizados mediante el uso de dispositivos de medición especializados esta empresa no es una excepción y por lo tanto, podemos disponer de una gran cantidad de datos de los sensores ubicados en las factorías.

### Bloque 2

Como disponemos de esa gran cantidad de datos sobre el funcionamiento de la instalaciones (recordar que los datos solo se pueden usar a nivel interno, es decir, de manera local) es totalmente lógico tratar de resolver el problema mediante el uso de técnicas de Machine Learning.

### Bloque 3

A pesar de poder aplicar técnicas de Machine Learning tradicional para resolver el problema a nivel local, es importante darse cuenta cuenta que es imprescindible tratar el problema a nivel global por el siguiente motivo, la necesidad de una implantación rápida del sistema de detección de fallos en las nuevas instalaciones. Abordando el problema de forma tradicional tendríamos únicamente dos alternativas:

* Repetir el proceso de la creación de modelos que se siguió en el resto de plantas. Esto podría llevar meses o incluso años, lo que va en contra del requerimiento de rápida implantación.

* Juntar los datos de todas las plantas en un único lugar y construir un modelo con todos los datos. El problema es que esta aproximación violaría el requerimiento de privacidad de los datos.

Por estos motivos se propone el uso del Federated Learning que permite crear modelos de gran calidad y cumplir con las restricciones de privacidad y rápido despliegue.

Hemos mencionado conceptos como Machine Learning, modelo, Federated Learnign pero no hemos explicado en que consisten. En las siguientes diapositivas realizaremos una breve introducción de estos conceptos.


## Diapositiva 5 (Machine Learning I)

### Bloque 1

¿Qué es el Machine Learning? El Machine Learning (en español, aprendizaje automático) es un subcampo de la computación y una rama de la inteligencia artificial. Su objetivo es crear programas (llamados comúnmente modelos) capaces de generalizar comportamientos a través de la información suministrada en forma de ejemplos (también llamados instancias).

### Bloque 2

¿Qué puede hacer por nosotros?, ¿Qué aplicaciones tiene? Hoy en día el Machine Learning tiene una gran variedad de aplicaciones, entre las que se incluyen: motores de búsqueda, diagnóstico médico, detección de fraude en el uso de tarjetas de crédito, clasificación de secuencias de ADN, videojuegos, etc.


## Diapositiva 6 (Machine Learning II)

Bien, ¿Y como funciona? Para explicarlo de una manera sencilla estableceremos una analogía entre nuestro cerebro y un modelo de Machine Learning mediante un simple ejemplo:

### Bloque 1

Cuando somos niños y vemos por primera vez un pez no reconoceremos el objeto porque nunca hemos visto algo similar. Si nos explican de que se trata, la próxima vez que veamos un pez, lo reconoceremos inmediatamente. Esto sucede debido a que, de forma inconsciente, nuestro cerebro ha almacenado las características del animal: que tiene aletas, cola, escamas, etc.

### Bloque 2

El aprendizaje automático funciona, en la mayoría de los casos, de forma análoga. Al modelo (nuestro cerebro) se le suministra un conjunto de datos etiquetados (pez o no pez) y el modelo “aprende” a reconocer patrones en los datos. Posteriormente ese modelo, gracias a la generalización, será capaz de reconocer peces cuando los vea.


## Diapositiva 7 (Machine Learning III)

¿Cómo se aplica el Machine Learning? La construcción de un modelo de aprendizaje automático puede dividirse, a grandes rasgos, en tres etapas:

### Bloque 1

Etapa 1, adquisición y preparación de los datos. Como ya se menciono, los modelos aprenden mediante el uso de ejemplos etiquetados. Es, en esta etapa, donde se obtienen estos ejemplos. Las fuentes pueden ser variadas y dependen del escenario en el que estemos trabajando. Por ejemplo, en nuestro caso, los datos provendrían de dispositivos ubicados en las factorías con el objetivo de monitorizar el comportamiento de las máquinas por un lado y, de los registros de mantenimiento de dichas máquinas por otro lado. Esta parte se correspondería con la parte de adquisición. En muchos casos, los datos capturados no están en un formato adecuado y tienen que ser procesados para darles la estructura adecuada para poder entrenar a los modelos. Esta sería la parte de que se encargaría la preparación de los datos.

### Bloque 2

Etapa 2, entrenamiento del modelo. Esta es la etapa donde el modelo "aprende" gracias a los datos adquiridos y preparados en la etapa anterior.

### Bloque 3

Etapa 3, validación del modelo. Esta es la etapa donde se "examina" el modelo construido. Es decir, donde se valora si ha aprendido lo suficiente. Aunque existen multitud de técnicas, la forma más habitual es la de suministrarle al modelo un conjunto de datos que no hayan sido utilizados en el entrenamiento y ver que sus resultados en función de una o varias métricas.

### Bloque 4

Es importante destacar que, aunque estas fases se han presentado de manera secuencial, no necesariamente se aplican de esa manera y es habitual que se salte de una a otra dependiendo de las características del problema tratado.


## Diapositiva 8 (Federated Learning)

Veamos ahora en que consiste el Federated Learning (aprendizaje federado en español). El aprendizaje federado es una técnica de aprendizaje automático que entrena un algoritmo a través de múltiples dispositivos o servidores sin intercambiar datos entre ellos. Este proceso se divide en 4 fases:

### Bloque 1

El servidor central elige el tipo de modelo a entrenar, en principio podrías ser cualquier tipo de modelo basado en la optimización de parámetros, en nuestro caso se usará una red neuronal.

### Bloque 2

El servidor central transmite el modelo al resto de participantes (nodos).

### Bloque 3

Los nodos entrenan el modelo de forma local con sus propios datos.

### Bloque 4

El servidor central solicita los modelos locales y a partir de ellos genera otro modelo si acceder a ningún dato.

Una ciclo completo se denomina ronda. Todo este proceso se repetirá hasta que se cumpla la condición de parada establecida, que puede estar basada en un criterio de calidad o en un número máximo de iteraciones.


## Diapositiva 9 (Diseño del experimento I)

Para mostrar que la aplicación del Federated Learning es una solución que puede ofrece unos buenos resultados compararemos sus resultados con los dos alternativas:

* La primera sería que cada factoría pudiese construir su propio modelo local. Sabemos que esta alternativa no es viable pero nos dará una cota superior de la calidad de los modelos que es posible construir.

* La segunda, basada en la intercambiabilidad de los modelos. Es decir, que un modelo entrenado en una planta pueda ser utilizado en otra sin falta de ser reentrenado. Notar que esta alternativa si que cumpliría con los requisitos establecidos de velocidad de implantación y privacidad.  

Para poner a prueba las tres alternativas ...

### Bloque 1

Se generarán cuatro conjuntos de datos simulados con diferentes condiciones ambientales y de funcionamiento. Estos datos contendrán tanto, de información de los sensores instalados en las máquinas como de los datos de los informes de mantenimiento.

### Bloque 2

De esos conjuntos se elegirá uno (que denominaremos Piloto) y con estos datos se construirá un modelo. Este modelo será el modelo base común para todos los demás conjuntos de datos. Notar que lo que nos interesa es la estructura del modelo y no el valor de sus parámetros.

### Bloque 3

Después se entrenará un modelo (con la estructura definida en el paso anterior) por cada uno de los conjuntos de datos y se evaluarán los modelos obtenidos. Esto nos dará unos la base de comparación ya que nos proporcionará una cota máxima del rendimiento que podríamos esperar.
Hasta aquí la parte que concierne al entrenamiento de modelos de modo local.


## Diapositiva 10 (Diseño del experimento II)

La parte del experimento que concierne a la aplicación del aprendizaje federado y consta de las siguientes fases ...

### Bloque 1

Se entrenará un modelo federado con tres de los cuatro conjuntos de datos. En esos tres se incluirá el conjunto de datos llamado Piloto. Los otros dos conjuntos pasarán a denominarse A y B.

### Bloque 2

Se compararán los resultados del modelo federado global con cada uno de los modelos locales de los conjuntos de datos que han participado en la federación. Es decir con los conjuntos Piloto, A y B.

### Bloque 3

Se evaluará el rendimiento del modelo federado con respecto al modelo local del conjunto de datos excluido de la federación (este conjunto se pasará a denominar N (letra inicial de New))

Una vez examinadas todos los resultados no será posible evaluar la idoneidad del uso del aprendizaje federado frente a su alternativa, la intercambiabilidad de los modelos.


## Diapositiva 11 (Tecnologías)

Para la implementación de todas las herramientas y  modelos desarrollados para este trabajo ha sido necesario el uso de multitud de tecnologías. Todas ellas tienen como nexo de unión el lenguaje de programación Python.

### Bloque 1

¿Por qué usar Python? Se ha elegido este lenguaje por diversos motivos:

* Simplicidad. Python ofrece la posibilidad de desarrollar programas muy potentes con muy pocas de líneas de código. En general, resulta un lenguaje fácil de usar y no se requiere mucho tiempo de codificación.

* Compatibilidad. Muchas de las tecnologías actuales relacionadas con el Machine Learning están pensadas para ser utilizadas con este lenguaje.

* Facilidad de aprendizaje. En comparación con otros lenguajes, Python es fácil de aprender incluso para los programadores con menos experiencia.

### Bloque 2

Todos los scripts han sido desarrollados en el Entorno de Desarrollo Integrado "PyCharm". Este entorno, específicamente diseñado para Python ofrece herramientas muy útiles como por ejemplo, análisis de código fuente y control de versiones.

### Bloque 3

Pandas, Pandas es una librería de para el lenguaje Python que permite la manipulación de datos y su análisis de una manera sencilla y eficiente.


### Bloque 4

Matplotlib, esta potente librería también para el lenguaje de programación Python permite realizar visualizaciones de gran calidad de una forma muy sencilla.

### Bloque 5

scikit-learn, es una librería especializada para el uso de Machine Learning para Python. Contiene multitud de implementaciones de diferentes tipos de modelos. Aunque no ha sido la librería que se ha utilizado para la creación de los modelos, si que se han utilizado para la evaluación de los modelos construidos con ...

### Bloque 6

... PyTorch, librería desarrollada principalmente por Facebook, de código abierto y que permite desarrolla modelos de redes neuronales profundas de una manera rápida y eficiente.

### Bloque 7

Y por último pero no menos importante PySyft. PySyft es una biblioteca de Python para el aprendizaje profundo seguro y privado. PySyft permite desacoplar los datos privados del entrenamiento del modelo, utilizando el aprendizaje federado. Esta librería es compatible con múltiples frameworks entre los que se incluyen TensorFlow y PyTorch.


## Diapositiva 12 (Simulación de un entorno industrial)

Para el entrenamiento de los modelos necesitamos datos. Sin embargo, es prácticamente imposible obtener un conjunto de datos real por ser este tipo de datos muy sensible para las compañías. Por este motivo no se utilizarán datos reales y en su lugar, se desarrollará un software que nos permitirá la simulación de estos datos.

### Bloque 1

Debido a las características únicas de cada tipo de instalación no nos es posible construir un software que simule cada una de las diferentes máquinas que pudieran existir en un entorno industrial. Por lo tanto, consideraremos únicamente un tipo de máquina: máquinas rotatorias genéricas.

### Bloque 2

Basándonos en conjuntos de datos ya existentes (por ejemplo, Turbofan Engine Degradation Simulation Data Set) para cada máquina se estudiarán las siguientes variables operacionales :

•	Velocidad rotacional
•	Temperatura
•	Presión

Ademas como consideramos muy importante condiciones ambientales, la presion y la Temperatura


### Bloque 3

Esta incorparados el desgate que influye en la mediciones reaza



Adicionalmente a la simulación de se simularan también el desgaste

una vez que se llegan final de la vida util la maquina se estroperá y se ceraá un registro del mantenimiento, en nuestro caso una máquina puede

fallar por dos motivo defiertens  y asu vez se registrará qur tipo de error se produjo


## Diapositiva 13 (Construcción del Modelo Base)

Siguinedo la planificación descrita en la en el diseño del experimento se ha creado un modelo local explicaremos a conitnuación al creación del modelo

local



Recordemos nuevamente nuestro objetivo, El objetivo es poder decidir si en un determinado momento una máquina está en riesgo de rotura o no utilizando únicamente sus datos telemétricos. Dicho de otra manera, se debe clasificar esa máquina como potencialmente peligrosa o como segura. Teniendo en cuenta esto parece claro que será necesario el uso de un modelo de clasificación.


Del total de los cuatro  conjunto de datos de datos simulados vamo a seleccionar un para el diseño, notar que lo que nos importa ahora no son los parámetros de entrenamiento del modelo sino que solo resulta relevante su estructrura


### Bloque 1

Como ya se en la transparencia Ml: ¿Como funciona?


Se ha procedido a la preparación de los datos para la tarea de clasificación. Podemos dividir esta preparación en 4 fases:

* Agregación:

Muchas máquinas del mundo real funcionan en ciclos. Un ciclo puede considerarse como un período temporal que describe un estado de funcionamiento de una máquina. Por ejemplo, la operación de un motor en un avión puede describirse por los ciclos: motor en funcionamiento (avión en vuelo) o motor apagado (avión en tierra).

Las transmisiones de telemetría sin procesar, si bien pueden ser muy útiles para tareas como el monitoreo en tiempo real, pueden causar problemas a la hora de construir modelos de detección de fallos. Es frecuente que en entornos no controlados (como puede ser una fábrica) los datos no sean del todo precisos. Para mitigar los posibles errores en las mediciones (fallos puntuales en los equipos de medida, ruido, etc.) que pueden afectar a la calidad de los modelos construidos, suele ser una buena opción considerar datos agregados por ciclo

* Etiquetado:

Gracias a los datos de mantenimiento podemos saber cuando una máquina ha fallado, ese registro será etiquetado como fallo (la etiqueta depende del tipo de fallo recodermos que teniamos dos tipo de  fallo). Como es interesante detectar el fallo antes de que se produzca se etiquerant tambien como fallos n  registros anteriores al fallo real  (en nuestro trabajo hemos utilizado 7 )


* Enriquecimiento:


Una forma de añadir mayor cantidad de información al conjunto de datos disponible es lo que se conoce como feature engineering (ingeniería de características). La ingeniería de características consiste en la creación de nuevos atributos a partir de los ya existentes. En muchos casos este tipo de procedimientos mejora notablemente la calidad de los modelos obtenidos.

Aquí se añadirán cuatro nuevos atributos. Cada nuevo atributo se corresponderá con los datos promediados de cierto atributo de las n instancias  precedentes (nosotros hemos seleccionado como n = 5 ).

Este procedimiento es muy interesante ya que permite añadir una cierta cantidad información histórica a cada registro. El modelo podrá, no solo tener información instantánea si no que tendrá también información sobre la tendencia.



* Balanceo:

Por la propia naturaleza de los datos que estamos manejando es normal que exista una gran diferencia en número entre los casos donde no se detecta ninguna anomalía (clase 0) y los casos donde es posible que se produzca una avería (clase 1 o clase 2 ). Sin embargo, este tipo de conjuntos de datos suelen ser problemáticos a la hora de entrenar los modelos. Existen multitud de técnicas para balancear conjuntos de datos: oversampling, subsampling, métodos basados en pesos, etc. [17]

Se ha decidido hacer uso del algoritmo SMOTE (Synthetic Minority Over-sampling Technique) [18]. Este algoritmo genera nuevas instancias a partir de las clases minoritarias dejando intacta la clase mayoritaria. Las nuevas instancias no son copias de los casos existentes si no que se calculan como combinaciones lineales de los vecinos más cercanos de esa misma clase.

### Bloque 2

Existen multitud de familias de modelos que pueden ser usados para tareas de clasificación: árboles de decisión, máquinas de soporte de vectores, k-vecinos más cercanos, etc. Sin embargo nos hemos decidido por el uso de redes neuronales. Esta elección está motivada principalmente por dos cuestiones:


-	Las redes neuronales han demostrado tener un rendimiento excelente en multitud de problemas.

-	Aunque en teoría el uso del Federated Learning es aplicable a cualquier tipo de modelo cuyo entrenamiento se base en la optimización de parámetros [15] , es cierto que al ser una tecnología relativamente nueva la mayoría de los frameworks actuales solo permiten el uso de redes neurales en sus implementaciones de aprendizaje federado.


### Bloque 3

Es muy impontabte seleccionar la medida correcta para el problema correcto. Un clásico error es considerar la medida Accuracy para problemas desbalanceados donde, un numero cercano al 1 (valor máximo) no indica en ningun caso un buen modelo

## Diapositiva 14 (medidas de calidad del modelo)


Es siguiente paso ha sido evaluar la calidad del modelo.


 Se ha tenido en cuenta tres medidas: precision, recall, f1-score

### Bloque 1

Esta medida responde a la siguiente pregunta ¿Qué proporción de instancias clasificadas como “X” son realmente “X”? Se busca maximizar esta medida cuando queremos estar muy seguros de nuestra predicción.

En nuestro la maximación de esta medida no ayudaría a estar seguros de que un error se va aproducir  esto ahorraria  dinero a la empresa ya que solo se harian mantenimientos cuando fueran necesdarios, el problema con esto es que podríamos dejar pasar fallos de los que no estamos seguros pero que se p roduccen esto puede representar un peligro en la segurirdad de la instalación


### Bloque 2


El recall responde a la siguiente pregunta ¿De todas las instancias “X” que existen, qué proporción han sido clasificadas como “X”? Se busca maximizar esta medida cuando queremos capturar la mayor cantidad de clases “X” posible.


La maximación de este esta medida es equivalente a la maximización de la segridad ya que nos garantiza que todos los casos en los que se produce seran detectados sin embargo esto puede ocasionar falsos positivos lo que desembocaría en un aumento en los costes de mantenimiento

### Bloque 3

Aunque obviamente la seguridad siempre debe ser lo primero es interesante establecer un balance (comrpomiso) de las dos medidas para ello es muy  La medida f1-score que consiste en la media armonica de los dos anteriores


## Diapositiva 15 (Resultados del modelo base)

Ya disponemos de un modelo base construido a parater del conjunto de datos de la plnata pilito. Aplicaremos la estructura de este modelo en el resto de instalaciones (recoredad que esto no servira como cota máxima del rendimiento de las dos opciones  que barajamamos)  


Los datos de la planta “Piloto” son los que se han utilizado en la construcción del modelo base y como se ha comentado anteriormente se han obtenido unos valores altos tanto para el recall como para la precision lo que se ve reflejado en el f1-score.



Se puede observar que los resultados que arroja el modelo entrenado con los datos de la instalación “A” son prácticamente idénticos a los obtenidos en el entrenamiento de la planta “Piloto”. Es posible que si hubiéramos aumentado el número de epochs hubieran mejorado los resultados ya que si nos fijamos en la evolución de las curvas de coste, parece no haberse alcanzado el punto de estabilidad.


En este entrenamiento se observa un ligero aumento de la precision en detrimento del recall. Vemos que ambos se compensan al ya que el f1-score se mantiene prácticamente igual que en los casos anteriores.


De todos los casos este podría considerarse el peor de todos ya que el recall y la precision han disminuido de forma notable con respecto al resto de fábricas. Al igual que ocurría con la instalación “A”, los resultados probablemente hubieran mejorado al aumentar el número de epochs.


## Diapositiva 16 (Intercanbiabilidad del modelo base)


En la sección anterior se ha visto que se  puede esperar del modelo base si se entrenase de manera local. En este apartado se pretende responder a la siguiente pregunta: ¿Funciona bien un modelo entrenado en una planta en otra sin necesidad de reentrenarlo? Para resolver esta duda presentaremos a continuación una tabla comparativa del f1-score (media de todas las clases) sobre todas las posibles combinaciones train-test. La razón de elegir esta medida es por simplicidad, ya que con un único valor podemos hacernos una idea tanto de la precision como del recall.


Observando la tabla se puede ver que por norma general el modelo entrenado de forma específica (diagonal) supera ampliamente en rendimiento a los entrenados en otras instalaciones. Se debe destacar el caso de la instalación “N” en la que para algunos casos el score es mucho mejor que para su propio conjunto de entrenamiento. El motivo seguramente guarde relación con el hecho de que el número de epochs no fuera lo suficientemente alto. Aunque este hecho aislado fuera generalizado existiría otro problema, ¿cómo saber a priori que modelo de todos de los que se dispone es mejor para la planta objetivo?

Teniendo en cuenta los datos y reflexiones anteriores no parece posible traspasar un modelo de una instalación a otra sin que esto tenga una repercusión negativa en la calidad de las predicciones del modelo.


## Diapositiva 17 (Resultados del modelo federado)

Veamos los resultados obtenidos

Aunque los scores no son tan buenos como los del entrenamiento de forma local vemos que se mantienen en un rango aceptable.

No encontramos ante una situación muy parecida al caso anterior los scores bajan pero aun así siguen estando en valores aceptables.

En el caso de esta instalación los valores de recall que en el entrenamiento local ya eran relativamente bajos bajan aún más siendo sin duda los peores valores de todas las instalaciones.


Es importante notar que esta instalación no participo en la fase de entrenamiento y eso explicaría los valores tan bajos de f1-score y que la curva de la función de coste del conjunto test esté por encima de la del entrenamiento. Aun así son valores bastante buenos.

## Diapositiva 18 (Comparación entre aproximaciones)

Como se lógico ambas alternativas presentan resultados peores al entrenamiento local que sería el caso ideal. Por otro lado, se puede ver que el método de aprendizaje federado es siempre mejor que el peor de los casos cuando se utiliza otra instalación, incluso en algunos casos supera a la mejor de las opciones. Hay que tener en cuenta que aunque en el mejor de los casos de usar el modelo de otra instalación supera al aprendizaje federado, nos encontramos con el problema adicional de encontrar, a priori, cual de todas las instalaciones disponibles será la más adecuada.  Por lo tanto, parece claro que el uso del Federated Learning es de gran utilidad en situaciones como la descrita en este trabajo.


## Diapositiva 19 (Conclusiones)


Hay que recordar que el objetivo de proyecto es explorar el posible uso del Federated Learning para la detección de eventos anómalos dentro de un entorno industrial. Para ello se ha descrito un escenario (o caso de uso) que podría corresponderse con las necesidades de una compañía multinacional como podría ser una compañía siderúrgica, minera, un fabricante de productos químicos, etc. En este caso de uso se han expuesto las limitaciones existentes en cuanto a la distribución de datos entre las distintas instalaciones en el ámbito de una organización con una gran dispersión geográfica y se ha puesto de manifiesto la necesidad de una rápida implantación de modelos de Machine Learning en instalaciones de nueva creación.

Para la solucionar el problema presentado se han comparado dos soluciones:

-	Una basada en la intercambiabilidad de modelos
-	Una basada en el uso del Federated Learning

Se ha mostrado que el método basado en la intercambiabilidad entre modelos puede resultar de utilidad pero añade complejidad al problema. Es necesario crear un método para decidir la planta de origen del modelo ya que como se ha visto, no todos los modelos ofrecen los mismos resultados. Otro problema que habría que añadir a esta alternativa es la propiedad del modelo, una planta podría exigir a otra algún tipo de contrapartida por la cesión del modelo creado con sus datos.

El método basado en Federated Learning ha demostrado ser más eficaz por dos motivos:

-	Ofrece resultados similares a la solución óptima basada en el intercambio de modelos y siempre resultados mejores que la peor de las soluciones de intercambiabilidad.

-	Todos los participantes son responsables en la creación del modelo por lo que nadie es propietario exclusivo de este.


## Diapositiva 20 (Trabajo futuro)

Como ya se ha mencionado el objetivo del trabajo es mostrar una metodología por lo que el objetivo de los modelos aquí generados no se extiende más allá de su uso a nivel didáctico y su aplicación en entornos reales dependerá mucho del tipo de entorno y de las fuentes de datos disponibles. Si embargo, todo el procedimiento hasta llegar a su construcción puede resultar de gran interés en la resolución de problemas similares y justamente esto, el proceso, es lo que deberá considerarse como el producto final de este trabajo.

Debido a las particularidades de cada tipo de instalación puede ser complicado que los modelos construidos aquí puedan aplicarse directamente en entornos del mundo real. Posibles líneas de trabajo futuro podrían ser la aplicación de los métodos aquí descritos en entornos industriales reales y no simulados.


## Diapositiva 21 (Gracias)

Y aquí finaliza la presentación. Gracias por su atención.