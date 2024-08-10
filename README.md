# Proyecto de detección y traducción de señas LESCO al español

## Justificación

El presente proyecto fue creado como parte  del desarrollo de la investigación e implementación de un traductor de la Lengua de Señas Costarricense al español para optar por el título de Máster.

## Explicación

Como parte del desarrollo del proyecto se exploraron dos formas alternativas de abordar el modelo de machine learning.
* La primera corresponde a la creacion de un modelo utilizando el metodo de Random Forest.
* La segunda opcion fue realizada por medio de una red neuronal creada por medio del API de Keras en Tensorflow.

Ambos modelos utilzan la libreria OpenCV para la captura de las imagenes por medio de la camara web en el dispositivo de captura. Asi como la libreria Mediapipe para la deteccion y analisis de los puntos claves de las manos y dedos del sujeto.

Como referencia final, tambien se ofrece una version traducida de un proyecto creado por otro autor, el cual, debido a su forma de implementar el modelo y separar la señas estaticas y dinamicas en 2 partes, se toma como referencia para el proyecto. La version traducida al español del japones de repositorio puede ser encontrada en el siguiente enlace: https://github.com/AndresZU/reconocimiento-de-senas-con-mediapipe

Para más información, referirse al artículo publicado, el cual será agregado posteriormente a este repositorio para referencia.


