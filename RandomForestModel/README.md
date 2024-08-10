# Proyecto de detección y traducción de señas LESCO al español

## Opcion #1 - Random Forest

## Explicación

Esta opción 1 del proyecto fue definida inicialmente como una prueba de las capacidades de las librerías. Utiliza OpenCV para la captura de las imágenes, y un RandomForestClassifier de Scikit-learn para el entrenamiento del modelo.

El modelo cuenta con la limitacion de que solamente permite la captura y deteccion de señas estaticas, por lo que señas con movimiento como la letra J o Letra Z no pueden ser analizadas con el modelo actual.

## Uso

El proyecto incluye paso por paso los diferentes archivos Python que deben ser ejecutados. La version actual del proyecto ya incluye una version del modelo entrenado, por lo que se puede ejecutar el archivo "S.4 - Use_model.py" para probarlo.

El proyecto no cuenta con las imagenes originales por un tema de tamaño de almacenamiento, por lo que, en caso de necesitar entrar de nuevo el modelo, tambien se debe pasar por el proceso de capturar las imagenes.

