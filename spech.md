# Speech por Diapositiva – TP N°4: Eliminar manchas en videos
## 1) Portada

Speech:
Buenas, mi nombre es Tomás Amundarain y hoy voy a presentar el Trabajo Práctico N°4, cuyo objetivo principal es desarrollar un método para eliminar manchas fijas en videos, como las que aparecen cuando un lente está sucio.

## 2) Definición del problema

Speech:
El problema que queremos resolver es sencillo de describir, pero complejo de abordar.
Cuando aparece una mancha fija en el lente, esa mancha se mantiene en todas las imágenes del video, pero el fondo sí se mueve.
Entonces, a lo largo de los cuadros del video existe suficiente información para reconstruir lo que está detrás de la mancha.
La idea es aprovechar ese movimiento para estimar qué debería verse en esa zona, y así limpiar el video.

## 3) Armado del dataset

Speech:
Para trabajar con este problema necesitamos un dataset controlado.
Elegimos un dataset de Kaggle basado en dígitos MNIST en movimiento.
Es ideal porque los objetos se desplazan de forma constante y el fondo es simple, lo que facilita las primeras pruebas.
A partir de este dataset generamos nuestros propios videos artificiales con manchas superpuestas.

## 4) El conjunto de datos MNIST

Speech:
MNIST es un dataset muy conocido de dígitos manuscritos.
En este caso se usa en su versión animada, donde los dígitos se mueven dentro de un cuadro.
Esta simplicidad nos permite enfocarnos en que el modelo aprenda a quitar manchas sin preocuparnos por escenas reales más complejas.

## 5) Baseline

Speech:
El primer paso fue generar un baseline sencillo.
A partir de los videos de MNIST agregamos una única mancha uniforme.
Con esto armamos el dataset final, definimos el modelo y realizamos las pruebas iniciales.
Esto nos permitió tener una referencia clara para medir mejoras posteriores.

## 6) Generación del dataset (detallado)

Speech:
Las transformaciones que aplicamos fueron:

Generar clips de 6 segundos.

Agregar una única mancha fija que se mantiene en todos los frames.

Este preprocesamiento nos permitió simular exactamente el problema real: una mancha estática y un fondo que se mueve.

## 7) Modelo – U-Net + Temporal Smoothing

Speech:
El modelo principal es una U-Net, una arquitectura muy utilizada para segmentación y restauración de imágenes.
Para mejorar la estabilidad entre frames agregamos un componente de Temporal Smoothing, que ayuda a que la reconstrucción sea coherente a través del tiempo.
Así evitamos parpadeos y obtenemos una limpieza más consistente entre cuadros consecutivos.

## 8) Prueba del modelo

Speech:
Probamos el modelo con videos generados específicamente para validar la reconstrucción.
Vimos que la U-Net logra recuperar gran parte de la información detrás de la mancha, aunque todavía hay artefactos y pérdida de detalle.
Aun así, demuestra que el enfoque funciona y que es posible reconstruir zonas ocultas aprovechando el movimiento del video.

## 9) Mejoras

Speech:
A partir de los resultados planteamos distintas líneas de mejora:

Variar el tipo de mancha.

Cambiar la intensidad.

Probar manchas irregulares.

Usar múltiples manchas.

Estas mejoras permiten que el modelo generalice mejor y se acerque a escenarios del mundo real.

## 10) Modelo final

Speech:
Con las mejoras aplicadas reevaluamos el rendimiento del modelo.
El objetivo es obtener un modelo más robusto que pueda limpiar videos con distintos tipos de defectos, no solo una mancha uniforme.

## 11) Otra vez… ¿Mejoras?

Speech:
Finalmente, como ocurre en la mayoría de los proyectos de deep learning, siempre hay margen para seguir mejorando.
Se podrían probar arquitecturas con atención, modelos 3D, diferencias entre frames o incluso transformers específicos para video.
El objetivo es lograr reconstrucciones cada vez más precisas y estables.