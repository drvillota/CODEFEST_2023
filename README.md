# Co2ncientes CODEFEST 2023

## Problema de identificación de textos relacionados con la Amazonía

### Librerías necesarias

Instalar las librerias requeridas en el requirements.txt

### Proceso de desarrollo

- Basados en la información entregada se depuró la información para tener un texto más estandar (poner en minúsculas, arreglar errores ortografícos, lematizar, etc).
- Con la libreria spacy se pudo realizar identificación de entidades nombradas.
- Surge la necesidad de cargar un dataframe en inglés para poder diversificar la clasificación ya que con español esta resulta muy pobre.
- Se vuelve a intentar la identificación de entidades nombradas, con mejores resultados.
- Resultado de los pasos anteriores creamos un modelo para exportar.
- La herramienta final llama al modelo creado y se presenta como una librería.

## Problema de detección de objetos

(no hay código porque se borró todo, solo contamos qué hicimos)

### Librerias

Yolo, cv2 y matplotlyb

### Proceso de trabajo

- Basados en los videos que nos entregaron, con la herramienta v2 segmentamos el video en frames
- Los frames del video fueron examinados por una persona que se encargo de anotar cuando encontrara algo de interés (carros, barcos, zonas deforestadas, etc)
- Con un diccionario de imágenes y su correspondiente anotación (archivos en formato 1.1) pasamos esa información a YOLO
- YOLO generó un modelo que podía identificar objetos en un frame
- Aquí se murió todo, el notebook borró toda la información y al cambiar de notebook el kernel no permitia la instalación de ningúna de las librerías necesarias
- La idea era finalmente iterar con el modelo de YOLO por todos los frames

Imágen de la clasificación realizada

https://raw.githubusercontent.com/drvillota/CODEFEST_2023/main/resultados.jpg