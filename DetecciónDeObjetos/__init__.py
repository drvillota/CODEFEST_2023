import cv2
import easyocr
import numpy as np
#For the model
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

#Cargamos al lector de análisis de texto en imágenes
reader = easyocr.Reader(['en'])

# Cargamos el modelo obtenido del entrenamiento
PIPELINE_CONFIG = os.path.join('model', 'pipeline.config')
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('model', 'ckpt-3')).expect_partial()

#Usando los labels creamos una función que recibe una imágen y etiqueta
LABELMAP = os.path.join('model', 'label_map.pbtxt')
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(LABELMAP)

"""
Esta función recibe una imágen y le hace el procesamiento necesario para identificar elementos de interés en ella
y de hallarlos guarda la imágen etiquetada y con la info de interés (nombre del archivo, elemento encontrado
coordenadas y tiempo)
Parametros
img: imágen a analizar
image_number: número de imágen (sirve para nombrar el archivo)
destiny_folder: lugar de destino para las imágenes etiquetadas
"""
def process_and_save_images(img, image_number, destiny_folder) -> int:
    #Convertimos la imágen a formato np para procesarla y filtramos
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    f = lambda x : x > 0.35
    true_values = f(detections['detection_scores'])
    detections = {
        'detection_boxes' : detections['detection_boxes'][true_values],
        'detection_classes' : detections['detection_classes'][true_values],
        'detection_scores' : detections['detection_scores'][true_values]
    }

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    if len(detections['detection_boxes']) == 0:
        return image_number

    label_id_offset = 1    
    index = 0
    
    #Recortamos pedazos de imágen en donde se encuentran las horas y las coordenadas
    hour = img[157:173, 20:44]
    minutes = img[157:173, 47:71]
    seconds = img[157:173, 73:97]
    
    grado_1_N = img[40:60, 22:34]
    grado_2_N = img[40:60, 42:65]
    hora_N = img[40:60, 72:94]
    minuto_N = img[40:60, 101:124]
    grado_1_W = img[40:60, 181:203]
    grado_2_W = img[40:60, 211:232]
    hora_W = img[40:60, 241:262]
    minuto_W = img[40:60, 270:292]
        
    #Estimamos con el reader los valores de las zonas en donde estaban las horas y coordenadas
    predicted_hour = '00' if len(reader.readtext(hour))==0 else reader.readtext(hour)[0][1]
    predicted_minutes = '00' if len(reader.readtext(minutes))==0 else reader.readtext(minutes)[0][1]
    predicted_seconds = '00' if len(reader.readtext(seconds))==0 else reader.readtext(seconds)[0][1]
    
    time = "{}:{}:{}".format(predicted_hour, predicted_minutes, predicted_seconds)
    
    predicted_grado_1_N = '4' if len(reader.readtext(grado_1_N))==0 else reader.readtext(grado_1_N)[0][1]
    predicted_grado_2_N = '36' if len(reader.readtext(grado_2_N))==0 else reader.readtext(grado_2_N)[0][1]
    predicted_hora_N = '34' if len(reader.readtext(hora_N))==0 else reader.readtext(hora_N)[0][1]
    predicted_minuto_N = '96' if len(reader.readtext(minuto_N))==0 else reader.readtext(minuto_N)[0][1]
    predicted_grado_1_W = '74' if len(reader.readtext(grado_1_W))==0 else reader.readtext(grado_1_W)[0][1]
    predicted_grado_2_W = '4' if len(reader.readtext(grado_2_W))==0 else reader.readtext(grado_2_W)[0][1]
    predicted_hora_W = '54' if len(reader.readtext(hora_W))==0 else reader.readtext(hora_W)[0][1]
    predicted_minuto_W = '3' if len(reader.readtext(minuto_W))==0 else reader.readtext(minuto_W)[0][1]
    
    coordinates = "{}°{}′{}.{}″ N {}°{}′{}.{}″ W".format(
        predicted_grado_1_N,
        predicted_grado_2_N,
        predicted_hora_N,
        predicted_minuto_N,
        predicted_grado_1_W,
        predicted_grado_2_W,
        predicted_hora_W,
        predicted_minuto_W
    )
    
    #Para cada elemento encontrado procedemos a guardar
    while index < len(detections['detection_boxes']):
        image_np_with_detections = image_np.copy()
        image_number += 1
                
        curr_detections = {
            'detection_boxes' : detections['detection_boxes'][index:index+1],
            'detection_classes' : detections['detection_classes'][index:index+1],
            'detection_scores' : detections['detection_scores'][index:index+1]
        }
        
        #Aquí se añaden los recuadros a la imágen
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                curr_detections['detection_boxes'],
                curr_detections['detection_classes']+label_id_offset,
                curr_detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.35,
                agnostic_mode=False)
        
        file_name = str(image_number) + '.jpg'
        labels = {'0':'casa','1':'maquinaria'}
        
        #Guardamos la imágen y guardamos el resultado en el csv
        cv2.imwrite(destiny_folder + file_name,cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        
        result = file_name + ', ' + labels[str(curr_detections['detection_classes'][0])] + ', ' + time + ', ' + coordinates
                
        print(result)
        
        with open(destiny_folder + 'resultado.csv', 'a', encoding="utf-8") as f:
            f.write(result + '\n')
        
        index += 1
    return image_number

"""
Esta función busca un video y le haya objetos de interés
Parametros
video_path: ruta donde se encuentra el video
output_path: ruta donde se deben subir los resultados
"""
def detect_objects_in_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    framerate = 60
    save_interval = 5
    frame_n = 0
    image_number=0
    #Cada 60 frames busca objetos de interés
    while video.isOpened() and image_number<10:
        frame_n += 60
        video.set(1,frame_n)
        ret, frame = video.read()
        if ret:
            if frame_n % (framerate * save_interval) == 0:
                image_number=process_and_save_images(frame, image_number, output_path)
        else:
            break