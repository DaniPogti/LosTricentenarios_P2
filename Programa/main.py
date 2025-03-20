import cv2 #procesa imagenes y captura de video
import mediapipe as mp #biblioteca para detectar y rastrear manos, rostros, poses

def distancia_euclidiana(p1, p2): #funcion para calcular la distancia euclidiana entre dos puntos p1 y p2 en plano 2d 
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def draw_bounding_box(image, hand_landmarks): #dibuja un cuadro delimitador
    image_height, image_width, _ = image.shape #obtin el alto y ancho de la imagen
    x_min, y_min = image_width, image_height #coordenadas maximas y minimas para el cuadro
    x_max, y_max = 0, 0
    
    # Itera en los puntos de referencia de la mano
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height) #pasa coordenadas a pixeles
        if x < x_min: x_min = x #actualiza coordenadas minimas y max para el cuadro
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) #dibuja el cuadro con las coordenadas calculadas de un color


mp_drawing = mp.solutions.drawing_utils #dibuja los puntos de referencia de la mano
mp_drawing_styles = mp.solutions.drawing_styles #estilos para dibujar los puntos de referencia de la mano
mp_hands = mp.solutions.hands #modelo de manos

cap = cv2.VideoCapture(0) #captura de video
#cap = cv2.VideoCapture('c:/Users/Portillo/Desktop/USAC 2023/Samsumg/Proyecto 2/LosTricentenarios_P2/DataSet/A/A1.jpg')
cap.set(3,1920) #establece el ancho de la imagen
cap.set(4,1080) #establece el largo de la imagen
with mp_hands.Hands( #modelo de manos
    model_complexity=1, #complejidad del modelo
    min_detection_confidence=0.7, #confianza minima de deteccion
    min_tracking_confidence=0.7, #confianza minima de rastreo
    max_num_hands=1) as hands: #numero maximo de manos
  while cap.isOpened(): #mientras la camara este abierta
    success, image = cap.read() #lee la imagen de la camara
    if not success: #si no se pudo leer la imagen
      continue #continua con el siguiente ciclo

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks):
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Draw bounding box
                draw_bounding_box(image, hand_landmarks)

                index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
                
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                int(hand_landmarks.landmark[4].y * image_height))
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                                int(hand_landmarks.landmark[2].y * image_height))
                
                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                int(hand_landmarks.landmark[12].y * image_height))
                
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                int(hand_landmarks.landmark[10].y * image_height))
                
                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                int(hand_landmarks.landmark[16].y * image_height))
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                int(hand_landmarks.landmark[14].y * image_height))
                
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                int(hand_landmarks.landmark[20].y * image_height))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                int(hand_landmarks.landmark[18].y * image_height))
                
                wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))
                
                ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                int(hand_landmarks.landmark[5].y * image_height))
                
                #letra A
                if abs(thumb_tip[0] - thumb_pip[0]) > 30 and \
                    thumb_tip[1] < thumb_pip[1] and \
                    index_finger_tip[1] > index_finger_pip[1] and \
                    middle_finger_tip[1] > middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1]:
                        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
                        cv2.putText(image, 'A', (700, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    3.0, (0, 0, 255), 6)
                    
                #letra B  
                elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
                        middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<40:
                    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
                    cv2.putText(image, 'B', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                #letra C    
                
                elif abs(index_finger_tip[1] - thumb_tip[1]) < 50 and \
                    abs(middle_finger_tip[1] - thumb_tip[1]) < 50 and \
                    abs(ring_finger_tip[1] - thumb_tip[1]) < 50 and \
                    abs(pinky_tip[1] - thumb_tip[1]) < 50 and \
                    index_finger_tip[0] < index_finger_pip[0] and \
                    middle_finger_tip[0] < middle_finger_pip[0] and \
                    ring_finger_tip[0] < ring_finger_pip[0] and \
                    pinky_tip[0] < pinky_pip[0]:
                    print("cccccccccccccccccc")
                    cv2.putText(image, 'C', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)    
                    
                
                #letra D
                elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
                    and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
                    and  pinky_pip[1] - pinky_tip[1]<0\
                    and index_finger_pip[1] - index_finger_tip[1]>0:
                    print("DDDDDDDDDDDD")
                    cv2.putText(image, 'D', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6) 
                #letra E   
                elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
                        and abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                            thumb_tip[1] - index_finger_tip[1] > 0 \
                            and thumb_tip[1] - middle_finger_tip[1] > 0 \
                            and thumb_tip[1] - ring_finger_tip[1] > 0 \
                            and thumb_tip[1] - pinky_tip[1] > 0:
                    print("EEEEEEEEEEEEEEEEE")
                    cv2.putText(image, 'E', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                #letra F   
                elif  pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                    ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
                        and abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) <65:
                    print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
                    cv2.putText(image, 'F', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 30 and \
                    index_finger_tip[1] < index_finger_pip[1] and \
                    middle_finger_tip[1] < middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    thumb_tip[1] > thumb_pip[1]:
                    print("hhhhhhhhhhhhhhhhhh")
                    cv2.putText(image, 'H', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                
                elif abs(pinky_tip[0] - pinky_pip[0]) > 50 and \
                        abs(pinky_tip[1] - pinky_pip[1]) < 20:  # Meñique en posición neutral en el eje vertical
                        print("jjjjjjjjjjjjjjjjjj")
                        cv2.putText(image, 'J', (700, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    3.0, (0, 0, 255), 6)

                elif pinky_tip[1] < pinky_pip[1]:  # Meñique extendido hacia arriba
                    print("iiiiiiiiiiiiiiiiii")
                    cv2.putText(image, 'I', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    
                elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 30 and \
                    index_finger_tip[1] < index_finger_pip[1] and \
                    middle_finger_tip[1] < middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    abs(thumb_tip[0] - middle_finger_pip[0]) < 30 and \
                    thumb_tip[1] > middle_finger_pip[1]:  # Pulgar debajo de la base del medio
                    print("kkkkkkkkkkkkkkkkkk")
                    cv2.putText(image, 'K', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                elif abs(index_finger_tip[0] - thumb_tip[0]) < 50 and \
                    index_finger_tip[0] < index_finger_pip[0] and \
                    thumb_tip[0] < thumb_pip[0] and \
                    middle_finger_tip[0] > middle_finger_pip[0] and \
                    ring_finger_tip[0] > ring_finger_pip[0] and \
                    pinky_tip[0] > pinky_pip[0]:
                    print("llllllllllllllllll")
                    cv2.putText(image, 'L', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    
                elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 30 and \
                    index_finger_tip[1] > index_finger_pip[1] and \
                    middle_finger_tip[1] > middle_finger_pip[1]:
                    print("nnnnnnnnnnnnnnnnnn")
                    cv2.putText(image, 'N', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)

                elif (index_finger_tip[1] > index_finger_pip[1]  # Índice apuntando hacia abajo
                    and middle_finger_tip[1] > middle_finger_pip[1]  # Medio apuntando hacia abajo
                    and ring_finger_tip[1] > ring_finger_pip[1]  # Anular apuntando hacia abajo
                    and distancia_euclidiana(thumb_tip, thumb_pip) < 50):  # Pulgar empuñado
                    print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
                    cv2.putText(image, 'M', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                                        
                elif abs(thumb_tip[0] - index_finger_tip[0]) < 30 and \
                        abs(thumb_tip[1] - index_finger_tip[1]) < 30:
                        print("oooooooooooooooooo")
                        cv2.putText(image, 'O', (700, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    3.0, (0, 0, 255), 6)
                        
                elif abs(middle_finger_tip[0] - thumb_tip[0]) < 20 and \
                    abs(middle_finger_tip[1] - thumb_tip[1]) < 20 and \
                    index_finger_tip[1] > index_finger_pip[1] and \
                    middle_finger_tip[1] > middle_finger_pip[1] and \
                    thumb_tip[1] > thumb_pip[1]:
                    print("pppppppppppppppppp")
                    cv2.putText(image, 'P', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6) 
                                
                #print("pulgar", thumb_tip[1])
                #print("dedo indice",index_finger_tip[1])
                
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()



'''import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Tamaño de la imagen
IMG_SIZE = 200

# Categorías de las imágenes
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]

# Ruta del dataset (ajusta esta ruta a tu sistema)
DATASET_PATH = "c:/Users/Portillo/Desktop/USAC 2023/Samsumg/Proyecto 2/LosTricentenarios_P2/DataSet"

# Función para cargar la primera imagen de cada categoría
def CargarIMG(carpeta, img_size):
    imagenes = []
    for categoria in CATEGORIES:
        path = os.path.join(carpeta, categoria)
        class_num = CATEGORIES.index(categoria)
        
        # Obtener la lista de imágenes en la carpeta de la categoría
        img_list = os.listdir(path)
        
        # Si hay al menos una imagen en la carpeta
        if img_list:
            img_name = img_list[0]  # Tomar la primera imagen
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                imagenes.append((img, class_num))
    return imagenes

# Cargar las primeras imágenes de cada categoría
imagenes = CargarIMG(DATASET_PATH, IMG_SIZE)

# Mostrar las imágenes en una gráfica
def graficoIMG(imagenes):
    num_images = len(imagenes)  # Número de imágenes (una por categoría)
    rows = int(np.ceil(np.sqrt(num_images)))  # Calcular el número de filas y columnas para la cuadrícula
    cols = int(np.ceil(num_images / rows))
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(imagenes[i][0], cv2.COLOR_BGR2RGB))
        plt.title(CATEGORIES[imagenes[i][1]])
        plt.axis('off')
    plt.tight_layout()  # Ajustar el espaciado entre imágenes
    plt.show()

# Mostrar las imágenes
graficoIMG(imagenes)'''