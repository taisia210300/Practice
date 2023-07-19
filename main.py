import cv2      # Импорт необходимых для обнаружения объектов на изображении библиотек
import numpy as np
from tkinter import *  # Импорт необходимых для создания формы библиотек 
from tkinter import filedialog
from os import path
from tkinter.messagebox import showerror

def btn_click():
    # Функция для распознавания и определения координат объектов на изображении, 
    # где start_image - исходное изображение,
    # final_image - изображение с отмеченными объектами и подписями к ним
    def apply_object_detection(start_image):
        height, width, _ = start_image.shape
        
        # создать 4D blob (нормализация значений пикселей и изменение размера изображения)
        blob = cv2.dnn.blobFromImage(start_image, 1 / 255, (608, 608), 
                                 (0, 0, 0), swapRB=True, crop=False)
        
        # устанавливает blob как вход сети
        net.setInput(blob)

        # прямая связь (вывод) и получение выхода сети
        outs = net.forward(out_layers)
        class_indexes, class_scores, boxes = ([] for i in range(3))
        objects_count = 0

        # Запуск поиска объектов на изображении
        # перебираем каждый из выходов слоя
        for out in outs:
            # перебираем каждое обнаружение объекта
            for obj in out:
                # извлекаем идентификатор класса (метку) и достоверность (как вероятность)
                # обнаружение текущего объекта
                scores = obj[5:]
                class_index = np.argmax(scores)
                class_score = scores[class_index]
                # отбросим слабые прогнозы, убедившись, что обнаруженная
                # вероятность больше 0
                if class_score > 0:
                    # масштабируем координаты ограничивающего прямоугольника относительно
                    # размера изображения, учитывая, что YOLO
                    # возвращает центральные координаты (x, y) ограничивающего
                    # поля, за которым следуют ширина и высота поля
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    obj_width = int(obj[2] * width)
                    obj_height = int(obj[3] * height)
                    box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                    # обновим наш список координат ограничивающего прямоугольника, 
                    # достоверности, и идентификаторы класса
                    boxes.append(box)
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))

        # Выбор
        # выполняем не максимальное подавление с учетом оценок, определенных ранее
        chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
        # перебираем сохраняемые индексы
        for box_index in chosen_boxes:
            box_index = box_index
            box = boxes[box_index]
            class_index = class_indexes[box_index]

            # Для отладки рисуем объекты, входящие в нужные классы
            if classes[class_index] in classes_to_look_for:
                objects_count += 1
                start_image = draw_object(start_image, class_index, box)

        final_image = draw_object_count(start_image, objects_count)
        cv2.imwrite('Result/Output/res.png', final_image)
        return final_image

    # Функция, обводящая найденные на изображении объекты с помощью координат границ, 
    # полученных из функции apply_object_detection, с подписями, где 
    # start_image - исходное изображение,
    # index - индекс класса объекта, определенного с помощью YOLO,
    # box - координаты области вокруг объекта,
    # final_image - изображение с отмеченными объектами
    def draw_object(start_image, index, box):
        x, y, w, h = box
        start = (x, y)
        end = (x + w, y + h)
        color = (0, 255, 0)
        width = 2
        # рисуем прямоугольник ограничивающей рамки
        final_image = cv2.rectangle(start_image, start, end, color, width)

        start = (x, y - 10)
        # настройки текста
        font_size = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 2
        text = classes[index]
        # помещаем текст
        final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

        return final_image

    # Функция для вывода количества найденных объектов на изображении, где 
    # start_image - исходное изображение,
    # objects_count - количество объектов желаемого класса,
    # final_image - изображение с отмеченным количеством найденных объектов
    def draw_object_count(start_image, objects_count):
        start = (10, 100)
        font_size = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 3
        text = "Objects found: " + str(objects_count)

        # Вывод текста обводкой (чтобы его было видно при разном освещении картинки)
        white_color = (255, 255, 255)
        black_outline_color = (0, 0, 0)
        final_image = cv2.putText(start_image, text, start, font, font_size, black_outline_color, 
                              width * 3, cv2.LINE_AA)
        final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width,
                               cv2.LINE_AA)

        return final_image

    # Функция для получения результата - входного изображения с отрисованными объектами и их количеством
    def start_detection(img_path):
        try:
            # Применение методов распознавания объектов на изображении YOLO
            image = cv2.imread(img_path)
            image = apply_object_detection(image)

            # Отображение обработанного изображения на экране
            cv2.imshow("Image", image)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass
    
    if label_3.cget("text") != "":
        # Загрузка YOLO из файлов и настройка сети
        # конфигурация нейронной сети и
        # файл весов сети YOLO
        net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
        layer_names = net.getLayerNames()
        out_layers_indexes = net.getUnconnectedOutLayers()
        out_layers = [layer_names[index - 1] for index in out_layers_indexes]

        # Загрузка из файла классов объектов, которые YOLO может обнаружить
        with open("Resources/coco.names.txt") as file:
            classes = file.read().split("\n") 

        # Определение классов, которые будут приоритетными для поиска по изображению, 
        # имена находятся в файле coco.names.txt
        image = label_3.cget("text")
        look_for = text_2.get().split(',')
     
        # Удаление пробелов
        list_look_for = []
        for look in look_for:
            list_look_for.append(look.strip())

        classes_to_look_for = list_look_for
        start_detection(image)
    else: 
        showerror(title="Ошибка!", message="Файл не выбран!")

# Функция для получения пути к изображению
def get_file_path():
   file = filedialog.askopenfilename(initialdir = path.dirname("Input"))   
   label_3.config(text = file)

# Функция main, где создается форма для выбора изображения и ввода названий объектов для поиска
if __name__ == '__main__':
    root = Tk()
    root.geometry('450x250')
    root.title("Обнаружение объектов на изображении")
    
    canvas = Canvas(root, height=450, width=250)
    canvas.pack()
    frame = Frame(root)
    frame.place(relx=0.15, rely=0.15, relheight=0.7, relwidth=0.7)

    label_1 = Label(frame, text="Выберите изображение: ")
    label_1.pack()
    label_3 = Label(frame)
    label_3.pack()
    btn1 = Button(frame, text="Выбрать файл", command=get_file_path)
    btn1.pack()
    
    label_4 = Label(frame, text="Объекты, которые вы хотите найти на изображении")
    label_4.pack()
    label_5 = Label(frame, text="(если их несколько, вводите через запятую): ")
    label_5.pack()
    text_2 = Entry(frame)
    text_2.pack()

    
    btn2 = Button(frame, text="Найти", command=btn_click)
    btn2.pack()

    root.mainloop()