# Лабораторная работа №1. Оценка точности модели с непрерывной зависимой переменной

Задача1. Повторила расчёты и построила графики из первой лабораторной, изменив функцию согласно своему варианту (12)

![image](https://user-images.githubusercontent.com/93768556/193822005-b331735b-2b2b-4736-8264-acd05f274766.png)
 
 Подставила 5 различных значений df(число узлов сплайна). Кривая MSE на обучающей выборке стабильно снижается с ростом узлов сплайна. Чем больше наблюдений, через которые прошёл сплайн, тем точнее модель. Это говорит о переобучении. 

![image](https://user-images.githubusercontent.com/93768556/193822743-131c3e8a-b7bc-4f59-a96a-8597f76cbb41.png)

![image](https://user-images.githubusercontent.com/93768556/193822773-0f092cc3-2d86-48bf-8b1c-2b56363884cc.png)

![image](https://user-images.githubusercontent.com/93768556/193822851-e835516b-df4d-4b1b-810c-8d6c63fc0791.png)

![image](https://user-images.githubusercontent.com/93768556/193822878-8fb4d8e3-f82e-4975-92b8-ae78d776390d.png)

![image](https://user-images.githubusercontent.com/93768556/193823049-05482f22-0b2e-4ad4-a9ad-a7d47222c7ca.png)

Лучшую модель следуют выбирать по минимуму на кривой MSE на тестовой выборке. В моём случае оптимальным является сплайн с s=400.

![image](https://user-images.githubusercontent.com/93768556/193823115-ad5f436e-8ae4-425a-b12e-f223c917efe5.png)

![image](https://user-images.githubusercontent.com/93768556/193823302-8d8a4521-54b3-40c4-9284-08bf3080ff67.png)

Задача2.

Подставила 3 различных значения sigma (в случайном шуме) согласно с воему варианту. С уменьшением sigma уменьшается значение оптимальное число степеней свободы s.

При sigma = 2.5 оптимальное число степеней свободы s=600:

![image](https://user-images.githubusercontent.com/93768556/193826929-1ed61d5c-ad60-44ea-80f4-a9ada1942006.png)

При sigma = 2 оптимальное число степеней свободы s=525:

![image](https://user-images.githubusercontent.com/93768556/193827193-92ec1cbc-f464-4fe4-b3c5-20b2541e2034.png)

При sigma = 1.5 оптимальное число степеней свободы s=450:

![image](https://user-images.githubusercontent.com/93768556/193827316-73bd6011-5c06-454e-95f9-5f6259155dd7.png)

