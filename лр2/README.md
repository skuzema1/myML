# SR2
Самостоятельная работа 2
Предварительный анализ данных:

![image](https://user-images.githubusercontent.com/93768556/197277065-cb6950c1-03a0-4eb0-bc39-f42eb3ca2550.png)

Судя по этим графикам:  
* все данные имеют нормальное или близкое к нормальному распределение;  
* между `Price`и `Sales` прослеживается обратная зависимость`;  

Наблюдаемые закономерности могут объясняться влиянием одной или нескольких из фиктивных объясняющих переменных. Построим график, раскрасив точки цветом в зависимости от качеств стелажжа.  
# матричный график разброса с цветом по качеству стелажжа:

![image](https://user-images.githubusercontent.com/93768556/197277538-af6dfcea-d25b-489b-8a98-71d0b3835378.png)

Оценка взаимосвязи:

![image](https://user-images.githubusercontent.com/93768556/197279012-c2db4e39-684c-485a-8dcf-ef5a99567247.png)

Как и предполагалось между `Price`и `Sales` наблюдается умеренная обратная зависимость`

Проверка Y на нормальность:

![image](https://user-images.githubusercontent.com/93768556/197279434-5d16e0ed-1fea-4154-b933-e244efee7888.png)

Логарифмирование не требуется.

![image](https://user-images.githubusercontent.com/93768556/197281160-bb0369bf-fdbf-48ac-add0-48416f863a87.png)

Оценка параметров моделей:

![image](https://user-images.githubusercontent.com/93768556/197282191-c2db9fed-a8ef-4595-88fe-e57f7e8bbfb6.png)
![image](https://user-images.githubusercontent.com/93768556/197282589-2be5def7-ceb9-4ebc-91e5-10eef8fe9a68.png)
![image](https://user-images.githubusercontent.com/93768556/197282928-70bad994-0614-4529-a61d-7d8120a6d9dc.png)
![image](https://user-images.githubusercontent.com/93768556/197282952-a8d1c028-f8fa-4c1b-934c-698816f92c25.png)
![image](https://user-images.githubusercontent.com/93768556/197282967-9a47679d-06b5-4c90-994c-414c98d2848c.png)

Оценка точности моделей:

1. Ошибки для моделей на исходных значениях `Sales:

![image](https://user-images.githubusercontent.com/93768556/197344045-aa721d49-6a9d-44bf-98b4-3c9002633516.png)

2. Определили самую точную: 

![image](https://user-images.githubusercontent.com/93768556/197344413-b9d4c8c3-5739-4091-b9b1-e73fdcb36386.png)

Прогноз по модели на отложенные наблюдения:

![image](https://user-images.githubusercontent.com/93768556/197344444-f155f750-75e0-4e67-a028-ab2a73e7a915.png)



