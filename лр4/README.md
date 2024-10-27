{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6b8b1c8d",
   "metadata": {},
   "source": [
    "`Дисциплина: Методы и технологии машинного обучения`   \n",
    "`Уровень подготовки: бакалавриат`   \n",
    "`Направление подготовки: 01.03.02 Прикладная математика и информатика`   \n",
    "`Семестр: осень 2022/2023`   "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fc5c8ab",
   "metadata": {},
   "source": [
    "# Лабораторная работа №5: Методы, основанные на деревьях решений. Регрессионные деревья. Деревья классификации. Бэггинг.  \n",
    "\n",
    "В практических примерах ниже показано:   \n",
    "\n",
    "* как делать перекодировку признаков в номинальной и порядковой шкалах\n",
    "* как вырастить дерево и сделать обрезку его ветвей   \n",
    "* как настроить модель бэггинга\n",
    "* как подбирать настроечные параметры моделей методом сеточного поиска  \n",
    "\n",
    "Точность всех моделей оценивается методом перекрёстной проверки по 5 блокам."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61b9ab5d",
   "metadata": {},
   "source": [
    "# Указания к выполнению\n",
    "\n",
    "\n",
    "## Загружаем пакеты"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c8fa189c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# загрузка пакетов: инструменты --------------------------------------------\n",
    "#  работа с массивами\n",
    "import numpy as np\n",
    "#  фреймы данных\n",
    "import pandas as pd\n",
    "#  графики\n",
    "import matplotlib as mpl\n",
    "#  стили и шаблоны графиков на основе matplotlib\n",
    "import seaborn as sns\n",
    "# проверка существования файла на диске\n",
    "from pathlib import Path\n",
    "# для форматирования результатов с помощью Markdown\n",
    "from IPython.display import Markdown, display\n",
    "# перекодировка категориальных переменных\n",
    "from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder\n",
    "# хи-квадрат тест на независимость по таблице сопряжённости\n",
    "from scipy.stats import chi2_contingency\n",
    "#  для таймера\n",
    "import time\n",
    "\n",
    "# загрузка пакетов: данные -------------------------------------------------\n",
    "from sklearn import datasets\n",
    "\n",
    "# загрузка пакетов: модели -------------------------------------------------\n",
    "#  дерево классификации\n",
    "from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree\n",
    "# перекрёстная проверка и метод проверочной выборки\n",
    "from sklearn.model_selection import cross_val_score, train_test_split\n",
    "# для перекрёстной проверки и сеточного поиска\n",
    "from sklearn.model_selection import KFold, GridSearchCV\n",
    "# бэггинг\n",
    "from sklearn.ensemble import BaggingClassifier\n",
    "# случайный лес\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "# бустинг\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "#  сводка по точности классификации\n",
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "43a4a48e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# константы\n",
    "#  ядро для генератора случайных чисел\n",
    "my_seed = 13\n",
    "#  создаём псевдоним для короткого обращения к графикам\n",
    "plt = mpl.pyplot\n",
    "# настройка стиля и отображения графиков\n",
    "#  примеры стилей и шаблонов графиков: \n",
    "#  http://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html\n",
    "mpl.style.use('seaborn-whitegrid')\n",
    "sns.set_palette(\"Set2\")\n",
    "# раскомментируйте следующую строку, чтобы посмотреть палитру\n",
    "# sns.color_palette(\"Set2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "907a2353",
   "metadata": {},
   "outputs": [],
   "source": [
    "# функция форматирования результатов с использованием Markdown\n",
    "def printmd(string):\n",
    "    display(Markdown(string))\n",
    "    \n",
    "# функции для попарной конкатенации элементов двух списков\n",
    "concat_func_md = lambda x, y: '`' + str(x) + \"`:&ensp;&ensp;&ensp;&ensp;\" + str(y)\n",
    "concat_func = lambda x, y: str(x) + ' ' * 4 + str(y)\n",
    "\n",
    "\n",
    "# функция, которая строит график важности признаков в модели случайного леса\n",
    "#  источник: https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html\n",
    "def plot_feature_importance(importance, names, model_type) :\n",
    "    #Create arrays from feature importance and feature names\n",
    "    feature_importance = np.array(importance)\n",
    "    feature_names = np.array(names)\n",
    "\n",
    "    #Create a DataFrame using a Dictionary\n",
    "    data={'feature_names':feature_names,'feature_importance':feature_importance}\n",
    "    fi_df = pd.DataFrame(data)\n",
    "\n",
    "    #Sort the DataFrame in order decreasing feature importance\n",
    "    fi_df.sort_values(by=['feature_importance'], ascending=False,\n",
    "                      inplace=True)\n",
    "\n",
    "    #Define size of bar plot\n",
    "    plt.figure(figsize=(10,8))\n",
    "    #Plot Searborn bar chart\n",
    "    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])\n",
    "    #Add chart labels\n",
    "    plt.title('Важность признаков в модели: ' + model_type)\n",
    "    plt.xlabel('Важность признака')\n",
    "    plt.ylabel('')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87618f1b",
   "metadata": {},
   "source": [
    "## Загружаем данные"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a7e42f2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Число строк и столбцов в наборе данных:\n",
      " (30000, 24)\n"
     ]
    }
   ],
   "source": [
    "# загружаем таблицу и превращаем её во фрейм\n",
    "fileURL = 'https://raw.githubusercontent.com/ania607/ML/main/data/default_of_credit_card_clients.csv'\n",
    "DF_raw = pd.read_csv(fileURL)\n",
    "\n",
    "# выясняем размерность фрейма\n",
    "print('Число строк и столбцов в наборе данных:\\n', DF_raw.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "07670649",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LIMIT_BAL    int64\n",
       "SEX          int64\n",
       "EDUCATION    int64\n",
       "MARRIAGE     int64\n",
       "AGE          int64\n",
       "PAY_0        int64\n",
       "PAY_2        int64\n",
       "PAY_3        int64\n",
       "PAY_4        int64\n",
       "PAY_5        int64\n",
       "PAY_6        int64\n",
       "BILL_AMT1    int64\n",
       "BILL_AMT2    int64\n",
       "BILL_AMT3    int64\n",
       "BILL_AMT4    int64\n",
       "BILL_AMT5    int64\n",
       "BILL_AMT6    int64\n",
       "PAY_AMT1     int64\n",
       "PAY_AMT2     int64\n",
       "PAY_AMT3     int64\n",
       "PAY_AMT4     int64\n",
       "PAY_AMT5     int64\n",
       "PAY_AMT6     int64\n",
       "Y            int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# типы столбцов\n",
    "DF_raw.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fe68e45",
   "metadata": {},
   "source": [
    "Проблем нет. Все данные воспринимаются как `int`. Категориальные данные закодированны."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fca342e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>LIMIT_BAL</th>\n",
       "      <th>SEX</th>\n",
       "      <th>EDUCATION</th>\n",
       "      <th>MARRIAGE</th>\n",
       "      <th>AGE</th>\n",
       "      <th>PAY_0</th>\n",
       "      <th>PAY_2</th>\n",
       "      <th>PAY_3</th>\n",
       "      <th>PAY_4</th>\n",
       "      <th>PAY_5</th>\n",
       "      <th>...</th>\n",
       "      <th>BILL_AMT4</th>\n",
       "      <th>BILL_AMT5</th>\n",
       "      <th>BILL_AMT6</th>\n",
       "      <th>PAY_AMT1</th>\n",
       "      <th>PAY_AMT2</th>\n",
       "      <th>PAY_AMT3</th>\n",
       "      <th>PAY_AMT4</th>\n",
       "      <th>PAY_AMT5</th>\n",
       "      <th>PAY_AMT6</th>\n",
       "      <th>Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20000</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-2</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>689</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>120000</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>26</td>\n",
       "      <td>-1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3272</td>\n",
       "      <td>3455</td>\n",
       "      <td>3261</td>\n",
       "      <td>0</td>\n",
       "      <td>1000</td>\n",
       "      <td>1000</td>\n",
       "      <td>1000</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>90000</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>34</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>14331</td>\n",
       "      <td>14948</td>\n",
       "      <td>15549</td>\n",
       "      <td>1518</td>\n",
       "      <td>1500</td>\n",
       "      <td>1000</td>\n",
       "      <td>1000</td>\n",
       "      <td>1000</td>\n",
       "      <td>5000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>50000</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>37</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>28314</td>\n",
       "      <td>28959</td>\n",
       "      <td>29547</td>\n",
       "      <td>2000</td>\n",
       "      <td>2019</td>\n",
       "      <td>1200</td>\n",
       "      <td>1100</td>\n",
       "      <td>1069</td>\n",
       "      <td>1000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>50000</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>57</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>20940</td>\n",
       "      <td>19146</td>\n",
       "      <td>19131</td>\n",
       "      <td>2000</td>\n",
       "      <td>36681</td>\n",
       "      <td>10000</td>\n",
       "      <td>9000</td>\n",
       "      <td>689</td>\n",
       "      <td>679</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>50000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>37</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>19394</td>\n",
       "      <td>19619</td>\n",
       "      <td>20024</td>\n",
       "      <td>2500</td>\n",
       "      <td>1815</td>\n",
       "      <td>657</td>\n",
       "      <td>1000</td>\n",
       "      <td>1000</td>\n",
       "      <td>800</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>500000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>542653</td>\n",
       "      <td>483003</td>\n",
       "      <td>473944</td>\n",
       "      <td>55000</td>\n",
       "      <td>40000</td>\n",
       "      <td>38000</td>\n",
       "      <td>20239</td>\n",
       "      <td>13750</td>\n",
       "      <td>13770</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \\\n",
       "0      20000    2          2         1   24      2      2     -1     -1   \n",
       "1     120000    2          2         2   26     -1      2      0      0   \n",
       "2      90000    2          2         2   34      0      0      0      0   \n",
       "3      50000    2          2         1   37      0      0      0      0   \n",
       "4      50000    1          2         1   57     -1      0     -1      0   \n",
       "5      50000    1          1         2   37      0      0      0      0   \n",
       "6     500000    1          1         2   29      0      0      0      0   \n",
       "\n",
       "   PAY_5  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  \\\n",
       "0     -2  ...          0          0          0         0       689         0   \n",
       "1      0  ...       3272       3455       3261         0      1000      1000   \n",
       "2      0  ...      14331      14948      15549      1518      1500      1000   \n",
       "3      0  ...      28314      28959      29547      2000      2019      1200   \n",
       "4      0  ...      20940      19146      19131      2000     36681     10000   \n",
       "5      0  ...      19394      19619      20024      2500      1815       657   \n",
       "6      0  ...     542653     483003     473944     55000     40000     38000   \n",
       "\n",
       "   PAY_AMT4  PAY_AMT5  PAY_AMT6  Y  \n",
       "0         0         0         0  1  \n",
       "1      1000         0      2000  1  \n",
       "2      1000      1000      5000  0  \n",
       "3      1100      1069      1000  0  \n",
       "4      9000       689       679  0  \n",
       "5      1000      1000       800  0  \n",
       "6     20239     13750     13770  0  \n",
       "\n",
       "[7 rows x 24 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# первые 7 строк столбцов типа int64\n",
    "DF_raw.loc[:, DF_raw.columns[DF_raw.dtypes == 'int64']].head(7)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2a443be",
   "metadata": {},
   "source": [
    "Функция построения дерева классификации `DecisionTreeClassifier()` требует числовых порядковых значений переменных. Видно, что столбцы типа `int64` либо порядковые (`temperature`), либо бинарные (все остальные), их преобразовывать нет необходимости.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3623e5c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# меняем тип столбцов на категориальные\n",
    "#for col in DF_raw.columns[DF_raw.dtypes == 'object'] :\n",
    "#    DF_raw[col] = DF_raw[col].astype('category')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ce7ffd1",
   "metadata": {},
   "source": [
    "Отложим 15% наблюдений для прогноза.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "13f9300b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# наблюдения для моделирования\n",
    "DF = DF_raw.sample(frac=0.85, random_state=my_seed)\n",
    "# отложенные наблюдения\n",
    "DF_predict = DF_raw.drop(DF.index)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1635163a",
   "metadata": {},
   "source": [
    "# Предварительный анализ данных  \n",
    "\n",
    "## Описательные статистики  \n",
    "\n",
    "Стандартный подсчёт статистик с помощью фунции `describe()`.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "78108987",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1], dtype=int64)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# метки классов\n",
    "DF.Y.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "2e4cb7c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.78078\n",
       "1    0.21922\n",
       "Name: Y, dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# доли классов\n",
    "np.around(DF.Y.value_counts() / len(DF.index), 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9a7804d3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>LIMIT_BAL</th>\n",
       "      <th>SEX</th>\n",
       "      <th>EDUCATION</th>\n",
       "      <th>MARRIAGE</th>\n",
       "      <th>AGE</th>\n",
       "      <th>PAY_0</th>\n",
       "      <th>PAY_2</th>\n",
       "      <th>PAY_3</th>\n",
       "      <th>PAY_4</th>\n",
       "      <th>PAY_5</th>\n",
       "      <th>...</th>\n",
       "      <th>BILL_AMT4</th>\n",
       "      <th>BILL_AMT5</th>\n",
       "      <th>BILL_AMT6</th>\n",
       "      <th>PAY_AMT1</th>\n",
       "      <th>PAY_AMT2</th>\n",
       "      <th>PAY_AMT3</th>\n",
       "      <th>PAY_AMT4</th>\n",
       "      <th>PAY_AMT5</th>\n",
       "      <th>PAY_AMT6</th>\n",
       "      <th>Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>2.550000e+04</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "      <td>25500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>167848.614902</td>\n",
       "      <td>1.604510</td>\n",
       "      <td>1.854078</td>\n",
       "      <td>1.551137</td>\n",
       "      <td>35.473647</td>\n",
       "      <td>-0.019294</td>\n",
       "      <td>-0.136745</td>\n",
       "      <td>-0.171020</td>\n",
       "      <td>-0.226353</td>\n",
       "      <td>-0.272706</td>\n",
       "      <td>...</td>\n",
       "      <td>43011.959020</td>\n",
       "      <td>40047.777804</td>\n",
       "      <td>38657.944118</td>\n",
       "      <td>5721.081059</td>\n",
       "      <td>5.912399e+03</td>\n",
       "      <td>5195.790706</td>\n",
       "      <td>4851.610588</td>\n",
       "      <td>4847.703137</td>\n",
       "      <td>5248.818863</td>\n",
       "      <td>0.219216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>129894.926164</td>\n",
       "      <td>0.488965</td>\n",
       "      <td>0.791689</td>\n",
       "      <td>0.521709</td>\n",
       "      <td>9.224644</td>\n",
       "      <td>1.125633</td>\n",
       "      <td>1.197933</td>\n",
       "      <td>1.193489</td>\n",
       "      <td>1.167079</td>\n",
       "      <td>1.131698</td>\n",
       "      <td>...</td>\n",
       "      <td>63681.324672</td>\n",
       "      <td>60026.412733</td>\n",
       "      <td>59003.614301</td>\n",
       "      <td>16819.669309</td>\n",
       "      <td>2.259416e+04</td>\n",
       "      <td>16745.180224</td>\n",
       "      <td>15980.476266</td>\n",
       "      <td>15559.835697</td>\n",
       "      <td>17901.580110</td>\n",
       "      <td>0.413723</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>10000.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>-2.000000</td>\n",
       "      <td>-2.000000</td>\n",
       "      <td>-2.000000</td>\n",
       "      <td>-2.000000</td>\n",
       "      <td>-2.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>-170000.000000</td>\n",
       "      <td>-81334.000000</td>\n",
       "      <td>-339603.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>50000.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>2280.000000</td>\n",
       "      <td>1740.750000</td>\n",
       "      <td>1242.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>8.360000e+02</td>\n",
       "      <td>390.000000</td>\n",
       "      <td>298.750000</td>\n",
       "      <td>243.750000</td>\n",
       "      <td>108.750000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>140000.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>34.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>19010.000000</td>\n",
       "      <td>18067.000000</td>\n",
       "      <td>17001.500000</td>\n",
       "      <td>2111.500000</td>\n",
       "      <td>2.010000e+03</td>\n",
       "      <td>1826.500000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>240000.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>41.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>54429.500000</td>\n",
       "      <td>50065.250000</td>\n",
       "      <td>49162.750000</td>\n",
       "      <td>5015.000000</td>\n",
       "      <td>5.000000e+03</td>\n",
       "      <td>4512.750000</td>\n",
       "      <td>4027.000000</td>\n",
       "      <td>4064.250000</td>\n",
       "      <td>4000.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>800000.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>79.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>706864.000000</td>\n",
       "      <td>587067.000000</td>\n",
       "      <td>699944.000000</td>\n",
       "      <td>873552.000000</td>\n",
       "      <td>1.684259e+06</td>\n",
       "      <td>889043.000000</td>\n",
       "      <td>621000.000000</td>\n",
       "      <td>426529.000000</td>\n",
       "      <td>528666.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           LIMIT_BAL           SEX     EDUCATION      MARRIAGE           AGE  \\\n",
       "count   25500.000000  25500.000000  25500.000000  25500.000000  25500.000000   \n",
       "mean   167848.614902      1.604510      1.854078      1.551137     35.473647   \n",
       "std    129894.926164      0.488965      0.791689      0.521709      9.224644   \n",
       "min     10000.000000      1.000000      0.000000      0.000000     21.000000   \n",
       "25%     50000.000000      1.000000      1.000000      1.000000     28.000000   \n",
       "50%    140000.000000      2.000000      2.000000      2.000000     34.000000   \n",
       "75%    240000.000000      2.000000      2.000000      2.000000     41.000000   \n",
       "max    800000.000000      2.000000      6.000000      3.000000     79.000000   \n",
       "\n",
       "              PAY_0         PAY_2         PAY_3         PAY_4         PAY_5  \\\n",
       "count  25500.000000  25500.000000  25500.000000  25500.000000  25500.000000   \n",
       "mean      -0.019294     -0.136745     -0.171020     -0.226353     -0.272706   \n",
       "std        1.125633      1.197933      1.193489      1.167079      1.131698   \n",
       "min       -2.000000     -2.000000     -2.000000     -2.000000     -2.000000   \n",
       "25%       -1.000000     -1.000000     -1.000000     -1.000000     -1.000000   \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "75%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "max        8.000000      8.000000      8.000000      8.000000      8.000000   \n",
       "\n",
       "       ...      BILL_AMT4      BILL_AMT5      BILL_AMT6       PAY_AMT1  \\\n",
       "count  ...   25500.000000   25500.000000   25500.000000   25500.000000   \n",
       "mean   ...   43011.959020   40047.777804   38657.944118    5721.081059   \n",
       "std    ...   63681.324672   60026.412733   59003.614301   16819.669309   \n",
       "min    ... -170000.000000  -81334.000000 -339603.000000       0.000000   \n",
       "25%    ...    2280.000000    1740.750000    1242.000000    1000.000000   \n",
       "50%    ...   19010.000000   18067.000000   17001.500000    2111.500000   \n",
       "75%    ...   54429.500000   50065.250000   49162.750000    5015.000000   \n",
       "max    ...  706864.000000  587067.000000  699944.000000  873552.000000   \n",
       "\n",
       "           PAY_AMT2       PAY_AMT3       PAY_AMT4       PAY_AMT5  \\\n",
       "count  2.550000e+04   25500.000000   25500.000000   25500.000000   \n",
       "mean   5.912399e+03    5195.790706    4851.610588    4847.703137   \n",
       "std    2.259416e+04   16745.180224   15980.476266   15559.835697   \n",
       "min    0.000000e+00       0.000000       0.000000       0.000000   \n",
       "25%    8.360000e+02     390.000000     298.750000     243.750000   \n",
       "50%    2.010000e+03    1826.500000    1500.000000    1500.000000   \n",
       "75%    5.000000e+03    4512.750000    4027.000000    4064.250000   \n",
       "max    1.684259e+06  889043.000000  621000.000000  426529.000000   \n",
       "\n",
       "            PAY_AMT6             Y  \n",
       "count   25500.000000  25500.000000  \n",
       "mean     5248.818863      0.219216  \n",
       "std     17901.580110      0.413723  \n",
       "min         0.000000      0.000000  \n",
       "25%       108.750000      0.000000  \n",
       "50%      1500.000000      0.000000  \n",
       "75%      4000.000000      0.000000  \n",
       "max    528666.000000      1.000000  \n",
       "\n",
       "[8 rows x 24 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# описательные статистики\n",
    "DF.iloc[:, :].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "42a96cd9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Series([], dtype: float64)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# считаем пропуски в столбцах, выводим ненулевые значения\n",
    "nas = DF.isna().sum()\n",
    "nas = np.around(nas / DF.shape[0], 3)\n",
    "nas[nas > 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b618a25b",
   "metadata": {},
   "source": [
    "Пропусков нет.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8992461",
   "metadata": {},
   "source": [
    "## Распределение предикторов внутри классов  по зависимой переменной\n",
    "\n",
    "Все объясняющие переменные являются категориальными, поэтому оценивать их связь с зависимой переменной с помощью корреляционной матрицы некорректно. Вместо этого можно воспользоваться [критерием согласия Хи-квадрат](https://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B9_%D1%81%D0%BE%D0%B3%D0%BB%D0%B0%D1%81%D0%B8%D1%8F_%D0%9F%D0%B8%D1%80%D1%81%D0%BE%D0%BD%D0%B0), который рассчитывается по таблице сопряжённости. Нулевая гипотеза теста: распределение долей в таблице сопряжённости случайно, т.е. два показателя независимы друг от друга.     \n",
    "Проведём тест для всех пар \"объясняющая переменная\" – \"зависимая переменная\" и выведем те пары, для которых соответствующее критерию p-значение больше 0.05 (т.е. нулевая гипотеза принимается, переменные независимы).  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "14078cd3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BILL_AMT1 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.8652\n",
      "BILL_AMT2 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.7968\n",
      "BILL_AMT3 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.7261\n",
      "BILL_AMT4 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.6965\n",
      "BILL_AMT5 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.794\n",
      "BILL_AMT6 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 0.7267\n",
      "PAY_AMT1 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n",
      "PAY_AMT2 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n",
      "PAY_AMT3 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n",
      "PAY_AMT4 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n",
      "PAY_AMT5 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n",
      "PAY_AMT6 и Y \n",
      "H_0: переменные распределены независимо друг от друга \n",
      "P-значение: 1.0\n"
     ]
    }
   ],
   "source": [
    "for col in DF.columns[:24] :\n",
    "    con_tab = pd.crosstab(DF[col], DF['Y'])\n",
    "    c, p, dof, expected = chi2_contingency(con_tab)\n",
    "    if p > 0.05 :\n",
    "        print(col, 'и Y',\n",
    "              '\\nH_0: переменные распределены независимо друг от друга', \n",
    "              '\\nP-значение:', np.around(p, 4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7d21a27",
   "metadata": {},
   "source": [
    "Интересный результат: полное совпадение p-значений – объясняется тем, что на самом деле `PAY_AMT6`, `PAY_AMT5`, `PAY_AMT4`, `PAY_AMT3`, `PAY_AMT2`, `PAY_AMT1` противоположны друг другу. Связь между ними функциональная: если направление на статуса платежа не совпадает с направлением на исходное место назначения (`PAY_AMT6 == 1`), то оно противоположно (`PAY_AMT6 == -1`), и наоборот. Поэтому в модель имело бы смысл включать только одну из, так и сделаем, однако стоит учитывать что P-значение равно 1, слишком много, смысла огромного нет.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "eae7ad7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# исключаем direction_opp \n",
    "#  из обучающей выборки\n",
    "DF = DF.drop(['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2'], axis=1)\n",
    "#  и из отложенных наблюдений\n",
    "DF_predict = DF_predict.drop(['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04137e37",
   "metadata": {},
   "source": [
    "## Перекодировка номинальной и порядковой шкалы   \n",
    "\n",
    "Перекодировка не треубуется так как исходные данные уже заходированы.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "45c489fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>LIMIT_BAL</th>\n",
       "      <th>SEX</th>\n",
       "      <th>EDUCATION</th>\n",
       "      <th>MARRIAGE</th>\n",
       "      <th>AGE</th>\n",
       "      <th>PAY_0</th>\n",
       "      <th>PAY_2</th>\n",
       "      <th>PAY_3</th>\n",
       "      <th>PAY_4</th>\n",
       "      <th>PAY_5</th>\n",
       "      <th>PAY_6</th>\n",
       "      <th>BILL_AMT1</th>\n",
       "      <th>BILL_AMT2</th>\n",
       "      <th>BILL_AMT3</th>\n",
       "      <th>BILL_AMT4</th>\n",
       "      <th>BILL_AMT5</th>\n",
       "      <th>BILL_AMT6</th>\n",
       "      <th>PAY_AMT1</th>\n",
       "      <th>Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3195</th>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>39</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>7824</td>\n",
       "      <td>9000</td>\n",
       "      <td>9867</td>\n",
       "      <td>11929</td>\n",
       "      <td>12091</td>\n",
       "      <td>12245</td>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15960</th>\n",
       "      <td>300000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>217329</td>\n",
       "      <td>206703</td>\n",
       "      <td>203164</td>\n",
       "      <td>164371</td>\n",
       "      <td>161331</td>\n",
       "      <td>154515</td>\n",
       "      <td>8000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17713</th>\n",
       "      <td>120000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>170263</td>\n",
       "      <td>157881</td>\n",
       "      <td>160796</td>\n",
       "      <td>160168</td>\n",
       "      <td>156165</td>\n",
       "      <td>158850</td>\n",
       "      <td>5520</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17897</th>\n",
       "      <td>110000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>87487</td>\n",
       "      <td>90120</td>\n",
       "      <td>93661</td>\n",
       "      <td>96185</td>\n",
       "      <td>99667</td>\n",
       "      <td>102978</td>\n",
       "      <td>5000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23049</th>\n",
       "      <td>200000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>-1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2217</td>\n",
       "      <td>61328</td>\n",
       "      <td>2877</td>\n",
       "      <td>160944</td>\n",
       "      <td>156864</td>\n",
       "      <td>160066</td>\n",
       "      <td>61634</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \\\n",
       "3195       20000    1          3         2   39      0      0      0      0   \n",
       "15960     300000    2          1         2   30      0      0      0      0   \n",
       "17713     120000    2          1         2   30      0      0      0      0   \n",
       "17897     110000    1          1         2   27      0      0      0      0   \n",
       "23049     200000    2          1         1   41      1     -1      0     -1   \n",
       "\n",
       "       PAY_5  PAY_6  BILL_AMT1  BILL_AMT2  BILL_AMT3  BILL_AMT4  BILL_AMT5  \\\n",
       "3195       2      2       7824       9000       9867      11929      12091   \n",
       "15960      0      0     217329     206703     203164     164371     161331   \n",
       "17713      0      0     170263     157881     160796     160168     156165   \n",
       "17897      0      0      87487      90120      93661      96185      99667   \n",
       "23049      2      0       2217      61328       2877     160944     156864   \n",
       "\n",
       "       BILL_AMT6  PAY_AMT1  Y  \n",
       "3195       12245      1307  0  \n",
       "15960     154515      8000  0  \n",
       "17713     158850      5520  0  \n",
       "17897     102978      5000  0  \n",
       "23049     160066     61634  1  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# чтобы меньше исправлять\n",
    "DF_num = DF\n",
    "\n",
    "# результат\n",
    "DF_num.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7746004e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>LIMIT_BAL</th>\n",
       "      <th>SEX</th>\n",
       "      <th>EDUCATION</th>\n",
       "      <th>MARRIAGE</th>\n",
       "      <th>AGE</th>\n",
       "      <th>PAY_0</th>\n",
       "      <th>PAY_2</th>\n",
       "      <th>PAY_3</th>\n",
       "      <th>PAY_4</th>\n",
       "      <th>PAY_5</th>\n",
       "      <th>PAY_6</th>\n",
       "      <th>BILL_AMT1</th>\n",
       "      <th>BILL_AMT2</th>\n",
       "      <th>BILL_AMT3</th>\n",
       "      <th>BILL_AMT4</th>\n",
       "      <th>BILL_AMT5</th>\n",
       "      <th>BILL_AMT6</th>\n",
       "      <th>PAY_AMT1</th>\n",
       "      <th>Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>200000</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>34</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>-1</td>\n",
       "      <td>11073</td>\n",
       "      <td>9787</td>\n",
       "      <td>5535</td>\n",
       "      <td>2513</td>\n",
       "      <td>1828</td>\n",
       "      <td>3731</td>\n",
       "      <td>2306</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>120000</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>316</td>\n",
       "      <td>316</td>\n",
       "      <td>316</td>\n",
       "      <td>0</td>\n",
       "      <td>632</td>\n",
       "      <td>316</td>\n",
       "      <td>316</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>280000</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>40</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>186503</td>\n",
       "      <td>181328</td>\n",
       "      <td>180422</td>\n",
       "      <td>170410</td>\n",
       "      <td>173901</td>\n",
       "      <td>177413</td>\n",
       "      <td>8026</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>50000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>25</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-2</td>\n",
       "      <td>-2</td>\n",
       "      <td>-2</td>\n",
       "      <td>0</td>\n",
       "      <td>780</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>780</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>280000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>31</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>2</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>-1</td>\n",
       "      <td>498</td>\n",
       "      <td>9075</td>\n",
       "      <td>4641</td>\n",
       "      <td>9976</td>\n",
       "      <td>17976</td>\n",
       "      <td>9477</td>\n",
       "      <td>9075</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \\\n",
       "10     200000    2          3         2   34      0      0      2      0   \n",
       "21     120000    2          2         1   39     -1     -1     -1     -1   \n",
       "36     280000    1          2         1   40      0      0      0      0   \n",
       "38      50000    1          1         2   25      1     -1     -1     -2   \n",
       "39     280000    1          1         2   31     -1     -1      2     -1   \n",
       "\n",
       "    PAY_5  PAY_6  BILL_AMT1  BILL_AMT2  BILL_AMT3  BILL_AMT4  BILL_AMT5  \\\n",
       "10      0     -1      11073       9787       5535       2513       1828   \n",
       "21     -1     -1        316        316        316          0        632   \n",
       "36      0      0     186503     181328     180422     170410     173901   \n",
       "38     -2     -2          0        780          0          0          0   \n",
       "39      0     -1        498       9075       4641       9976      17976   \n",
       "\n",
       "    BILL_AMT6  PAY_AMT1  Y  \n",
       "10       3731      2306  0  \n",
       "21        316       316  1  \n",
       "36     177413      8026  0  \n",
       "38          0       780  1  \n",
       "39       9477      9075  0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# чтоб меньше исправлять\n",
    "DF_predict_num = DF_predict\n",
    "\n",
    "# результат\n",
    "DF_predict_num.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3a48810",
   "metadata": {},
   "source": [
    "# Модель дерева  \n",
    "\n",
    "В этом разделе построим:  \n",
    "\n",
    "* дерево классификации  \n",
    "* дерево классификации с обрезкой ветвей  \n",
    "\n",
    "\n",
    "## Дерево на всех признаках    \n",
    "\n",
    "Построим модель и выведем изображение дерева в виде текста.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "688312d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4018"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# выращиваем дерево на всех объясняющих\n",
    "X = DF_num.drop(['Y'], axis=1)\n",
    "y = DF_num['Y']\n",
    "\n",
    "# классификатор\n",
    "cls_one_tree = DecisionTreeClassifier(criterion='entropy',\n",
    "                                      random_state=my_seed)\n",
    "\n",
    "tree_full = cls_one_tree.fit(X, y)\n",
    "\n",
    "# выводим количество листьев (количество конечных узлов)\n",
    "tree_full.get_n_leaves()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "cce2b6f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# глубина дерева: количество узлов от корня до листа\n",
    "#  в самой длинной ветви\n",
    "tree_full.get_depth()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fa61de0",
   "metadata": {},
   "source": [
    "Очевидно, дерево получилось слишком большое для отображения в текстовом формате. Графическая визуализация тоже не поможет в данном случае. Посчитаем показатели точности с перекрёстной проверкой.   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a0127bc8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acc с перекрёстной проверкой \n",
      "для модели one_tree : 0.731\n"
     ]
    }
   ],
   "source": [
    "# будем сохранять точность моделей в один массив:\n",
    "score = list()\n",
    "score_models = list()\n",
    "\n",
    "# считаем точность с перекрёстной проверкой, показатель Acc\n",
    "cv = cross_val_score(estimator=cls_one_tree, X=X, y=y, cv=5,\n",
    "                     scoring='accuracy')\n",
    "\n",
    "# записываем точность\n",
    "score.append(np.around(np.mean(cv), 3))\n",
    "score_models.append('one_tree')\n",
    "\n",
    "print('Acc с перекрёстной проверкой',\n",
    "      '\\nдля модели', score_models[0], ':', score[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e66a2df5",
   "metadata": {},
   "source": [
    "## Дерево с обрезкой ветвей   \n",
    "\n",
    "Подберём оптимальное количество ветвей, которое максимизирует $Acc$, для экономии времени рассчитанный методом проверочной выборки.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2935464d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Всего значений alpha: 1672\n",
      "Энтропия листьев для первых 5 значений alpha: [0.00134371 0.00136294 0.00138216 0.00143099 0.00147982]\n"
     ]
    }
   ],
   "source": [
    "# рассчитываем параметры alpha для эффективных вариантов обрезки ветвей\n",
    "path = cls_one_tree.cost_complexity_pruning_path(X, y)\n",
    "ccp_alphas, impurities = path.ccp_alphas, path.impurities\n",
    "print('Всего значений alpha:', len(ccp_alphas))\n",
    "print('Энтропия листьев для первых 5 значений alpha:', impurities[:5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "701c413e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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fPPNN9StW/e+8kJDQ/nnP/9p0mVhb29vEufly5epVasWOp2u0PMtrv4L06lTJ7744gsURSEoKIh58+bRoEGDQve1s7PD19eXTZs2kZSUhJ+fH5D3ZWZra8vmzZuJiori008/5bnnngMKr+Pc3Nz71iuKQk5ODg4ODuoX0J07d0hLS1Ov/bvdWy9Xr16latWq1KpV676YCzuPourKzs7uvtjOnDmDq6sr9vb3pzStOi/u9/5ue/fuJSgoiLfeeosXXngBV1fXQnPMvedkNBrV5bvjs7GxUX83/vGPf9CqVSu8vLx4+eWXOXr0aLG/Nw+jQnbjzJs3j9DQ0EL/HBs8eLDaInN2dub27dvqNltbW7p27crEiRPVllS+rl27sm7dOgwGA0ajkenTpzN//vxi4+jSpQsxMTGcP38egP/3//4fr732Wolar5988glTp041ae2Vpry7z+Obb75BURQMBgMbN27k2WefVbfXq1eP9957j48//hiAn376iaFDh9K3b1/q1KnD/v37yc3NZdeuXQwZMoSsrCyqVKmCo6Mjt2/fLnL/4uJZvXq1Gs+//vUvtQWYnZ1d6C+vra0tp0+f5p133mH79u0P3I3TvHlztXsJ8pLQq6++yokTJ2jXrh3h4eFs2bKF1atXmxx36tQprl69Sr9+/UzWd+nShe+++w6AO3fuMHToUC5evFjoZ3t5efHdd9+prbawsDA6d+5cZKv+bjY2NtSsWVPz3/uNN94gOjqakydP4u3tDcDYsWPV83FxceHOnTtAXpdWYXXcvn17EhISOHbsGABnz57ll19+4emnnwYKunF2797NoEGD+PTTT+8ro0uXLvz3v//FaDRiMBgYPXq02sK922+//cZvv/0GQHh4OB06dMDFxaXIa9XV1RUbGxtiYmIAOHnyJEOHDi2y+03rmi/u9/5uMTEx/P3vf+fNN9+kTZs2REdHF3ltb968WY3twoULdO7cuchyb9++zfHjxxk/fjwvvvgif/75J5cuXdLsTnxQFbJlP2jQILp06VLotoCAAGbOnMny5cvVG1V38/HxYdOmTSxfvpyDBw+q60eOHEloaCivv/46ubm5eHh4qC3ForRo0YLZs2czduxYFEXB3t6eZcuWFfmn3928vLzUXyxzlJdv2rRpBAYG0rt3b7Kzs/Hy8uK9994z2WfAgAGsWbOG//3vf7z/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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# изображаем на графике\n",
    "plt.plot(ccp_alphas[:-1], impurities[:-1], marker=',', drawstyle=\"steps-post\")\n",
    "plt.xlabel(\"значение гиперпараметра alpha\")\n",
    "plt.ylabel(\"общая энтропия листьев дерева\")\n",
    "plt.title(\"Изменение показателя нечистоты узлов с ростом alpha\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d3276234",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Расчёты по обрезке дерева заняли 538.35 секунд\n"
     ]
    }
   ],
   "source": [
    "# обучающая и тестовая выборки, чтобы сэкономить время\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, \n",
    "                                                    random_state=my_seed)\n",
    "\n",
    "# модели\n",
    "clfs = list()\n",
    "\n",
    "# таймер\n",
    "tic = time.perf_counter()\n",
    "# цикл по значениям alpha\n",
    "for ccp_alpha in ccp_alphas:\n",
    "    clf = DecisionTreeClassifier(random_state=my_seed, ccp_alpha=ccp_alpha)\n",
    "    clf.fit(X_train, y_train)\n",
    "    clfs.append(clf)\n",
    "\n",
    "# таймер\n",
    "toc = time.perf_counter()\n",
    "print(f\"Расчёты по обрезке дерева заняли {toc - tic:0.2f} секунд\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "33434419",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Расчёты показателей точности заняли 12.66 секунд\n"
     ]
    }
   ],
   "source": [
    "# извлекаем характеристики глубины и точности\n",
    "#  таймер\n",
    "tic = time.perf_counter()\n",
    "node_counts = [clf.tree_.node_count for clf in clfs]\n",
    "train_scores = [clf.score(X_train, y_train) for clf in clfs]\n",
    "test_scores = [clf.score(X_test, y_test) for clf in clfs]\n",
    "#  таймер\n",
    "toc = time.perf_counter()\n",
    "print(f\"Расчёты показателей точности заняли {toc - tic:0.2f} секунд\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f502a713",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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iqIoyr59+5S///3vRTK/+uqriqIoypo1a5Tr168riqIoK1euVFasWKFcunRJad++vXLnzh0lLy9PCQgIUE6cOKHs2LFDOX36tKIoirJ161Zl2rRp+ud6++23lfz8fOXHH39UfH19lUWLFpWY78+vedOmTcqgQYOU/Px8JTU1VenSpYuSm5urrFy5Uv++TZ8+XYmOji70vsXFxSm//PKLoiiKkpCQoAwaNEhRFEXx9vZWsrOzS3wf2rdvr9y9e1fJz89Xunbtqty6davE11bTSBuSNqQoxm9DZtmDeumll7h06RL9+vWjcePG2NnZARAWFoadnR03b97kzTffpE6dOvz+++/88ssvODg4oNVqcXR0ZNCgQQwePBh7e3uuXLkCwNmzZ/niiy/46quvUBQFKysrAP744w9CQ0P1PxfQarV88803jBo1iujoaC5evEhaWhrvvvsu8OBii5cuXTL4WjZu3EinTp3IzMwscZuzZ89y9epVBg0aBMC9e/e4ePEiAB06dACgdevWzJ07t8TnqFevHh9//DG1atXixo0b+k88TZs21V/jy8PDg/Pnz1O/fn1WrFiBra0tGRkZha6A3LZtWxISEti+fTs+Pj7cvn27xHwPX66mwMsvv4yFhQVPPfUUTk5OpKWl8eSTTxIWFoa9vT3nzp2jVatWhfapW7cuK1euZOPGjahUKvLy8oo8b3Hvw3PPPUft2rUBePLJJ8nKyuLpp58u8bXVJNKGpA39mTHakFkWqF27duHm5saOHTsYOHCg/r4mCxYsKDQ8sXnzZhwdHZk9ezYpKSlERUWhKApDhgxhyJAhAPovAd3c3BgyZAienp4kJydz+PBhAF544QUiIiIKbQuwatUqRo4cqW/YDRo04JlnnuGbb77BysqKzZs306xZs1Jfx507d9i1axdffvklP/74Y4nbubm58cILL/DVV1+hUqn417/+hbu7Ozt37uTkyZO0bduWo0eP0rhx4xKfIzw8nN27d+Pg4EBYWBjKf6+AlZycTFZWFtbW1hw/fhx/f3+mTJnCJ598QqNGjVi6dKn+DxBA7969+fzzz7G0tKR27drcvn27xHzFOXnyJAC3bt1Co9FgZ2fH0qVL2bdvHwCDBw/WZyvw2WefMWDAAF577TU2bdrEd999V+zz/vl9KO7WBB9//HGJr60mkTYkbai4563qNmSWBapu3bpMmjQJKysrnJ2dS/yleuWVVxg/fjwJCQnY2dnRsGFDbt68Sb169YpsGxYWph+jzc7OZtq0aaVmaNCgAV26dCE+Ph4AZ2dnBg0aRGhoKPn5+Tz77LP06tWr1Oe4evUqy5cvx8Ki9LksTZs25ZVXXiE4OBitVouHh4f+NcTGxhITE4NOp2PevHklPkffvn0JCAjAycmJp556ips3bwLox9Bv3bpFz549adq0KX369OH999/nySefpH79+ty5c0f/PI0bN+aPP/5gwoQJ+uWl5fuzW7du8be//Y309HRmzpyJg4MDnp6e+Pr6UqtWLZycnPTZCvTs2ZOPP/6YL774gmeeeaZQngKP+j6U9tpqEmlD0ob+zChtqMyDg8LkFYzjl9Wfx9or26ZNm5RFixZV+POW930QNZe0oQeM1YZkmrkQQgiTJLfbEEIIYZKkByWEEMIkSYESQghhkqRACSGEMElSoIQQQpgkKVBCCCFM0v8DE1FrTcwBdnEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# изображаем на графике\n",
    "fig, ax = plt.subplots(1, 2)\n",
    "\n",
    "# график глубины дерева\n",
    "ax[0].plot(ccp_alphas, node_counts, marker=',', drawstyle=\"steps-post\")\n",
    "ax[0].set_xlabel(\"значение гиперпараметра alpha\")\n",
    "ax[0].set_ylabel(\"количество узлов\")\n",
    "ax[0].set_title(\"Сложность модели vs alpha\")\n",
    "\n",
    "# график точности\n",
    "ax[1].plot(ccp_alphas, train_scores, marker=',', label='train',\n",
    "           drawstyle=\"steps-post\")\n",
    "ax[1].plot(ccp_alphas, test_scores, marker=',', label='test',\n",
    "           drawstyle=\"steps-post\")\n",
    "ax[1].set_xlabel(\"значение гиперпараметра alpha\")\n",
    "ax[1].set_ylabel(\"Acc\")\n",
    "ax[1].set_title(\"Точность модели  vs alpha\")\n",
    "ax[1].legend()\n",
    "fig.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42254d2b",
   "metadata": {},
   "source": [
    "Находим оптимальный размер дерева по максимуму $Acc$ на тестовой выборке.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "22746203",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Оптимальное количество узлов: 23 \n",
      "соответствующая Acc на тестовой: 0.827 \n",
      "\n",
      "Acc с перекрёстной проверкой \n",
      "для модели pruned_tree : 0.731\n"
     ]
    }
   ],
   "source": [
    "# оптимальное количество узлов\n",
    "opt_nodes_num = node_counts[test_scores.index(max(test_scores))]\n",
    "\n",
    "# считаем точность с перекрёстной проверкой, показатель Acc\n",
    "cv = cross_val_score(estimator=clfs[opt_nodes_num], X=X, y=y, cv=5,\n",
    "                    scoring='accuracy')\n",
    "\n",
    "# записываем точность\n",
    "score.append(np.around(np.mean(cv), 3))\n",
    "score_models.append('pruned_tree')\n",
    "\n",
    "print('Оптимальное количество узлов:', opt_nodes_num,\n",
    "      '\\nсоответствующая Acc на тестовой:', np.around(max(test_scores), 3),\n",
    "      '\\n\\nAcc с перекрёстной проверкой',\n",
    "      '\\nдля модели', score_models[1], ':', score[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be66ffe3",
   "metadata": {},
   "source": [
    "Посмотрим на характеристики глубины и сложности построенного дерева с обрезкой ветвей."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "9d36d60b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2789"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# выводим количество листьев (количество узлов)\n",
    "clfs[opt_nodes_num].get_n_leaves()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "be182d99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "43"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# глубина дерева: количество узлов от корня до листа\n",
    "#  в самой длинной ветви\n",
    "clfs[opt_nodes_num].get_depth()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55d3d3e8",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "📚 **Пример визуализации небольшого дерева**\n",
    "\n",
    "Лучшее дерево с обрезкой по-прежнему слишком велико для визуализации. Для примера нарисуем одно из небольших деревьев с обрезкой и выведем его же в виде текста.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c7e4eba9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[19, 17, 17, 17, 17, 17, 15, 11, 11, 11, 9, 9, 9, 9, 7, 5, 5, 3, 1]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# находим деревья с количеством листьев меньше 20\n",
    "[i for i in node_counts if i < 20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "af09592a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Количество узлов: 19 \n",
      "Точность дерева на тестовой: 0.826\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABYEAAARNCAYAAAAEko/OAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOzdd3RUdd7H8fekd9J7o4USIIAUKQKCIBC6igV11bWsu67uuq5rWwi6qz6uZe0u9oKdHooKSO+9BpJASCEJ6T2ZZGaePwLRmAKhDYTP6xzO83B/v3vne8Oe48wn3/leg8VisSAiIiIiIiIiIiIirZKNtQsQERERERERERERkQtHIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVUwgsIiIiIiIiIiIi0oopBBYRERERERERERFpxRQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGF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Iyzbk876wyewt7WhS0gbHh3Tjas7+Dc4p7DMeEYPeSuqMGK21IanzVl1MJOPVidyz9CO5y0E3nYkB4AdKXnsONkp/ltzHhneohAYYGzPMPzbOPP5uiQ+XZuEscZMsJcL9w2LYtqg9tj9ZlawiIjIpcxgOR+P1RURERERkSuewWAgd/aj1i7jspJZWM4Nr6+se3jbleC1pfvwcXfizsEdrF3KeTVwZjy9Irx5+66Bp93rO+1V9FFcREQuJv3qUkRERERERC6KjIIyfj6YSUy4t7VLERERuaJoHISIiIiIiIiVZRSUs2xPOoFtnOn5m4eztSa5JVX8dXS3VhMC15jNLN933NpliIiInJZCYBERERERESvbnZrP7tR8hnUJbNUhcGsJf08x1ph5dt4ua5chIiJyWpoJLCIiIiIi54VmAoucGc0EFhGRi00zgUVERERERERERERaMYXAIiIiIiIiIiIiIq2YQmARERERERERERGRVkwhsIiIiIiIiIiIiEgrphBYREREREREREREpBVTCCwiIiIiIiIiIiLSihksFovF2kWIiIiIiMjlLzIslGPpGdYuQ+SSFxEaQkpaurXLEBGRK4hCYBERERERueRkZ2fzwgsvsHfvXqZPn87QoUOtXZJcYVasWMG///1vevXqxRNPPIGfn5+1SxIRETlrGgchIiIiIiKXjJqaGj799FMmTpxIREQE8fHxCoDFKkaMGEF8fDxBQUGMHz+e2bNnYzKZrF2WiIjIWVEnsIiIiIiIXBJ2795NXFwcbm5uxMXF0b59e2uXJAJAYmIicXFxVFZWMnPmTLp162btkkRERFpEIbCIiIiIiFhVUVERr776KitWrODvf/87EyZMwGAwWLsskXosFgvz5s3jlVdeYfTo0fzlL3/B3d3d2mWJiIicEY2DEBERERERq7BYLMyfP5/Y2FgMBgOLFy9m4sSJCoDlkmQwGJgyZQrx8fFUVVUxduxY4uPjUV+ViIhcDtQJLCIiIiIiF11ycjJxcXGUlpYyc+ZMevToYe2SRFpkx44dxMXF4ePjw4wZM4iMjLR2SSIiIk1SCCwiIiIiIhdNRUUF7733Ht988w1//OMfue2227Czs7N2WSJnpaamhs8//5z33nuPadOm8cADD+Do6GjtskRERBrQOAgREREREbkoVq1axbhx40hNTWXBggXceeedCoDlsmZnZ8fdd9/NggULSEpKYvz48axbt87aZYmIiDSgTmAREREREbmgMjMzef755zl06BDTp09n8ODB1i5J5IJYvXo1zz77LN27d+fJJ58kICDA2iWJiIgA6gQWEREREZELpLq6mo8++ohJkyYRFRXFokWLFABLqzZ06FDi4+OJiIhg4sSJfPrpp9TU1Fi7LBEREXUCi4iIiIjI+bdjxw5mzJiBn58f06dP10Oz5IqTnJzMzJkzKSkpIS4ujpiYGGuXJCIiVzCFwCIiIiIict4UFBTw8ssvs2bNGp588knGjBmDwWCwdlkiVmGxWFi0aBEvvfQSI0aM4NFHH6VNmzbWLktERK5AGgchIiIiIiLnzGw2M2fOHMaNG4ezszNLlixh7NixCoDlimYwGJgwYQKLFy/GYDAQGxvL/PnzUS+WiIhcbOoEFhERERGRc3L48GHi4uIwGo3MnDmT6Ohoa5ckcknas2cPcXFxuLq6EhcXR/v27a1dkoiIXCEUAouIiIiIyFkpLy/nrbfeYu7cuTz88MPcfPPN2NraWrsskUuayWTiyy+/5K233uLmm2/mwQcfxNnZ2dpliYhIK6dxECIiIiIi0mLLly8nNjaWEydOsGjRIm677TYFwCJnwNbWljvuuIOFCxeSlpbGuHHjWLVqlbXLEhGRVk6dwCIiIiIicsYyMjJ47rnnSElJYcaMGQwYMMDaJYlc1tatW8ezzz5LVFQUTz/9NEFBQdYuSUREWiF1AouIiIiIyGkZjUZmzZrFlClT6NGjBwsXLlQALHIeDB48mEWLFtGpUycmTZrERx99RHV1tbXLEhGRVkadwCIiIiIi0qytW7cSFxdHcHAw06dPJywszNolibRKKSkpPPvss+Tk5DBz5kx69+5t7ZJERKSVUAgsIiIiIiKNys/P56WXXmLjxo089dRTjBo1CoPBYO2yRFo1i8XC0qVLeeGFFxgyZAiPPfYYXl5e1i5LREQucxoHISIiIiIi9ZjNZr755htiY2Px9PRk8eLFXH/99QqARS4Cg8HA2LFjWbJkCc7OzowbN445c+ZgNputXZqIiFzG1AksIiIiIiJ1EhISiIuLAyAuLo7OnTtbtyCRK9z+/fuZMWMGDg4OxMXFERUVZe2SRETkMqQQWEREREREKC0t5c0332ThwoX89a9/5cYbb8TGRl8cFLkUmEwmvvnmG9544w2mTJnCQw89hIuLi7XLEhGRy4je1YmIiIiIXMEsFgvLli0jNjaW4uJiFi9ezNSpUxUAi1xCbG1tue2224iPjycnJ4fY2FiWL19u7bJEROQyok5gEREREZErVGpqKs8++yxZWVnExcXRp08fa5ckImdg48aNzJw5k8jISJ555hlCQ0OtXZKIiFzi9Ot9EREREZErjNFo5J133uGmm26if//+zJs3TwGwyGVkwIABLFy4kJiYGG644QZmzZqF0Wi0dlkiInIJUyewiIiIiMgV5FQHYdu2bXnmmWcICQmxdkkicg7S0tJ47rnnyMjIYMaMGfTr18/aJYmIyCVIIbCIiIiIyBUgNzeXF198ke3bt/P0009z3XXXWbskETlPLBYLP/30E88//zz9+/fn8ccfx8fHx9pliYjIJUTjIEREREREWjGTycTs2bMZN24cAQEBLF68WAGwSCtjMBgYNWoU8fHxeHl5MW7cOL755hvMZrO1SxMRkUuEOoFFRERERFqpffv2ERcXh6OjIzNmzCAqKsraJYnIRZCQkEBcXBwWi4WZM2fSuXNna5ckIiJWphBYRERERKSVKSkp4fXXX2fp0qX87W9/Y/LkyRgMBmuXJSIXkdls5vvvv+e1115jwoQJ/PnPf8bNzc3aZYmIiJVoHISIiIiISCthsVhYvHgxY8eOpbKykvj4eKZMmaIAWOQKZGNjw9SpU1m8eDHFxcXExsaybNky1AcmInJlUiewiIiIiEgrkJKSwsyZM8nLyyMuLo7evXtbuyQRuYRs27aNuLg4AgMDmT59OuHh4dYuSURELiJ1AouIiIiIXMaqqqp44403uPnmmxkyZAhz585VACwiDfTp04d58+bRv39/brrpJt555x2MRqO1yxIRkYtEIbCIiIiIyGVq3bp1jB8/nqSkJBYsWMDdd9+NnZ2dtcsSkUuUvb099913H3PnzmXv3r1MmDCBjRs3WrssERG5CDQOQkRERETkMpOdnc0LL7zA3r17mT59OkOHDrV2SSJyGVqxYgX//ve/6dWrF0888QR+fn7WLklERC4QdQKLiIiIiFwmampq+PTTT5k4cSIRERHEx8crABaRszZixAji4+MJCgpi/PjxzJ49G5PJZO2yRETkAlAnsIiIiIjIZWD37t3ExcXh7u7OjBkzaN++vbVLEpFWJDExkbi4OCorK5k5cybdunWzdkkiInIeKQQWEREREbmEFRUV8eqrr7JixQoef/xxxo8fj8FgsHZZItIKWSwW5s2bxyuvvMLo0aP5y1/+gru7u7XLEhGR80DjIERERERELkEWi4UFCxYQGxuLwWBg8eLFTJgwQQGwiFwwBoOBKVOmEB8fj9FoZOzYscTHx6PeMRGRy586gUVERERELjHJycnMnDmTkpISZs6cSY8ePaxdkohcgXbu3MmMGTPw8fFh+vTptG3b1toliYjIWVIILCIiIiJyiaioqOC9997jm2++4Y9//CO33XYbdnZ21i5LRK5gNTU1fPHFF7z77rtMmzaNBx54AEdHR2uXJSIiLaRxECIiIiIil4BVq1Yxbtw4UlNTWbBgAXfeeacCYBGxOjs7O+666y4WLFhAUlIS48aNY+3atdYuS0REWkidwCIiIiIiVpSZmcnzzz/PoUOHmD59OoMHD7Z2SSIiTVq9ejXPPvss3bp146mnniIgIMDaJYmIyBlQJ7CIiIiIiBVUV1fz0UcfMWnSJKKioli0aJECYBG55A0dOpT4+HgiIyOZMGECn376KTU1NdYuS0RETkOdwCIiIiIiF9mOHTuYMWMGfn5+TJ8+ncjISGuXJCLSYr9+iGVcXBwxMTHWLklERJqgEFhERERE5CIpKCjg5ZdfZs2aNTz55JOMGTMGg8Fg7bJERM6axWJh0aJFvPTSS4wYMYJHH32UNm3aWLssERH5DY2DEBERERG5wMxmM3PmzGHcuHE4OzuzZMkSxo4dqwBYRC57BoOBCRMmsHjxYgwGA7GxscyfPx/1m4mIXFrUCSwiIiIicgEdPnyYuLg4jEYjM2fOJDo62toliYhcMHv27CEuLg5XV1fi4uJo3769tUsSEREUAouIiIiIXBDl5eW89dZbzJ07l4cffpibb74ZW1tba5clInLBmUwmvvzyS95++22mTp3Kgw8+iLOzs7XLEhG5omkchIiIiIjIebZ8+XJiY2PJyckhPj6e2267TQGwiFwxbG1tueOOO1iwYAFpaWmMGzeOVatWWbssEZErmjqBRURERETOk4yMDJ577jlSUlKYMWMGAwYMsHZJIiJWt379embOnElUVBRPP/00QUFB1i5JROSKo05gEREREZFzZDQamTVrFlOmTCEmJoaFCxcqABYROWnQoEEsWrSIzp07M2nSJD788EOqq6utXZaI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E3h7e+Pj44OXlxfHjh1j5syZ5ObmcuLECebNm0dYWJi1SxURaTGFwCIiIiJyybjuuuvIyMigXbt2AOTm5lJSUoKnpyc+Pj54e3szbtw4brrpJitXKiIirc1XX33FsmXL6n7pWFRUhIeHBz4+PlgsFo4ePUpkZCRLly61dqkiIi2mEFhERERELhnvvPMOfn5+hIaG4u3tja+vL56entja2lq7NBERucKYTKZ630RJT0+noKCABx54wNqliYi0mEJgERERERERERERkVbMztoFiIiISMtFhoVwLP24tcsQueRFhAaTkpZh7TJEROQyERYeSXraMWuXIXLehIZFkJaaYu0y5BKgTmAREZHLkMFg4MSsu61dhsglz//+j9HbXREROVMGg4FXNxZauwyR8+bRAZ56LyQA2Fi7ABERERERERERERG5cBQCi4iIiIiIiIiIiLRiCoFFREREREREREREWjGFwCIiIiIiIiIiIiKtmEJgERERERERERERkVZMIbCIiIiIiIiIiIhIK6YQWERERERERERERKQVs7N2ASIiItL6fbg+lY83pjU4bm9rwMvFnu4hHtzRP5QOfq4N9izck8VLPybj7mTH/D/0xdHul99h/3tpIkv3n2BSTCCPjWzf5Os/Of8ga5Pyefjatky9KrjF9f90MIfvd2SSnFOGBWjr48JNVwVxfVf/Fl/rYvtqawZvr05h3WODzvicOz7eydG88kbX5j7QB393x/NVnoiIiPzG1nmz2D7/gwbHbezscfbwIqhjT3qN+x0+4R0b7Dmwaj5rPn4eR1cP7vjvYuwcfvlv9sr3Z3J43WK6Dr+BIb/7R5Ovv+z1v5OyYzUDpz1Kj1G3tLj+krwsts55j7S9m6iuLKdNYDgxo28jatDYFl/rYtu9dDYbv36dP3y65YzPKc7JYMt375KRsI3qygr8IjvTd8oDBHfufQErFWk5hcAiIiJy0UzoEUBMqEfd36tNFtIKKpi7M5P1Sfm8c2t3ogLc6p2zZN8JnO1tKKmsYWVCLmO6/RK8PnxtW7YeK2TB7iyu6+xLz7A2DV5zRUIua5Py6R3Whpt6B7W45jk7M3ltxRGi/F35/aBwbAzww8EcnluSyPHCSu4eGN7ia14s65Lz+d/aYy06x1hjJrWggn6Rnlzf1a/BuoeT3j6KiIhcDF2GTSYoqmfd382magqzUtm/4ntSdq5h0jPv4xvRqd45h9Yuwt7JhaqyYpK3/ESnwePq1gbd9ijp+7Zw4Oe5dOg/stGQMmnzT6TsWE1wl6voPvLmFtdcnJPBvOd+j6m6mu4jp+Ls7sXhDUtZOSuOipICYkZPa/E1L5aUnWvZ/N3bLTqnrCCHBc8/QHVlBd1H3oyzuyf7V85h0f/9idjHXic0ut8Fqlak5TQOQkRERC6absHuXN/Vv+7PuO4BPDgkkqfHdKSyxswH61Pr7T+WV86+4yXc2DsYJzsb5u/Oqrfu7mTHYyPbYwFe/DGJqhpzvfWiimr+u/IIrg62PDW6AwaDoUX1llTW8PbqFKL8XZl1ewy39g3h5j4hzJoWQ5dANz7dlE5uadVZ/SxOKa6sOafzG2MyW/h8czpPzz9IjdnSonNT8soxmS0MbOdV79/q1B8ne9vzXq+IiIg0FNChO1GDxtT96TxkAldPfYhr751OjbGSLXPeq7e/4HgK2Ul76XbdVOwcnDiwcm69dUdXd4bc9QRYLKz+6N/UGOu/h6ksLWT9F6/g4OzKtfdOb/H7JoB1n79MVWkx4x5/k75THqDbyKlMfPp9vILbsm3e+5hqqlv+g/iVqrLiczq/MWaziZ3xn/LDG49jNrXsfdm2+R9QVpBD7GNv0HfK/XQbOZVJz3yASxsf1n3+HyyWlr0PE7mQFAKLiIiI1Q3t6IOLgy270+u/sV+87wQAA9p50TfSk/2ZJSTllNXbM7i9N6O6+JFeUMmHvwmRX195lILyah6+ti2BbZxaXNfujGKMNWZiuwVgZ/PLByE7GwPXdfalxmxhb0ZJi68LsDejmGcXH+b2j3ac1flNKa6s4Xef7uR/a48xsL03nQIajthoTnJO7RiIdr4u57UuEREROT/a9rkWeydXsg7vqnf80NpFAITHDCK0W3+yk/eRl5pYb09kr2voOGA0RdlpbJs3q97a+i9epaI4n4G3PYq7b8u/PVWal03qng1EDY7Fv23XuuO2dnZcfcvDxIy5HWN5aYuvC5CVuIcV703n6yenntX5TakqK+a7p29j83dvE9FzEH6Rnc/4XLPZRNKmHwjs2IOA9tF1xx1d3ekydCKFmcc4kbzvvNYrci70fT4RERGxOoPBgI0Baky/dEuYzBZ+OHACN0dbuga6cW0nX9Ym5TN/V1aD+b9/Gd6WbccK+WZbBiO7+NHR35WNRwr48WAOg9t7E9s94Kzq6hvehs/v6oW3q32DtcKK2k4RW5sz75IpN5r48cAJ5u3OIjmnHBcHW8ZG/zLe4sZZ28gqbr6z+I2p3egd3nDsxSllVTVUmyzEjYvius5+PPT13jOuD6gL2dv6utbV7Gxvc1bdQCIiInL+GQwGDDYGzDW/dK2azSYOb1iKg4sb/u2iad9vBCk7VrP/57kN5v8Ouv1vpB/Ywu5lX9JhwPX4hkeRunsDiRuXEdlrCJ2HjD+ruo4n7ACLhfAeA+uOGSvKcHB2JSJmEBExZ/58AoDqynIOb1jKgZVzyEtLwt7JlU7XxNatf/G3iZTmZjZ7jfFPvEtIl6uaXK8qL8VUU811D/6LDlePYsELfzjj+grSj1BdWU5Au24N1vza1YbgJ44cIKBD9zO+psiFpBBYRERErC4hq4TSKhM9fzUveNPRAvLKqhkT7Y+drQ2D23vjaGfDjwdz+OPQSFwcfhlL4OFsz9+ua8/TCxN4bcURXr2xK68uT8bT2Y5/jGr6gXGn42hvS9tGOmJLq2pYtCcbe1sD3UM8GjmzviM5ZczbncUPB3IoN5roFODK46Pac11nv3r38fC1bamoNjV7rUgf52bX/dwd+er3vbE5y9A2KacMJzsbPtqQyvKEXEoqa3BztOX6rv784ZoInB00DkJERMSaThw9gLG8lKBOveqOpe3ZQHlhLlGDY7G1syOy1xDsHBxJ3LCMATf/GXunX97POLm14Zo7/8GPb/6DdZ//h9jH3mTtZ/+Hk7snQ+956qzrKsg8CoCzhxfrZ7/KoXWLMZaX4OTuRY/rb6HXuLvO6JfK+enJ7F85h8Prl1JdWYZfZGeG3P0UHa8eVe8+Bt32V6orK5q9lldwZLPrbt7+3Pp/32OwafkX5UsLar+x5urTsNnA1av2uQolucdbfF2RC0UhsIiIiFw0FUYzheW/zIKrqjGTkF3Ku6tTALijf2jd2uJ92QCM6OwLgIuDLQPaebHqcB7LE3KY0COw3rWHRvkwopMvKw7l8udv9pFZXMW/J3TGy9XhvN5DjdnCc0sOU1hRza19gvFyadglfEpmUSX/WprI7vRiXBxsGdnFl4k9Ahs8/O6UIR19zrk+uxZ0JjcmOaeMyhozmUVVPHZdOyzAmsQ85uzMJCGrlLdu6Ya9rSaKiYiIXGg1leVUlBTW/d1krCLn6AE2fvsWAL3H31W3lrCmdhREh/6jALB3ciE8ZhBHtq4kcdOPdB02qd612/W5lvb9R5K8+ScWvfggJbmZjPrz/+Hs4X3W9VaV1Y7IWv3R8xhsbBh46yPY2jtycPV8tnz/LuWFuQy+4+9Nnl+cc5yfZ8WReXgX9k6udBwwiq7XTmnw8LtT2l417KxrPcXG9uxjsVOjLewdGo4cszt5rLqq8qyvL3K+KQQWERGRi+a1lUd4beWRBscDPRyZERtF/7ZeABSWV7MhuQBPZzv6RHjW7RvZ2Y9Vh/OYvzurQQgM8NcR7dieWsTBrFKu7+rH0KhzD1V/zVhjJm7xIdYnF9A9xJ37r4lodn9mURW704txtrfhL8PbMqqLH3bNBKjFlTWYT/MgNzdH22avcS5qzBam9QvF3tbAjb2D645f19kPb5cjfL8zkyX7TjAxpuHPXkRERM6vdV+8zLovXm5w3M03iBF/eI6w7gMAqCgp5NiudTi5exIa3bduX4err+fI1pUcWDm3QQgMMPiOv5NxYBsnjuwnauAY2vW59pzqNZ986FtNdRU3P/819o61317q0H8k8/51L/tWfE+366biGdT4+6eS3MyTAbALg27/Gx0HjMbWrunYqqqsGLPZ3OQ6gIOzW7PXODcn37M10t18quNZ07TkUqIQWERERC6a2/qG0DfSEwADYG9rg5+bA8Ge9TsofjyYQ43ZQq+wNuSU/DIjN9LHGQc7Gw5nl3Ews4QuQe71zvN0sWdAOy+W7j9BbLezmwPclMLyap5ccJC9GSV0D3Hn5SldT9sRGx3kxmPXtWPeriyeX5bE/9YeY2y3ACb0CCCokQfV3fPZrnOeCXwu7GwM3No3pNG1m64K4vudmWxNKVQILCIichHEjL2dsG5X1/3d1t4BVy8/PPzq/7c6ccNSzKYagjtfRWn+ibrjXsFtsbV3JPdYAieOHMC/Xdd65zm7exIeM5DD6xbTaciEc67X7mTo22lwbF0ADGCwsaHrsEmcSN5H+v4tTYbAAe2jueZ3/2D/ijms+uBZtnz3Np2GjKfL0El4+AU32P/d9DvOeSbwubB3rB1NUWNs2O1bc7ID2MG58W9/iViDQmARERG5aCJ9nOn7q87eppwaBfHz4Tx+PpzX6J75u7MahMAXSkZhJX+bs5/0gkoGtvPi2fGdcLI//WxcR3tbJvUMYlLPIHalFTFvVxZfbc1g9pZ0+kZ4MjEmkEHtveseLjc9NoqqmuY7Wjr4u56Xe2opb5fasRrlp5lZLCIiIueHV3A7QqP7nXbfobW1oyCObF3Bka0rGt1z4Oc5DULg883Nu/Zhty5tGn4Ty6VN7Xiv6sryJs+3c3AievgNRA+/geMJO9i/cg67l3zBrvjPCO3Wn67XTiai1zXY2NS+BxvxwLOYqpv/5blveMezvZ3Tcj8ZTJcX5DZYq5sX7O3fYE3EWhQCi4iIyCUlIauU5JxyQj2d+OPQyAbrRRXV/N+PyaxIyOXP17bFzfHCvp3JKqrkz9/s5USJkUkxgfx1RLu60LYleoa1oWdYG/LKjCzcncXCPdk8tSCBMC8nvvp9bYdKjzN4yNyFtDu9iJd+TOa6zr7cPTC83lpKXu2HthDPhh3MIiIiYh05Rw+Sl5aER0AYA27+c4P1ypJCVn/8PEmbfmLArX/F0eXCdab6t4sGoCCj4eiv4hPpwC/B6ekEd+5NcOfelBfmcmDVfA6ums8PbzxOm8Bwbv2/7wEIioo5T5WfHc+gSOydXDlx9ECDtRNH9gPg367bxS5LpEkKgUVEROSScqoLeFLPwCYflPbDgRx2pRfzw/4cbugddMFqqTaZeWpBAidKjEzrF8KDQyLP+Zo+rg7cPTCcO64OY11SHj8ezDn3Qs+TCG8XMosqWbAnmxt6BeHhXPvQuxqzhffXp2IAxkaro0VERORSkXCyCzh6+JQmH5R2eOMyMhN2kLh+Cd1GTr1gtQRF9cTDP5TD65fQY/S0uhEO1VWV7F3+be3D6noMaNE1XTx96TPpXnqPv4uUHWtI3LjsQpR+Vmzt7GjXdziH1y0m99ihugfYVZWVkLBmIV7BbS9497VISygEFhERkUuGscbM8oRcHOxsmg0bp14VzK70YhbsybqgIfDivdkcPlGGj6s9bX1c+OHAiQZ7ugV7nFV3rJ2NgWFRvgyL8j0fpZ6VrSmF5Jcb6RvhiberA54u9vxhSCRv/HyU+2bvYWKPAGxtDPx0MJeE7FLuHhB20UZwiIiISPNM1UaSNv2Irb0jnQaPa3Jfj1G3kJmwgwOr5l3QENhgY8Ow3z/DklceYd5z99BtxE3YO7mQsGYRRdlpDPv9P896Rq6NbW3g2q7v8PNc9ZlL27eZiqJ8Qrv1qxt50XfK/RzbtZb4lx6ix+jbcHByZf/KOVQUFzD8/ri6B8SJXAoUAouIiMglY21SHiWVNYyJ9q/rQm3M4A7eBLdx4khuObvTi4kJvTBjFLYdKwIgr6yafy1NbHTPU6M7XLYjEj7dlMau9GLemNoNb9famb9TrwomwMORr7dl8OGGNAxAez9XZsRGMbKLn3ULFhERkTpHd6ymqqyYqMGxOLk1/dDYyF5D8PALIT89mczDuwiK6nnBagru3JtJ//yQbfNmsXvZl5hNNfiEdWTMX18lImbQBXvdi2HHoo/JTNjB+CferQuB3bwDmPTMB2z+9i12Lf4MAJ+ITgy56wmCOvWyZrkiDRgsFovF2kWIiIhIyxgMBk7MutvaZYhc8vzv/xi93RURkTNlMBh4dWOhtcsQOW8eHeCp90ICgI21CxARERERERERERGRC0fjIEREROSKUVVtotRoOqO9NgYDXi5Nj6QQERERac1qjJUYy0vPaK/BxhZnD68LXJGInAuFwCIiInLFWHEol+eXJZ3R3kAPR76/v88FrkhERETk0pS0eTmrPnj2jPa6+QZx+ysLLnBFInIuFAKLiIjIFaNfpBev3RR9Rnsd7TQ1S0RERK5cYd2vZtzjb53RXlt7xwtcjYicK4XAIiIicsXwdXPA183B2mWIiIiIXPJcPX1x9fS1dhkicp6oxUVERERERERERESkFVMnsIiIiFxWBr+8np6hHrx1S/eLem5LlRtNfLopjZUJueSVVxPSxokbewcxMSbwjM4vrarh4w1prEnKI6fEiKujLb3C2nDfoHAifFwAyCyq5Kb3tzd7nd/ONt54pIDPNqdxOLsMGwN0DXLn94PC6RHicfY3KyIiIpeV937Xj6DOvZn45HsX9dyWqq4sZ/vCj0je/BPlRfl4+IfQfeTNdL128lldb/O3b7Nz8aeMf+JdQrpcVW+tMPMY2+a9T8bBbVSVFePi6Utk76H0nfIAji5u5+N2RKxKIbCIiIhcVv45tiPeLmc30uFczm0Js8XCUwsOsv1YERNiAojyd2NtUj7/+SmZ3FIjvx8U3uz5NWYLf/v+AAcySxgd7U90kBtZxVXM25XFlpRC3rutB+18XfB0tuefYzs2eo0fD+SwOaWQoR196o6tPpzHMwsT8Hd3rK3BYuH7nZk8/M0+Xrspml5hbc7rz0FEREQuTcPvn4lzG++Lfm5LWMxmlr3xOBkHttJ12CR8IzpzdMdq1nzyAmWFOfSdfH+Lrnc8YQe7lnze6FppXjbznvs9ZpOJ6BE34OEXTHbyfvYv/47jB7czefqH2Ds6n4/bErEahcAiIiJyWbm+q79Vzm2JFQm5bDtWxB+HRnJb3xAAJvQI4B/zDvL55nRiu/kT2MapyfPj92azP7OEPw2N5NaT5wMMi/LlgS/38O7qFP5zQ1ecHWwbvafEE2XsTEsiJtSDB4dG1h3/YH0qDnY2vH1rdwI9HOuuefvHO3hvzTH+N63HefoJiIiIyKUsatAYq5zbEkmbfyJj/xauvvlheo69HYAuwyax7L9/Y+eiT+h8zXjcfYPO6FpVZSWsnBWHjZ0dpmpjg/VN37yJsbyUSf/8kID2tQ8R7nrtFHzDo1g/+xX2r/ienmPvOH83J2IFmgksIiIicp4t238Ce1sDU3r+MvrBYDBwa98QaswWfkrIbfb8rSmFAEyMCah3vHOgG5E+zuxKL2ryXLPFwgs/JALw5PUdsLMx1K2lF1bQ1selLgAGCPZ0ItLXhcQTpWd8fyIiIiIX2uH1S7Cxsyd6xI11xwwGAzFjbsdsqiFp049nfK21n/4fFouZrtdOabBmsVhIP7AF34iougD4lFOB9/GEHWd5FyKXDnUCi4iIyCVh3/FiPtqQxoHMEgD6R3oxtU8wD8zew90DwupGKPx2ru9DX++lqKKG6bEdeXfNMfZmFGMwGOgZ6sEfhkTSztel7jXOZCbwh+tT+XhjWrO1jon25+kxjY9hADiQVUo7Xxec7G3rHe8SWDtP7mBWSbPXf+y6dvzu6lBcHBq+VSuqqMH2V8Huby3Zd4LD2WXcPSCMUK/6X1sM93Ymq7iSqhozjna1vQDGGjM5JUZ83Rwbu5yIiIhcRrKS9rBt3vucOLIfgLDuA+hx/a3Me/Yerpp0b90Ihd/O9V3wwh+oLClkxAPPsunbN8lO2gsYCOrUk6unPoR3aPu61ziTmcBb581i+/wPmq01anAsw++b0eT6iSP78Qltj71j/W9P+bXtWrd+Jg6vX0LSlp8Y9/c3yTy8q8G6wWBgyoxPMJuqG6xVlBQAYGNr22BN5HKjEFhERESsbkdqEY/N2Y+7kx239AnByd6GpftO8PjcA2d0fn65kT9/s4/B7b15aFhbknPKmL87i8QTZXx3f5963bCnMzTKh1Cvpkc1AIR4Nr1eWW2ipLIG/9CGD1pzsrfFzdGWrKKqZq/v5eqAl2vD2cU/Hcwht9TINR0an8NXY7bw0YZUPJ3tmNYvpMH6I9e24x/zDvDcksPcMzAcA/DRhlQKyqv545jIZmsSERGRS1vGwe0seeURHF3ciRk9DTtHJw6tXczSV/96RudXFOWx8MU/ENlrCANueYS8tCQOrJxDXmoi016Zj43tmUdI7a66ljb+Yc3u8Qho+F7llOqqSqrKinHt1KvBmr2jEw4u7pTkZp62juKcDNZ9/h96jLqF0Oh+jYbAAB5+wY0e3730SwCCu/RpdF3kcqIQWERERKzu1RXJ2NoYeP/2GPzdaztSJ8cE8sCXtV2+p1NUUVNv/i5AtcnCor3Z7Ewtom+k5xnX0sHPlQ5+ri2+h1NKq0wAONs33jHiZG9LRbWpxddNySvntRVHsLMxcPeAxj9UrTyUy4kSI/cOCm/QhQzQLdidm68K5pNN6aw6nFd3/A/XRDAm+uLMSxYREZELY91nL2Fja8eUuE9w864dKRU9/AbmPfd7KkubHiV1SmVpUb35uwCmGiMJqxeQcXA7Yd36n3EtPuEd8Qlv+ltTp2OsqB1T1dTD2OwcnKiuqmj2GmaziZX/i8PVO4B+N/6xxTUkbfqRg6vn4+YbRJehE1t8vsilRjOBRURExKqO5JSRklfB6Gj/ugAYwNHeltsa6WZtyqgufvX+3unk6IW8soYP/2hOZbWJwvLqZv+UG5sLcS0AGJpoPjZQ+7XDlkjKKeORb/dRXFnDn69tS1SAW6P75u3MxNHOhht6Nf6QlCfmH+STTen0jfBk+tiOzIiN4poO3ry39hhv/ny0RTWJiIjIpSM/PZmC40eJGjS2LgCG2rC0JQ806zhwdL2/+7ftAtR2CbdEdVUlFSWFzf6prixv+gKW2vdTTb2hMhgMp30/tWPhx5w4eoARDzyLnUPLxl4lblzGylkzsLN3ZNSfXmgyjBa5nKgTWERERKwqtaC2iyPcu+Gb67Y+Z/6G29vVvt7fHWxrPxiYT32IOEOzt2Sc00zgUx3AldXmRtcra8z4ujUc9dCUzUcLmL7oEGVGE38aGtlkwJtbamTf8RKGRfng7tTwLd7WY4VsSSlkQFsv/nND17rjI7v48eIPiXyz/Tj9Ij3p39brjGsTERGRS0Nh5jEAPAMjGqx5Bbc74+u4eNQfOWVjV/uexWxu/H1NU3Yt+eycZgLbO9W+B6wxVja6XmOsxNXLr9E1gOzkfexY+CE9Rk/D1dufipLCetczVpRSUVKIk6sHBpv6/ZE7Fn3CljnvYu/ozJi/vop/u66/vbzIZUkhsIiIiFiVyVwb0trbntsXlGxa2F3blNHR/vRoZJ7vr/k2Mq/3FFdHOzyc7MgtbdiBXDcv2P3MulEW783mpZ+SAXjy+g7Edg9ocu+65HwswHVdGv9AlHiiDICx3RqOfRjfI5D4vSfYklKoEFhEROQyZDbXfkvJ1t7+NDub99tA9Gx1GhRLUFTPZve4ePo2uebg7IajqwflBTkN1urmBXs3Pcoqdc8GzCYTuxZ/xq7FnzVY/+H1vwNw28vz6+YBm80m1n76EgdXzcPZw5sxj76Kf1sFwNJ6KAQWERERqwr1qu30SM1v+JXA1PzGuz8upBBPp2Yf/HYmOge6sSe9mGqTuV64fSCzdr5d16DGxzn8WvzebF78IQlnexv+NaHzacPZXWlF2Bigb0SbRteb64y2nDxmamHXtIiIiFwa2gTUPi+g4HhKg7XCrGMXuRrw8A/Bw//Mx3o1xr9dVzIP7cJUU42t3S/h9okj+0+uRzd5blMh9OF1izm8YSkDbnkEn/COuLTxAWrfC6364DkOr19Cm8BwYh97HQ+/c6tf5FKjmcAiIiJiVVH+roR5OfHTwVzyfzW/t8Zk5uttGVas7OyN7OJHZY2Z+buz6o5ZLBa+3paBva2B6zo3/fVFgH3Hi/nPT8m4ONjy+tRuZ9Sdm5BdSriXMy4Ojf+Ov39bL2wNMG9XVl339Slzd9bW2b8FD9ATERGRS4dvRCfaBIaTtOlHyn81v9dUU8OeZV9asbKz1+Hq66kxVnJg5dy6YxaLhT3LZmNjZ0/Hq69v8lwP/xBCo/s1+ON+Mpj2jexMaHS/ulnBu5d+weH1S/AObc+kp99XACytkjqBRURExKoMBgOPXteex+Yc4J7PdzMpJhAXB1t+PJDD0bzyk3usXGQLXd/Vj4V7snjr56NkFFTS3s+F1Yn5bDpawH2Dwwnw+GUcREZhJfuOFxPi6US34NoxFG/8fBST2cLVbT1JK6ggraDh06+v7/rLVyBNZgvHCyvpG+HZZE1hXs7cNSCMDzek8cDsPYzq6ofBAGsT89mRVsR1nX0Z0M67yfNFRETk0mUwGLjmzsdZ8sojfD/9TqKHT8HeyYXEjcsoyDhSu4fL6w1V1MAxHFw1nw1f/ZeiE2n4hHbgyPZVpO3ZQN8b/oCbzy9jsopPZJCVuAePgBACO/Ro0etUlhay7eT84rZ9riVt76YGe5zbeBPWrf+53ZCIlSkEFhEREavrG+HJazdG89GGVL7YnI6drYGB7by5oXcQ/16aeM7zgi82G4OBl6d05f31qfx8KJeFe02EeTrxxPUdGPebub6704t4flkSY6L96RbsQbnRVDc2YuWhPFYeavxp3L8OgYsrqjFbwK2RB8L92t0Dw4n0ceHb7ceZtfYYZouFcG8X/jq8HZN7BZ7jXYuIiIg1hUb3I/bvb7Ft3ix2xn+KjZ0dETGD6XbdVH5+fya29mf+YNpLgcHGhrF/e40tc/7Hka0rOLhqAW0Cwhh6z9N0GTqx3t7jh3ay6oNniRoc2+IQODtpLzVVtb9wb+phdkGdeysElsuewWLR8DcREZHLjcFg4MSsu61dxnlhsVjIL6/Gp5GHra1IyGFG/GGeGt2Bsd2afiiaSFP87/8Yvd0VEZEzZTAYeHVjobXLaDGLxUJFUV6jD1tL2vwTy995mmH3TqfzNeOsUJ1Y06MDPPVeSADNBBYREZFLwNT3t/PwN/saHP/hQO0TobsFu1/skkREREQuK1/+fTILX3ywwfHEDUsBCOzQ/WKXJCKXEI2DEBEREasyGAyM7ebPvF1ZPDHvIFe39cRkhnXJ+Ww9VsiUnoGEe7tYu0wRERGRS5bBYKDTNePZv+J7lv33McJ6DMBiMpGycy3p+zcTPeImPIMirF2miFiRQmARERGxukeGtyPC25kl+07wzppjAER4O/OPUe0Z30OzakVEREROZ9C0R/EMiuDQ2ng2ffMWAF7BkQy9+ym6DJtk3eJExOoUAouIiIjV2dkYuLF3MDf2DrZ2KSIiIiKXJRtbO7qPvJnuI2+2dikicgnSTGARERERERERERGRVkwhsIiIiIiIiIiIiEgrpnEQIiIi0qot2ZfN88uSeGp0B8Z2C7B2Oeekxmzh/i92097PlafHdKy3Nvjl9ac9f91jgxo9vvpwHk8vTOC7+64iqI1Tw/OS8vhiSwZHc8uxtzXQJ8KT+wZHEOJZf6+xxsxnm9L48WAuJ0qqaONsxzUdfLj/mgg8nPS2U0RE5HKVsDaeVR88y7B7p9P5mnHWLuecmE01zJ15N95hHRh+34wG6yV5WWz9/l1S926kxliFV3Bbuo+8mahBYxrszU7ay9Z5s8hO2ofZVINPeEd6j7+byF7X1O3ZOm8W2+d/0GxNV026l76T7z/3mxNpht6Ni4iIiFwGTGYL/156mMMnymjv59pg/Z9jOzZyFmxJKeSHAzkM7ejT6PqBzBKeX5bY5OvG783mxR+S6Bzgxn2DwymtMvH9jkw2Hd3FB7fHEOrlXLd3Rvwh1iblM7CdF7f1DSYpp5xFe7LYe7yYWdNicLTTl9BERETEesxmEyvfn0nusUN4h3VosF6Sm8ncZ++moriAToPH4de2C5kJO1g5awa5xxIYeNtf6/ZmJ+9jwQt/wN7RmZ5jb8fe2ZWDP89j2X//xvAHniVq4GgA2l11LW38wxqtZcOXr2KqNhIRM/jC3bTISQqBRURERC5xuaVVPLckke2pRU3uub6rf4NjOSVVvPHzUcK8nHh6TMMPOov3ZvPaiiNU1pgbvWZVjZk3fj5KhLcz79zaHYeTIe6g9l7c/dluPt6Yxj/HRgGQkFXC2qR8BrT14qUpXeuu4e/uwKx1qSzdf4JJMYEtum8RERGR86WsIIeVs2aQcWBbk3s2fvU6FUX5DLnrCbpeOwWAbiNuxM03kN1LviCy91CCO/cGYPuCDzHXVBP79Pv4t6t979NpUCxfP3ETm755g44DrsdgMOAT3hGf8Ia/rN/07VsYy0sZcteTdeeLXEhqxxARERG5hK1OzOPWD3ew93gJd/QPbdG5r644QlFFDU9c3wEXh/q/+//Dl3t44YckOvq70i/Ss9Hzs4oqifJ3ZXLPwLoAGKCjvxttnO04lFVadyytoBKAq9t51bvGoPbeACSeKEVERETEGo5u+5mv/nEjWYl76TX+rkb3mGqqObZ7PR4BYXQZNrneWu9xdwNw4Oe5dceKstNwcvesF+A6uroTGBVDeWEuFUV5TdaTk5LA7iVfENK1L12vndzkPpHzSZ3AIiIiclrlRhPvrklh89FCckqrcHWwo2eoB3cNDKPDr0YTVJvMfLf9OCsP5XEsv5xqkwVvV3v6t/XivkHheLs6ALAjtYiHv93HzHGdOJJbxtL9JyisqKGdjwt/GhZJl0A33l+XyvKEHMqNJjr6u/HQsEi6BrkDkFlUyU3vb+eBayJwsDXw3Y5M8surCfV0YkrPQCb1DDrtPe3NKOazTensPV6MscZMmLczE3oEMqVnIAaDoW7fgcwS3l+XSlJOGWVVNQR4ODIsypffXR2Kk71tk9c/dY/NCfRw5Pv7+zS750hOGX0iPHlwSCT2tgY+35x+2nsD2J5ayNqkfEZ39SMmtE2D9cyiSh4d0Y5JPQN5YVlSo9eI8HHhrVu6N3puUUUNnQPcftnrXTsWIjW/ot7e9MLacNjPzfGM6hYREWmtqivL2fTtW6Tt2UhpwQkcnN0I7tSLqyb+vl6nqKmmmr0/fk3yluUUZh7DVG3EpY0PYd0H0PeGB3BpUzviKePgdha9+CDX/fHf5Kcnc2hdPJUlRXiHtmPALY/g17YrW+e+R9LGH6iuqsAnvCMDbvkLAe2jASjOOc6Xj02i/01/wsbOnr0/fUNFUT5tAkKJHnEj0cNvOO09ZSXuYceij8lK3IOpugrPwHC6DJtM9Igb672fyk7ez9Y575GXloixohQ3n0Da9RlO7wn3YO/Y8HkEp5y6x+a4+QZx+ysLmt2Tl55MaHRfrr75z9jY2rNz0ScN9lSWFGKqrsInrEO92qE23HVy9yLn6MG6Y15BkRzbtY6K4nycPbzrjhefSMfOwRFHt4bvv07Z8OVrGGxtuebOx5utW+R8UggsIiIip/XPhQnsTC/mxl5BhHs7k11cxXc7jrPlWCFf3tML35MB3z8XHmJ9cj5juvkzvkcARpOZLUcLWbQnm+ziKl69Mbredd9efRRXBztu6xtCSWUNX2zJ4Mn5B+ng54rJYuHO/mEUVlTz5dYM/jHvIF//vjeujr+8fVm4O4uC8mpu6B2Ej6sDPx7I4eXlR8gsruLBIZFN3s/Ph3KJW3yYME8nbu8XipO9DZuOFvLaiiMcyirlqZMPXUsrqOCv3+3Hz82Baf1CcHGwZXtqEZ9vTictv4J/Tezc5GtE+jg3Oaf3FOdmQuRTbu8fir1tbRduZlHlafef8r+1x7C3NXD/NRGNrn9/f5+6656pEyVVHMwq5X9rj+FgZ8PdA8Pr1qIC3JjSM5CFe7Jo6+PM1W29SMmv4PWVR/B1c2B8j8v7oXwiIiLn6se3nuR4wg66jbwJz8AISvOz2fvjN6Tt28wtL36Lq5cfAD+99SQpu9bSafA4ugydhKnaSNq+TRxcPZ/S/CxiH3uj3nU3ff0G9s6u9Bx7B1VlJexa/CnLXn8Mn7COWMxmeo+/m4rSQnYv+Zxl//0bt770PQ7Ov/wi98DPc6koLqDbyKm4ePqSuGEpaz/9P0pyM7l66kNN3k/y1hWsePcZ2gSE02vcndg5OJG6ZyPrPv8POUcPcu190wEozEpl8X8ewtXLn56xd2Dv5MrxA9vYGf8JRVmpjPrzi02+hldwJMPvn9nsz9XeybnZdYBe436HrZ09UBt+N3cdY0VZgzWz2YSxvIQa4y/vxfrd+CDZR/bx49tPMfCWR7B3dmXfT9+Ql5pI3ykP1L3eb6Xu2UjmoZ10HX4DnkGNv08TuRAUAouIiEizCsqr2ZxSyOSegfxxaGTd8Y7+rsxae4xD2WX4ujmSeKKMdcn53NQ7iEeGt6vbd1PvYO77YjdbUgopqazB3emXtx8ms4X/TeteN6qgtMrEN9uPU15t4oPbY7A52YVhNJmZvSWDg1ml9InwrDs/q7iKd2/rTrdgDwAm9wzkD1/u4eutGYzvHlDvoWWnVBhN/OenZKL8XXnn1u51QeiNvYN5feURvtuRyXVdfOkX6cXaxDzKjCb+O6YjXU52IU/oEYitwUBGYSXGGnO9MQm/5u3q0Oic3pZqaVALtV3OBzJLGdfdH3/3xjtwW3rdymoTU/73ywy9+waH0zXIrd6em/sEk3iijJeXH6k75ulszxtTo/E52QUuIiJyJaooLiBt70aiR9zIgJsfrjvuGx7Flu/fJfdYAq5efuSmHiZl5xq6j7qFQdMerdvXfdTNzJ15N2l7N1FVVoKjq3vdmtlUw+R/foiDc+23s4wVpexZ9iXVlRXcEPcJBpva/+abqo3sWvwZJ44cIDS6X935JXlZTHrmfQI79AAgevgNzH/u9+xeOpsuQyfSJqDhQ82qqypY8/EL+IR3YtIz79cFnt1H3sz62a+y98ev6XD1KMK6X03KjtUYK8oY9/iMutEJXYdNwmBjQ/GJdEzVRmztG3+f4NLGh6hBY87qZ/5rTQWyv+bg7IZ3aAeyE3dTkpuJu+8v3yxL2b4as6kGi/mX5yh4hbTjqvH3sOGr15gT97u6412H38BVE3/f5OvsWfYlNra29Bx7x1nejcjZUQgsIiIizXJ1sMXVwZafD+XSwc+VwR288XF1YEhHH4Z09Knb19HflR8f7l8X3J5SUGbE7WT3bpnRVC8EvrqtV71ZtZE+taHtsI4+9a4TdjLMzSk11rv2gHZedQEw1Aabt/UNZUb8IdYl53NLn5AG97P1WCHFlTVMi/KhrMoEmOrWRnT247sdmaxOzKNfpBd+JwPUd9cc446rQ4kJ8cDBzobpsVGn/bnVmMyUVpma3WNjY8DD6fy/HZu7KxOA2/o2vP+zZTJbmB4bhY0BfjqYw/vrUknOKePZ8bXd0Edzy/nj13upqjZxW98QugW7k1Ni5OttGfzx67383+QujY6lEBERuRI4OLvi4OxK8pbl+IR1JLLXNbh4+tL2qmG0vWpY3T7f8Cjuee9nDDb1vy1UUZxf171rrCyrFwKHxwysC4ChdkwBQLu+19YFwEBdmFtWkFPv2hExg+oCYKgNTGPG3sHyd54mZecaYkZPa3A/6fs2U1VWTLu+wxt0znboP5K9P37N0e0/E9b9aly9an8pvunbN+k9/m6Conpia+/AiD88e9qfm6mmBmNF888VsLGxwdHVo9k9Z6rv5Pv44c1/sPjlRxh421/xDIog6/Au1s9+FUdXj3qdwGs+eZGDq+bhF9mZ6BE3Ye/sQuruDRz4eS5VpUWM+MOz2NjWf59XmJVK+v7NdOg/Cg+/4PNSs8iZUggsIiIizXKws+GJ6zvw4g9J/OenZP7zUzLtfF24uq0XY7v5E+njUrfX3taG5Qm5bE0pJL2wgsyiKgrKqzkV51oslnrX9v5Nd6itTe3O33aNnjzc4Pz2v5pHfErEySA5o7Dx0QlpBbUza99dc4x31xxrdE9mURUA13byZfPRApYdyGFHWhGOdjbEhHowuL03Y6L9cXZoepzDnoyS8zITuKWqTWY2JBfQJdCNcG+X059whlwd7RjVpfZrqtd19uPpBQmsPJTHpJgieoe34fPN6ZRU1jBzXCdGdPatO29EZ1/u/GQnzy5O5Jt7e2N3Fp3NIiIilztbeweG/v4ZVn/4L9Z88gJrPnkB79D2hPUYQOdrxuMV3Lbe3qRNP5K+bzNFJ9IpyTlORXE+nPwF+a+7UQGc2/jU+7vhZPDo0sa3/vGTwXKD92OhHRrUeypILj6R0ej9FGalArD527fY/O1bje4pya39pXT7fiNI27uRw+uXcPzgduwcHAmM6klk76F0GhyLvWPT4xyyEnefl5nAZ6ptn2sZ9vt/suGr/7LklUcAcHT14OqpD5G8ZQX5GclA7f0fXD0fn/COTPrnh3Wdxu37jsDDL5itc/9HSNc+dL12Sr3rH932MwAdz0N3s0hLKQQWERGR07q2ky/923qx8UgBW1IK2JFWxJdbM/hmWwYzxnVieCdfyqpqeOTb/RzKLiUm1IOuQe7EdgugS6Ab32w/zg8Hchpc187G0Mir1X3GOa3GzjeZLc1e++Qy9w4KJzrYvdE97ic7l+1sDDwzNoq7B4axNimf7ceK2J1RzJaUQr7edpxZ03rg6dL41ws7+Lvy2k3Rja6d4tjEKIlzsSO1iDKjies6+55+8zkY1cWP1Yl5HMoupXd4GxJPlOFsb8PwTvU/iHq62DOkow/zd2dxLL+i0eBeRETkStC+7wjCuw/g2O71pO/bRMbB7exe8gV7ln3FdQ8+R/t+12GsKGXR//2JnJQEgjr1wr9dNJ2vGY9/u67sXvYliRuWNrjub7tN65zhG6rGzjebTc1f+2SQ3HfKAwR0aPgQWQBHF/e6awy/P46rJt5Lys7VZBzYRuahXaTv28yepbOZPONjnN09G72Gb3hHxj3eeMh8iq39+X34bOch4+lw9Sjy0hIxGGzwCeuArb0D2xd+RBv/UADy0hLBYqHjgDENRk10GTaJrXP/R9q+zQ1C4JQdq3Fya0NodP/zWrPImVAILCIiIs0qN5pIzikjqI0jIzr71nV57kor4pHv9jN7SzrDO/ny3Y5MErJLeWxkeybFBNa7Rl6ZsbFLn7PUk129v3Ysv/ZYuHfjXSXBbWo/KDja2dD3V/OFAYorqtmWWkSAe20nclZxFekFFfSJ8OSWPiHc0ieEapOZt1alMGdnJisScrmhd9BvXwIADye7Bte/GHalFwPQL9LrnK/144ETvLvmGA8Ni2REZ796a+XG2g+Hp4Jse7vaD5pmC9j+5jOn+eQHxVMBvYiIyJWmurKcvLQk3H2D6NB/JB36jwTgeMIO4l/6E7sWf0b7ftex98dvyDl6kCF3PdEgQCwvyrsgtRWd7Or9tcLMFAA8A8MbrAG4nxxlYOfgVG++MEBlaREZB7bi5l37UNiSvCyKslIJje5HzOhpxIyehqmmmo1f/Zd9y78jedOPdBs5tdHXcXT1aHD9Cyl19waqyovpOGA0Ae271R0vOH6U0rwsOg2OBcDWrva9osXScPTXqU5ri6n+Wo2xkhNHD9L2qmHY2imOk4tP38cTERGRZh3JLefBr/byycb0esejAtxwsDXUjXAoqqgGoL1v/REE+44XsyutNpg83yHg2sQ80n8VBBtrzHy1NQMHO5t684p/rW+kF872Nny34zjFlTX11j5Yn8r0RYfqgtTPN6fxl+/2cyCzpG6Pva0NnQJqu1kvxckGCVmlONvb1I3FOBft/FzJLTXy9bbj1Pzq366quvYBfnY2Bga2qw2bB7b1oqLazOJ92fWukVdmZE1iHj6u9rRTF7CIiFyh8tOTmf+ve9m+8KN6x/0iO2Nj51A3wqGytAhoOKIh6//Zu8vwKM62jeP/jbu7QwIEgru7u0ORthRKW0rtoUrLW4HSUqPeQkspxaG4u7snBEsChDhx1919PwRClwhBJwnX7zieD5m5Z/acJX0ye+091x0aSMyl0wBoNWWvOXC/rp3eS2pcRNHP6vw8zm5ehL6hsU6/4v/yrNsCQxMzArcvJTczTWffidVz2fHLVKIvnwHgzIb5bPxqMnFhwUVj9A0McfCpDYBKv/T2Wk9a2PGd7J77qU4bDHVBAUeX/4SBkTF1Og8BwLVWIwxNzLm0fwP5ubqTEoJ3rgTAo15Lne0J4VfQatQ4Vqv9mK9CiJLJVw9CCCGEKFNdN0ua+9iw9lwsmXkFNPCwJq9Aw9YLN8nJ1xQtvtbWz45/T8fw2eYrDGroioWxPhdjM9gWfBN9PRUFGu09F0q7byoVLy8JZHAjVyyMDdgSfJOQm5m81bl6sb7Ct1mZGPBm5+p8uS2U5/4+Q7/6ztiZG3Hyegp7QxJp6GFFzzqFC5gMb+zGzosJvLv6AgMauOBqbUJUSg6rz8TgZGlE57tmx1YEN5KycbI0LrZA34PwczRnRFM3lp2M5tWlQXSv7Uh2vppN5+OISM7hzc7VcLE2AWBUc3cOhiXzzY4wzkenU8/NkviMPNadiyU9V80XA/xLbdEhhBBCVHXOfvXwqNuCC7tXkZedgVutRhTk53Hl4GYK8nJo0HMUAD6N2hG0Yzm75vwfAZ2HYmRmQfzVC1w5vBk9fX006gLysspeKO3+qVgzfTx1uwzDyMyCywc3kXjjCm3HvI2ZTcntpYzNrWgzegp7/5rBig9HUbvDAMxs7Ik8f5yrJ3fj6t+YWm16A1Cv+zOEHt3OltlvUafTYCwdXEm7GUXwrpWY2zkVzYquCBr0Gk3YiV1s+OpV6nYZhoGRMSFHthEbGkjHFz7E/Nb7YWxmQduxU9jz53T+/b+x+Lfvj5GJGZEXTnDt5B5cazWidoeBOudOiSlci8LSoeSnyIR43KQILIQQQoh7mtG/FkuOR7H7SiIHQpLQ11NRy9mCrwbXplV1OwCaeNnwSd9aLD4eyfzDNzDU18PFypgJbb3xsTPl3TUXORGegr+LxSPL1aWWA9UczFhxKpqMnAL8nMz5YqA/7fxKngV8W596zrhYG7P4eBQrTkWTp9biamXM+NaejGzqjtGtFgfe9mb8PLIuC45EsiX4JslZ+VibGtKplj0vtPbCyqTi3UqlZOfjZ/HoZtxO7lgNH3szVp+J4ed91zDU16O2iwVvdamu03LCzMiAX5+py4Kjkey5nMj2C/GYGulT392S51p6Use15P7LQgghxNOi++QvObdlEWHHd3L91D709PVx8PGn11vf4d2gDQDudZrR9ZUZnN30DyfX/oG+gSGWDq40G/wytm4+bJn9PyLOH3uks0l9W3TFzt2XwG1LyctKx96rBj3e+JpqjTuUeZx/+35YOrhwdvNCArcvQ52fh6WDK00HTaRBr9HoGxZ+IW/r5kP/D+Zwev1fXD64key0ZEwsbKjerAtNB72IsbnVI7uWh2Xn4Uv/D37jxKo5nNn4NxqNBgfvmvR5+0c86+r28a3Vti8Wds6c3riAMxvmU5Cfh5WjO80Gv0TD3mOLtXzITk8GCgvIQihBpb17WUghhBBCVHgqlYqbc8cpHUMxMak5DPvjFL0CnPiwVw2l44gKzGni/GKroAshhBClUalUfHckRekYT0RafDRL3h5IzbZ96Pzix0rHEY/J/1rZyL2QAKQnsBBCCCGEEEIIIYQQQlRpUgQWQgghhBBCCCGEEEKIKkyKwEIIIYQQQgghhBBCCFGFVbzVTIQQQggh7sHV2oSDb7dROoYQQgghRKVl5ejGywuOKx1DCPGEyExgIYQQQgghhBBCCCGEqMJkJrAQQgghyuX0jVReX3Geca08Gd/GS+k49y0mNYdhf5wq+rljDXtmDPAvNu5CTDqvLAlk9rC6NPayLvOcCRm5PLfgLNXszfh5ZD2dfRqtljVnYtkQFMeN5Gz0VVDL2YLRzT1oVd1WZ2xGbgF/HLzB/pBEkrPycbAwoqu/A+Nae2FsUPp39veT9V52XIzn39MxhMVnogWq2ZsxrIkrPeo46YzLK9Dwz9EItl9M4GZ6LtamBrTzs2diO2+sTHRvLb/bGcbqs7Elvt603jXoUcep6PeqLA09rPhqcB26/3hUZ9vd77kQQghRGUVdPMWGL1+hycAJNBs0Uek49y0tPpolbw8s+rl60850f+1LADZ/9xY3zh0q8bgBH87FtWbDop+vnznAmQ3zSYwIwdDEHDf/xjQZMB47D99ixyZFhnFi9RyiL51Go1Zj51Gdxv1fwLvBo3lSLD0hhhUfPkObMW/j365vsf2JEaGcWPU7cVfPU5Cbi2M1fxr3G4dHQPNiYy/t30DQjmWkxIRjYmGDZ/1WNB0wAQt751JfX6MuYPWn47Dz9KPzix8XbT+y9AfObV1c9HO/93/DvXaTh7xa8bSQIrAQQgghnioNPKzoX98ZFyvjYvuiUrL5cN0l1Np7n0er1fL5llBSswtK3P/bvussPRlNI08rXmnvTV6BhvWBcby7+gJTe9agV93C4mqBRsubK4K5FJdBjzqO1HOz5FxUGouOR3HlZibfDqmDSqV6qKz3supMDLN3XaWmkznj23ihp4JtF+OZvjmE6JQcxrW+U/T/eONlDoQm0bq6LaOauREan8WGwFiCotOYO7qBTtE6LCELJ0sjXmrnXew167lZAeBjb8q03jVKzLXiVDSX4zLpWNMeIwO9onHTN4c8/EULIYQQ4pFyrdmQ2h0HYengUrQtKSIUB29/6vd4pth4G5c79wfBu1dxYMEszKztadT3efQNjbm0fz1rpk+g95Tvca3ZoGhsXFgwG2ZNwsjUgga9xmBgZMyFPavZ8t1bdH9tFtWbdnqo68hOT2HL7P+Rn5NV4v7EiFDWTB+PkYkZ9bqNxNDYlEsH1rPx69fo9upMfJt1KRp7dMXPnN30D9bOnjQdNBGtRs35nf9y4+xB+k/9Xec9uE2jUbP7j09JCL+Mnaefzj6/Vj2w96rJtVN7uHZq70Ndp3j6SBFYCCGEEE8VN2uTYrNbAQ6FJTFza0ipRd27LT8VzbnI1BL3XU/MYtnJaNr42vLlwNpFRdyBDVx4bsFZftx7jS7+DhgZ6HEgJJFLcRkMaujClK6FM10GNnTFxFCfDYFxHL+eQotqujOH7zdrWdJzCvhl33VqOpkzd0wDDPQKsw5p7MYrSwJZcDSSfvWdcbAw5lJsOgdCk2hVzZavBtcpOoeTpRFzD95gS/BNBja488EvLD6TJl42Jb7ft9mZG5W4/+i1ZK7EZdLV34Ghjd0AisZJEVgIIYSoeCyd3KnZplfRz7mZaWQkxVG9eRed7XfLTkvmyNLvMbGwZvAnC7CwK/x7H9B5MCs+GsXeedMZ/vky9A0M0Gq17PtrBvoGhgz86A+sHAvvEWq17cvS94ZybMXPD1UEjgsNYudvH5GeEFPqmCNLf0CrUesUcf3b92PZ+8M5svQHqjftjEqlIjEilLObF2Lj6s3gj+djZGpRlHX51BHs//tL+r//m865M5Pj2T33Y6IunCzxtR19/HH08Sf1ZoQUgcV9k57AQgghhHjqfbLxMu+tuYitqSFd/R3uOT40PpO5B8KZ0Lb47A2Ak+EpaIH+9V10ZvGaGxvQ1s+O9JwCriZkAhCZkgNAq7sKvW187QC4cjPzobLey7moNPIKNPSp61xUAAYw0FPR1d+BAo2WoKh0ACKSC7O2rF5y1pCbGUXbYlNzyMhVU93B7L4zZeepmbUtFGtTA/7XtfgjoEIIIYSo+BIjQgGwcy/7b3lE0BEK8nKp131kUQEYwNDEjHrdRpAae4Poi4VF0diQQJIiw6jfc1RRARjA2NyS1qPeomab3qjz8x4o77EVv7BmxgQ0ajV1Og0qcYxGo8bA2Bjf5l11ZvEampjh7FuXjMRYctKTAbh+eh9otTTu90JRARjA3NaRWm37En3xFCmx4UXbr53cw9L3hhIbEkSjfs8/0DUIURaZCSyEEEJUQT/svsrK0zH8PqoedW89dn/bgqMR/HHwBj8MD6CJlw35ag0rT0Wz+3Ii4UlZ5Ku12Jkb0qKaLS+28cLO3KjU1xk6t/CG/N+JTXW2zzt0g/lHIvhxuG6v2sNhSSw5EcXluAw0WqjuYMaIpm509Xe85zW1/abkfnL/tfLFJrham9xz3N2uJ2YzoY0XzzRzZ9GxyDLH5hZo+HTjFQLcLBnZ1I1f910vNqZ3XScaeVqXmCUlKx8A/VsFVy87UwDCk7Jp/Z/PSFHJ2QA4Wui+//eTtTyaeVmz8PlG2JkbFs96a6bx7azet7LeSMrWGXe7kO1ocafFRmh8YfG62q0icG6+GgN9vaJzlWXx8UjiM/L4oIdfsT7DQgghhJIOLf6OoO3LGDjtT1z86uvsO7X+L06s+p1+7/2Ce51mqAvyCdq+jLDjO0mJCUedn4eZtT2e9VrRbMhLmFnbl/o6i6YMAGDMt+t0tp9YM5dTa/8s1gs2/OxBzm5ZRML1S2g1auw8/Kjf4xn8Wna/5zX9/lzxPrZ3G/XNWp2ia3kk3ih8cud2T9/83GwMDI1R6enOR8xIigPA3qtmsXNYuxS2pIq/fhHPei2LisFe9VsDhe25CnKzMTQxo2brnveV724JESHU7z6SJgNf5NqpvVzYs6bYGD09fXq+8U2x7eqCgqJexkZmhffeGYm3r6t426ui67p2saiYnBgZhkdAM1qOeA09fUPObPj7oa5HiLvJXbUQQghRBfWp58zK0zFsuxBfrAi87UI8LlbGNPYsLM5OW3+ZQ2FJ9KrrRL/6zuSpNRy/lsKGwDji0nL5bmjAI8m04lQ0P+65RoCrZdHCcvuuJPLJxiuEJ2bfc7G50vrG/peNafFCZnn8MaY+hvrle0Dq133Xic/I5eshddAroVcvgJmRAb6OxW+zYlJz2B+SiI2pIdUczAFo62tHOz87FhyNwNHSiLqulgTHZDD/SATV7M3oVFP3A+L9ZC0PY0P9okLtf2XkFrAhMA5DfRX13At/h2o6WzC4oQvrA2OpZm9Ky2q2XE/K5ofdV3GwMKJf/TsLnITGF/bROxGewm/7rhOTlouhvooWPra81skHdxvTEvOkZuez7GQ01ezN6F239DYSQgghhBL82/UjaPsyQg5vLVYEDjm8BQsHV9xqF345vuPnD7h+9gC12valdoeBqPPziDh/lIv71pKRFEuft398JJkCty3l8JLZOPvWpemgFwG4emIPO3/7iOSY6/dcbK7zxE/v+RqmVrb3HHO3xIgrAFw5vIWt308hKzURAyMTqjXtROtn3sDUqvBJIkPjwvuQvOzMYufISU8BIDM5AYDkmOuFx5iYsvuPT7l6fBcFeTlY2DnTZMB4ancceN85b+v5xtfoG9zfvWRORirJUVc5veFvUuMiaD36f+gbGBRlhPJdF0Cjvs8VvX5afPSDXIIQZZIisBBCCFEF+TmaU8vZnD2XE3ijc/Wix/yDY9K5kZTNuFaeqFQqQm5mcjAsiWGNXXmjc/Wi44c1duPFRec4fj2F9JwCLB9yNmZsWi6/7LtOOz87Zg7wL2qRMLyJGx+tv8SCoxF09XfA27701gFl9ZV9WOUtqh65msyqMzFM612jxIXlypKVp+b/NlwmT63l1daeRf8m+noqXmjtyccbs/lk45Wi8R42Jnw3tA7GhvoPlPVhFGi0TN98hZTsfJ5p6oat2Z0PRCOauhFyM5Nvdl4t2mZjasiPwwOw/8+s8dszgYOi0hjTwgNbM0POR6ez8nQ05xcXLiLnZlN8pvS6c7HkFGgY08K9xAXxhBBCCCXZe9XA0cefsOM7aTP6f+jpF94jxYWdJyUmnCYDJ6BSqUi4cYXrZ/ZTr/tI2oz+X9Hx9bqPYPWn44gIOkpuZjrG5pYPlSc9MZajy3/Ep3F7erz+ddHfzvrdn2H7z+9zet1f+LXojq2bT6nnKKtf78O43Q4i/toFWgyfjKGxKRHnj3Fx31riws4z5OP5GJtb4XJr0bfQo9uKzea9emIXAOr8wqeOcjMLW1Rt/f4dTCytaf/8B2g0aoK2L2Pf/JnkZWfSoNfoB8p7vwVggE3fvEH8tQsA+DRuj3+7vkX7XGo2JHDbUkKPbtNZ2E6r0XDt1J5b15X7UK8vxP2QIrAQQghRRfWu68zsXVc5di25qGfr1uCbqIBeAYUF1RpO5mx/vUWxGa3JmXlYGBfeJmTmqR+6CLw/JBG1RksXf4dii5l19Xdkf0gS+0OTGFtGEfh2G4WyWJkalDo792ElZ+XzxdYQOteyv++CdHpOAe+uvsDF2Aw61bRnSCPXon2nb6Ty9qpgDPX1mNDGC19HM24kZbP0RBQTFwcye1gAPmW8L49aXoGGTzZd5lBYMvXcLZnY7k6/u2sJWUxaFkRuvppRzdyp62ZJfHoey05GMWlZELMG1aaBR+EM80417fGxM2VMCw9MbhWy29ewJ8DVkg/XX2LuwXA+6VtL57W1Wi1rz8XiZGlEl3K0CBFCCCGUUKtdPw4u/JqIoCN4N2wHwJVDm0GlolbbPgA4eNXkhd/3oNLT/TI3Oy2pqD9sXk7mQxeBr53cg0atxq9Fd3IydBes9WvZnWun9nL99L4yi8DZt2allsXE3KpYG4d7qd1hANWadKRhn2fRu/U+VG/WGRtXL44s/YGzmxfRYtgkHH388WnUnutn9rN33nTqdRuJSk+P4N2riAsNAigqtmsKCu8Hjcws6P/B70Xn9W3eleVTR3Bi9Rz82/d/6Pe1vBr2HoOevgHRl04TvGslqz99gYEf/YGJhTU+jdvj4O3Phd2rMTQxo1bbPqjz8zi9YX5Rq4jb1yXEkyC/bUIIIUQV1a22I7/svcb2i/G08bWjQK1h9+UEGnla68zANNTXY+elBE5cTyEyJZuY1FySs/K5XUrVarUPneV2D9n/znS9W2xaTpnn6Pvr8Xu+zoP2BC6PL7aGoNZqmdDGu1hBWq3RkpKVj5GBHmZGuh/2olJyeG/1Ba4nZdOhhj0f99Htd/fHoXAKNFp+HFFHp3VHOz97xv1zli+2hjJntO7jpo9LSlY+H6y7SFBUOvXcLflmcB2dmccLj0WSnlPAp31r0eU/i9J18Xfg2b/P8NmmEJZPaIyBvl6pRdwONe1xsjTiRHhKsX3BMRncTM9jVDN3nUXqhBBCiIqkRqseHFn2AyGHt+HdsB3qggLCju3Azb8xVo7uReP0DY0IPbqdyPPHSL0ZSXp8NNlpSXDrC2utRvPQWVJibwCw87ePSh2TnlB2a4EFk+/dN/hBegLX6TS4xO11uwzj6PKfiTx/lBbDJgHQ5eXP2Pf3F1w6sJFL+zcA4FitDl0nfc7GryZjbF74JbOhsWnRufX+U2A3NDahVpvenFo3j9iQc3g3bHtfWR+Ub/OuAFRr0hErJ3cOLfqWoB3LaTZoInp6+vSeMpvdcz/h7KZ/OLvpHwA86rag/bgP2PHzBxibW5V1eiEeKSkCCyGEEFWUlYkB7fzsORiaRFZeASfCU0nNLqBPvTuzWDNzC3hjRTCX4zJo4GFFHVdL+tR1praLBctPRbPtQvwDvbb6rsKxlsKf3+3uW2qR1qGMBegAZg+7d2/ishaxe1iHrxau9Dzqr9PF9gVFp9P31+P0CnDiw153ehdfis3gndUXSM7Kp399Z6Z09S22MFrozUy87cyK9W72sjOloYcVx66nkJlbgLnx471ti0rJYcqqYCKTc2hd3ZbP+tUqmsF7W8jNTEwN9ehcS7dPsY2ZIe1r2LP2XCzhSdn4OpqX+Vp2ZkaEJRTvj3cwNBGArv8pMAshhBAVjbG5FT6NO3D9zD7ysjOJCj5OTkYq/u36F43Jy85gw6xXib9+CddajXCqHoB/u344Va/Dua1LCDm85YFeW6tW37WhsJDcftzUUou0ZjZl/13t++7P93zdshaxu1/6hkYYm1uSl5NVtM3QxIyuL0+n1cg3SIuLwNTaHhsXL6JuLQRn7ewBgLmdU6l5bm/L/895n6QarXpyaNG3JFy/pJOp7zs/kRYfTWZSHBYOrljauxQVum9flxBPghSBhRBCiCqsTz0ndl1O4GBYMvuvJGJupE+HGndumleejuFSXAZvd/NlYAMXnWMTM/PueX59PRU5+epi2xMzdI91tSos/FqZGNDM20ZnX2xaLpfjMjC11S043u3u45600orQb60MxtfRjMkdq+kUsq/EZfDmyvNk5KqZ0MaL51t5lni8ob4emlJmW9/eqn74ydhlik3N4bXlQdxMz2NgAxfe6lK9WLEawNCgcJtGC/p37b59DWqNluw8NS8vDcTRwphvhtTRGVeg1hCZko17Cf2Az0amYWNqSE1ni0d0ZUIIIcTj4d+uH2HHdhB+9gDXTu7FyNScak07Fe0P2r6c+GsXaf/8+8VmxGalJt7z/Hp6+hTkZhfbnpWaoPOz5a3Cr4m5FR4BzXX2pSfGknDtIoYuZS++e/dxj0JqXARbf3gHF7/6dHhhqs6+7LQkctJTcKpeeG+Vm5XBtVN7sffwxbFabcz/U7SOCDwCgGutxgA4VQ8geNe/JEddxat+K53zpt2MAu68J49DSkw4m797C6/6rWg79h2dfbeLz/qGhetGZCbHExF0BJeaDbBx8dYp0kcEHUHf0Bgn37qPLasQd3v8K4sIIYQQQjFNvW1wsjRia/BNjlxLprO/g87sztTswrYGvg66PWfPR6dxNiINKCzqlcbBwojkrHzi0+8sapGWU1A0a/a29jXs0VMVthPILbjz6KNWq2X2rjA+XHeJiKTiH3QqkmbeNiX+D8DSuLC4Xe3W+5iWU8D7ay+Skavm7a7VSy0AA7Sqbkt4UjYnrqfobL8an8nZyDRqu1hg9ZA9mcuSr9Ywdd0lbqbnMbq5O293Kz5b+bbW1WzJztew6XyczvbEzDz2hyRib25IdUdzTI30MdTT4/j1ZIKi0nTGLjwWSUaumt51nXW2qzVaQm9m4u8iBWAhhBAVn0dAcyzsnLlyaDM3Ag/h27wbhsZ3vuC83Z/XzsNP57jY0EBiLhU+VaTVFP8i/TZzGwey05LJSLpZtC03M43ws4d0xlVr0hGVSo8zG/+mIO/O/ZhWq+XgP1+z7af3SIkJf/ALfUCWDq7kZqQScnQbqXEROvuOrfwFgFq3FlHTNzDg4D+zOLriJ51xydHXuLB3DT6NOxTNmK3WpANGZhYE7VxBbuade4zstGQuHdiApYMrTtV0v4B+pNfl6E5BbjZXDm8p6ut72+kNf9/K2BEAjbqAvfNmcGbjPzrjYi6f4dqpPdTpNKiovYUQT4LMBBZCCCGqMD2Vil4BTiw4GglAn7q6C5q19bPj39MxfLb5CoMaumJhrM/F2Ay2Bd9EX09FgUZLRm7pH1B6BThxLjKNt/4NZlBDV3Lz1awLjMPKxIDk//TN9bIz5flWnvx1OIIX/jlLzwAnzI302ReSyKkbqXT1d6CZj81jeQ+UsPh4JDfT8/B1MMPUSJ9tF24WG9PM2wY7cyNebufNmYhU3lt7kf71nfF1MCMyJYe1Z2PRV8GUrr4PlCE0PpOw+Ex8Hc3xK6M9w6agOK7czMTe3JBq9mYlZq3rZoW7jQmjmrtzMCyZb3aEcT46nXpulsRn5LHuXCzpuWq+GOBf1Mv3zS7VeGNFMG+vusCghi44WxlzKjyVvSGJNPGyZnhjV53XiEvLJadAg4uV8QNdrxBCCPEkqfT0qNm2D6fX/wVArfZ9dfb7NGpH0I7l7JrzfwR0HoqRmQXxVy9w5fBm9PT10agLyMvKKPX8Ndv2IebKWTZ9/RoBXYZSkJfDhT1rMDa3KuwrfIuNizdNBozn5No/+Pf/xlKrbR8MTc25dnI3URdO4teyO551WzyeN6EMevoGtH32XXb8/AFrP3+Rul2GYWRmwfXT+4m6cIKabXrj26wLAAZGJjToPZZTa/9k6w9v49WgLdmpiQRtX4aRqQVtRv+v6LxGpha0f/4Ddv02jVUfP0edzkPQajUE715Ffk4WXSd9rrOA3ZVDhW03arbp9UiuS9/AgHbPvcf2nz9gzYzxBHQeiqGJKdfPHCAq+DjVm3XBr2Vhj2VLB1dqtu3D5QMbQKvFpUZ90m5GEbRjGbbuvjQd+OIjySREeUkRWAghhKjietd15p+jkXjamRbrO9vEy4ZP+tZi8fFI5h++gaG+Hi5Wxkxo642PnSnvrrnIifCUUmdn9qnrREZuAWvPxfLTnms4WRrTv74zHrYmfLT+ss7YF1p7Uc3ejH/PxPDP0cIZIe42przRuRqDGrqWdPpK6+StRc/CErKYvjmkxDE/Dq+LnbkRjpbG/DmmAX8djmB/SCJrzsZiZWJAq+q2vNDaEy87sxKPv5d9VxKZfySCca08yywCnwwvnKmUmJnPjC0lZ53a0w93GxPMjAz49Zm6LDgayZ7LiWy/EI+pkT713S15rqUndVzvrMRd182KOaPrM+/QDdYFxpGTr8bV2oQJbbwKF37T130gLeXWrHQL47LbggghhBAVhX+7fpzeMB8bFy9c/HQXcXWv04yur8zg7KZ/OLn2D/QNDLF0cKXZ4JexdfNhy+z/EXH+GI7Vapd87vb9ycvO4MLu1RxeMhsLO2dqdxqItZMn239+X2ds00EvYutRnfPbl3N6w3y0Wi3Wzh60GT2FgC5DHtv130v1pp3o9/6vnF4/n7ObF6FRF2Dj4knbse8Q0Fk3V9MBEzC1suPC7lUcXvwtJhY2VG/Wmcb9x2NhpzuJwa9FN8ys7Tm1bh6n1v2JSqXCqXpdur4yvdi/w+65HwOPrggMhTN9+7//K6fW/cWZjX8XXperT9F1qVR3nqjq8PwHWDt5cOXwFsKO78DMxpG6XYfTqO/zGJtblvEqQjx6Ku2jWPJbCCGEEE+USqXi5txxSseoVGJScxj2x6lii7dVZd/vvoq9uRFjW1StRUfafnOIhh5W/Dyy3j3HOk0s/DAshBBClIdKpeK7IylKx6hU0uKjWfL2QGq27UPnFz9WOo6O3Mw0/nmjNy/+eVDpKI/UiTVzObX2T/q9/xvutZuUOfZ/rWzkXkgA0hNYCCGEEKJKikrJYe+VROq7W917sBBCCCFEFaPVajm7eREuNRsqHUWICkHaQQghhBDiqRKdmsO2CzdxsTKmgYe10nEem4SMPN7sXI0GHlWjCFyg0bLrUrzSMYQQQghRivSbUVw5tAVLBxdcazVSOg4AhiZmdH15utIxHpn465dIjrpGUkSo0lFEJSRFYCGEEEI8Vc5FpnEuMo2ONeyrdBG4qhR/b8sr0JTaX1kIIYQQyou5cpaYK2ep3rRzhSgCq1QqGvd7XukYj1TokW2c27pY6RiikpKewEIIIUQlJD2BhSgf6QkshBDifkhPYFHVSE9gcZv0BBZCCCGEEEIIIYQQQogqTIrAQgghhBBCCCGEEEIIUYVJEVgIIYQQQgghhBBCCCGqMCkCCyGEEEIIIYQQQgghRBUmRWAhhBBCCCGEEEIIIYSowqQILIQQQgghhBBCCCGEEFWYSqvVapUOIYQQQoj74+PpTnhktNIxhKjwvD3cuB4RpXQMIYQQlYSnlw+REeFKxxDikfHw9CbixnWlY4gKQIrAQgghhKhSDh06xMcff8ymTZswNjZWOs5jEx4ezvDhw1m/fj3Ozs5KxxFCCCFEJRQVFcXgwYNZtWoVHh4eSsd5bLKzs+nduzezZs2iefPmSscRQhHSDkIIIYQQVUZeXh4zZsxg6tSpVboADODt7c3IkSP5+uuvlY4ihBBCiEpq1qxZjB07tkoXgAFMTU157733mD59OgUFBUrHEUIRUgQWQgghRJWxcOFCPD096dSpk9JRnoiXXnqJkydPcvLkSaWjCCGEEKKSOXz4MOfPn2fChAlKR3kievTogZ2dHUuXLlU6ihCKkHYQQgghhKgS4uLi6N+/P8uXL8fHx0fpOE/M5s2b+f3331m9ejUGBgZKxxFCCCFEJZCfn8+AAQP43//+R9euXZWO88SEhoYyZswYNm3ahL29vdJxhHiiZCawEEIIIaqEb775hhEjRjxVBWCAXr16YWNjw/Lly5WOIoQQQohKYtGiRbi6utKlSxelozxRfn5+DBw4kO+++07pKEI8cVIEFkIIIUSld/LkSY4fP85LL72kdJQnTqVSMW3aNH7++WeSkpKUjiOEEEKICu7mzZv8/vvvfPjhh6hUKqXjPHGTJ09m3759BAYGKh1FiCdKisBCCCGEqNTUajXTp0/nvffew9zcXOk4iqhRowb9+vVj9uzZSkcRQgghRAX37bffMnToUKpXr650FEVYWFjw9ttv89lnn6HRaJSOI8QTI0VgIYQQQlRqy5Ytw8rKil69eikdRVGvvfYae/bsISgoSOkoQgghhKigTp06xZEjR3jllVeUjqKo/v37Y2BgwKpVq5SOIsQTIwvDCSGEEKLSSkpKok+fPixYsICaNWsqHUdxq1evZtmyZSxbtgw9PfmuXwghhBB3qNVqhgwZwvjx4+nXr5/ScRQXHBzMxIkT2bx5M9bW1krHEeKxk08HQgghhKi0vv/+e/r27SsF4FsGDhyISqVizZo1SkcRQgghRAWzYsUKzM3N6du3r9JRKoSAgAC6du3KTz/9pHQUIZ4ImQkshBBCiEopKCiIV155hc2bN2NlZaV0nArj/PnzvPzyy/K+CCGEEKJIcnIyvXv3Zv78+fj7+ysdp8KQ90U8TWQmsBBCCCEqHY1Gw4wZM3jrrbek0HmXunXr0rlzZ5nVIoQQQogi33//Pb1795ZC511sbW15/fXXmTFjBjJHUlR1UgQWQgghRKWzdu1atFotgwYNUjpKhfTmm2+yceNGLl++rHQUIYQQQigsODiYnTt38vrrrysdpUIaPnw4mZmZbNq0SekoQjxW0g5CCCGEEJVKeno6vXr14rfffqNevXpKx6mwFi9ezNatW/nnn39QqVRKxxFCCCGEAjQaDaNGjWLIkCEMGzZM6TgV1unTp3nzzTfZsmUL5ubmSscR4rGQmcBCCCGEqFR++uknOnXqJAXgexg5ciTp6els2bJF6ShCCCGEUMj69espKChgyJAhSkep0Bo3bkyrVq347bfflI4ixGMjM4GFEEIIUWlcuXKF5557jk2bNmFnZ6d0nArv5MmTTJkyhc2bN8usFiGEEOIpc/vpqV9++YUGDRooHafCi4+Pp2/fvixdupTq1asrHUeIR05mAgshhBCiUtBqtcyYMYPJkydLAbicmjZtSvPmzZkzZ47SUYQQQgjxhP3yyy+0b99eCsDl5OjoyMsvv8znn38ui8SJKkmKwEIIIYSoFLZs2UJKSgojRoxQOkql8s4777B8+XKuXbumdBQhhBBCPCGhoaGsXbuWKVOmKB2lUhkzZgwxMTHs2rVL6ShCPHJSBBZCCCFEhZeVlcVXX33F//3f/2FgYKB0nErFycmJiRMnMnPmTJnVIoQQQjwFtFot06dPZ9KkSdjb2ysdp1IxNDRk2rRpfPHFF+Tk5CgdR4hHSorAQgghhKjw5syZQ9OmTWnatKnSUSqlsWPHEhkZyZ49e5SOIoQQQojHbNu2bSQlJTFq1Cilo1RKrVq1IiAggD///FPpKEI8UrIwnBBCCCEqtPDwcIYPH8769etxdnZWOk6ldejQIT7++GM2bdqEsbGx0nGEEEII8RhkZWXRu3dvvvrqK5o3b650nEorOjqaQYMGsWrVKjw8PJSOI8QjITOBhRBCCFGhzZw5kxdffFEKwA+pTZs21K5dm3nz5ikdRQghhBCPydy5c2ncuLEUgB+Sm5sbzz33HLNmzVI6ihCPjBSBhRBCCFFh7dmzh/DwcJ599lmlo1QJ77//PgsWLCAqKkrpKEIIIYR4xMLDw1m6dCnvvvuu0lGqhPHjx3Px4kUOHjyodBQhHgkpAgshhBCiQsrNzeXzzz/no48+wsjISOk4VYK7uztjx46VWS1CCCFEFfTFF18wfvx4XFxclI5SJRgbGzN16lRmzJhBXl6e0nGEeGhSBBZCCCFEhfTXX3/h7+9P27ZtlY5SpUyYMIHg4GAOHz6sdBQhhBBCPCJ79+7l2rVrPP/880pHqVI6deqEl5cXCxcuVDqKEA9NFoYTQgghRIUji3E8Xjt37uS7775j3bp1GBoaKh1HCCGEEA8hLy+Pvn378tFHH9G+fXul41Q5169fZ8SIEbJIsaj0ZCawEEIIISqcWbNmMXbsWCkAPyZdunTBzc2NRYsWKR1FCCGEEA9p/vz5+Pr6SgH4MfHx8WHEiBF8/fXXSkcR4qFIEVgIIYQQFcqRI0c4f/48EyZMUDpKlaVSqfjwww/5/fffuXnzptJxhBBCCPGAYmJi+Ouvv5g6darSUaq0l19+mZMnT3Ly5EmlowjxwKQILIQQQogKIz8/n+nTp/PBBx9gYmKidJwqrVq1agwbNoxvvvlG6ShCCCGEeECzZs1i1KhReHp6Kh2lSjMzM+Pdd99l+vTpqNVqpeMI8UCkCCyEEEKICmPx4sW4urrSpUsXpaM8FV555RWOHTvGqVOnlI4ihBBCiPt09OhRAgMDmThxotJRngq9evXCysqKZcuWKR1FiAciC8MJIYQQokKIj4+nb9++LF26lOrVqysd56mxceNG/vzzT1atWoW+vr7ScYQQQghRDvn5+QwaNIjXX3+d7t27Kx3nqXHlyhWee+45Nm3ahJ2dndJxhLgvMhNYCCGEEBXCt99+y9ChQ6UA/IT16dMHCwsLVqxYoXQUIYQQQpTTkiVLcHR0pFu3bkpHearUrFmTvn37Mnv2bKWjCHHfpAgshBBCCMWdPn2aw4cP88orrygd5amjUqn46KOP+Omnn0hOTlY6jhBCCCHuISEhgd9++42PPvoIlUqldJynzmuvvcaePXsICgpSOooQ90WKwEIIIYRQlFqtZvr06bzzzjtYWFgoHeep5O/vT+/evWVWixBCCFEJfPvttwwaNAhfX1+lozyVrKyseOutt5g+fToajUbpOEKUmxSBhRBCCKGolStXYmZmRt++fZWO8lR7/fXX2bVrF8HBwUpHEUIIIUQpzp49y8GDB3n11VeVjvJUGzRoEABr165VNogQ90EWhhNCCCGEYpKTk+nTpw9//fUX/v7+Ssd56q1cuZJVq1axZMkS9PRkroAQQghRkajVaoYNG8Zzzz3HgAEDlI7z1AsKCuKVV15h8+bNWFlZKR1HiHuSu3shhBBCKOaHH36gV69eUgCuIIYMGUJBQQHr169XOooQQggh7rJq1SqMjY3p37+/0lEEUK9ePTp16sRPP/2kdBQhykVmAgshhBBCEcHBwUycOJHNmzdjbW2tdBxxS2BgIJMmTWLLli1YWloqHUcIIYQQQEpKCr1792bevHnUrl1b6TjilqSkJPr06cOCBQuoWbOm0nGEKJPMBBZCCCHEE6fVapk+fTpvvvmmFIArmPr169OhQwd+/vlnpaMIIYQQ4pYffviB7t27SwG4grGzs2Py5MlMnz4dmWMpKjopAgshhBDiiVu3bh0FBQUMGTJE6SiiBFOmTGH9+vWEhIQoHUUIIYR46l28eJFt27bx5ptvKh1FlGDEiBGkpaWxZcsWpaMIUSZpByGEEEKIJyojI4OePXvyyy+/0KBBA6XjiFIsXLiQnTt38vfff6NSqZSOI4QQQjyVtFoto0ePZsCAAYwYMULpOKIUJ0+eZMqUKWzevBlzc3Ol4whRIpkJLIQQQogn6pdffqF9+/ZSAK7gnnnmGZKSkti2bZvSUYQQQoin1oYNG8jJyWHo0KFKRxFlaNq0Kc2aNWPOnDlKRxGiVDITWAghhBBPTGhoKGPGjGHTpk3Y29srHUfcw/Hjx3n33XfZvHkzZmZmSscRQgghnioZGRn06tWLH3/8kUaNGikdR9xDXFwc/fv3Z/ny5fj4+CgdR4hiZCawEEIIIZ4IrVbLjBkzmDRpkhSAK4nmzZvTpEkTmdUihBBCKOCXX36hTZs2UgCuJJydnXnxxRf5/PPPZZE4USFJEVgIIYQQT8T27dtJTExk1KhRSkcR9+Hdd99l2bJlhIeHKx1FCCGEeGqEhYWxZs0apkyZonQUcR+effZZIiIi2LNnj9JRhChGisBCCCGEeOyys7P58ssvmTZtGgYGBkrHEffB2dmZ8ePH88UXXygdRQghhHgqaLVaPv/8c15++WUcHR2VjiPug5GRER999BEzZ84kNzdX6ThC6JAisBBCCCEeu7lz59KoUSOaN2+udBTxAJ5//nmuXbvG3r17lY4ihBBCVHk7d+4kLi6O0aNHKx1FPIC2bdvi7+/PvHnzlI4ihA5ZGE4IIYQQj9WNGzcYNmwY69atw8XFRek44gHt37+f6dOns3HjRoyNjZWOI4QQQlRJ2dnZ9OnTh88//5xWrVopHUc8oMjISIYMGcKaNWtwc3NTOo4QgMwEFkIIIcRjNnPmTMaPHy8F4Equffv21KhRg/nz5ysdRQghhKiy/vjjD+rVqycF4ErOw8ODsWPH8uWXXyodRYgiUgQWQgghxGOzb98+rl27xvPPP690FPEIfPDBB8yfP5+YmBilowghhBBVTkREBIsXL+a9995TOop4BCZMmMD58+c5cuSI0lGEAKQILIQQQojHJC8vj88//5ypU6diZGSkdBzxCHh6ejJ69GhmzZqldBQhhBCiyvniiy8YN26ctA+oIkxMTJg6dSrTp08nPz9f6ThCSBFYCCGEEI/H33//ja+vLx06dFA6iniEXnzxRQIDAzl69KjSUYQQQogqY//+/YSEhPDCCy8oHUU8Ql26dMHV1ZVFixYpHUUIWRhOCCGEEI9ebGwsAwYM4N9//8XT01PpOOIR2759Oz/++CNr1qzB0NBQ6ThCCCFEpZaXl0e/fv344IMP6Nixo9JxxCN29epVnnnmGTZu3Iijo6PSccRTTGYCCyGEEOKRmzVrFqNGjZICcBXVrVs3nJycWLx4sdJRhBBCiErv77//xsfHRwrAVVT16tUZOnQo33zzjdJRxFNOisBCCCGEeKSOHTvG2bNnmThxotJRxGOiUqn48MMP+f3330lISFA6jhBCCFFpxcXFMW/ePKZOnap0FPEYvfLKKxw5coTTp08rHUU8xaQILIQQQohHpqCggBkzZvD+++9jamqqdBzxGPn6+jJo0CC+/fZbpaMIIYQQldZXX33FyJEj8fb2VjqKeIwsLCx45513mD59Omq1Wuk44iklRWAhhBBCPDJLlizBwcGB7t27Kx1FPAGvvvoqBw8e5OzZs0pHEUIIISqdEydOcOrUKV566SWlo4gnoG/fvpiZmbFixQqlo4inlCwMJ4QQQohHIiEhgb59+7J48WJ8fX2VjiOekHXr1rFgwQJWrlyJvr6+0nGEEEKISqGgoIBBgwYxadIkevXqpXQc8YRcunSJcePGsXnzZmxtbZWOI54yMhNYCCGEEI/Et99+y6BBg6QA/JTp378/JiYm/Pvvv0pHEUIIISqNpUuXYmtrS8+ePZWOIp4gf39/evfuzffff690FPEUkpnAQgghhHho586d49VXX2Xr1q1YWFgoHUc8YRcvXmT8+PFs3rwZGxsbpeMIIYQQFVpiYiJ9+vRh4cKF1KhRQ+k44glLTU2ld+/ezJ07l4CAAKXjiKeIzAQWQgghxEPRaDR89tlnvP3221IAfkrVrl2bHj168MMPPygdRQghhKjwvvvuOwYMGCAF4KeUtbU1b775JtOnT0ej0SgdRzxFpAgshBBCiIeyatUqjIyMGDBggNJRhILeeOMNtm3bxsWLF5WOIoQQQlRYgYGB7Nu3j8mTJysdRShoyJAhFBQUsH79eqWjiKeItIMQQgghxANLSUmhd+/e/Pnnn9SpU0fpOEJhy5cvZ+3atSxZsgSVSqV0HCGEEKJC0Wg0DB8+nNGjRzNo0CCl4wiF3W6ntmXLFiwtLZWOI54CMhNYCCGEEA/sxx9/pHv37lIAFgAMHTqU3NxcmdUihBBClGDVqlXo6+vL01MCgAYNGtC+fXt++eUXpaOIp4TMBBZCCCHEA7l06RIvvPACmzZtwtbWVuk4ooI4e/Ysr732Glu2bJEe0UIIIcQttxcDmzNnDnXr1lU6jqggbi8SuGjRIvz8/JSOI6o4mQkshBBCiPum1WqZPn06r7/+uhSAhY6GDRvStm1bmdUihBBC/MdPP/1Ely5dpAAsdNjb2zNp0iRmzJiBzNEUj5sUgYUQQghx3zZu3Eh2djbDhg1TOoqogKZMmcKaNWsICwtTOooQQgihuEuXLrFp0ybeeustpaOICmjUqFEkJiaybds2paOIKk6KwEIIIYS4LxkZGXz11VdMmzYNfX19peOICsjBwYGXX35ZZrUIIYR46mm1WmbMmMFrr70mT0+JEhkYGDBt2jS+/PJLsrKylI4jqjApAgshhBDivvz666+0adOGRo0aKR1FVGCjR48mPj6eHTt2KB1FCCGEUMymTZvIyMhgxIgRSkcRFVjz5s1p3Lgxc+fOVTqKqMJkYTghhBBClFtYWBijRo1i48aNODo6Kh1HVHBHjx5l6tSpbNq0CVNTU6XjCCGEEE9UZmYmvXr1Yvbs2TRp0kTpOKKCi42NZcCAAaxcuRIvLy+l44gqSGYCCyGEEKJctFotM2fO5JVXXpECsCiXli1bUr9+ff744w+lowghhBBP3G+//UbLli2lACzKxcXFhfHjxzNz5kylo4gqSorAQgghhChGq9UW6+W6a9cuYmNjGT16tEKpRGX03nvvsXjxYiIiIpSOIoQQQjw2d983Xbt2jZUrV/L2228rlEhURs8//zzXrl1j7969SkcRVZAUgYUQQghRzIkTJ3Q+tOTk5DBz5kw++ugjDA0NFUwmKhtXV1fGjRvHF198oXQUIYQQ4rFISkpiwIABRT/fXgzupZdewsnJScFkorIxMjJi6tSpzJw5k7y8PKXjiCpGisBCCCGEKCY6OhqVSlX08x9//EG9evVo1aqVgqlEZfXCCy8QGhrK/v37lY4ihBBCPHLx8fGo1eqin3ft2kVMTAxjx45VMJWorDp06ICvry/z589XOoqoYqQILIQQQohiUlJSsLGxASAiIoJFixbx3nvvKRtKVFq3Z7V8/vnnMqtFCCFElZOamoq1tTVQ+PTUF198IU9PiYcydepU/vrrL2JiYpSOIqoQKQILIYQQopjU1NSiIvCXX37J888/j5ubGwB5eXloNBoF04nKqGPHjlSrVo2///5b6ShCCCHEI/Xf+6Y///yTgIAAWrduDYBarSY/P1/BdKIy8vT0ZNSoUcyaNUvpKKIKkSKwEEIIIYq5PaPlwIEDXLlyhRdeeAGtVsuWLVvo3LkzgYGBSkcUldDUqVOZN28ecXFxSkcRQgghHpmUlBSsra2JjIxk4cKFRU9PnT17lgEDBrBq1SqFE4rKaOLEiQQGBnL06FGlo4gqQorAQgghhCgmNTUVc3NzZsyYwdSpU0lJSeHVV1/lp59+4scff6Rhw4ZKRxSVkJeXFyNHjuSrr75SOooQQgjxyNyeCTxr1iyeffZZbGxs+Pzzz5k8eTKTJk1ixIgRSkcUlZCpqSnvvfceM2bMkNnk4pGQIrAQQgghiklJSeHkyZN4e3tz8+ZNBgwYQK1atVi7di2NGzdWOp6oxF566SVOnz7N8ePHlY4ihBBCPBKpqamkpKRw8eJF6tSpQ79+/UhLS2PDhg307t1bZ7FdIe5H9+7dcXR0ZMmSJUpHEVWASqvVapUOIYQQQoiKZcCAAdy4cQM/Pz8AZsyYQa1atRROJaqKLVu28Ouvv7JmzRoMDAyUjiOEEEI8lKlTp7J7925q1qxJZGQkn376Ke3atVM6lqgiwsLCGDVqFJs2bcLBwUHpOKISk5nAQgghhCjmxo0b5Ofn06dPH5YtWyYFYPFI9ezZEzs7O5YuXap0FCGEEOKhnT9/nuTkZGrVqsWGDRukACweKV9fXwYPHsy3336rdBRRyclMYCGEEEIU88EHH/DCCy9Qo0YNpaOIKiokJISxY8eyadMm7O3tlY4jhBBCPLAffviBOnXq0K1bN6WjiCoqIyODnj178vPPP8vaHOKBSRFYCCGEEEIo4osvviAjI4PPP/+8aFtqair6+vpYWFgomEwIIYQQomJZu3Yt//zzDytXrkRfXx+A3Nxc0tPTpU2EKBdpByGEEEIIIRQxefJk9u3bR2BgYNG2hQsXsmjRIgVTCSGEEEJUPAMGDMDY2JhVq1YVbTt06BCfffaZgqlEZSIrcQghhCji4+lOeGS00jGEqNC8Pdy4HhGldIwqwdLSkilTpvDZZ5+xYsUK9PT0MDc3JyYmRuloQghRbp7ePkTeCFc6hhAVjoeXNxHh15WOUWWoVCqmTZvGhAkT6N69OzY2Npibm5OcnKx0NFFJSBFYCCFEkfDIaOJ+HaV0DCEqNOdJS5SOUKUMGDCA5cuXs2rVKoYNG4a1tTWXLl1SOpYQQpRb5I1wFp5LVzqGEBXO2AaWSkeocurUqUP37t354Ycf+Pjjj7G2tiYlJUXpWKKSkHYQQgghhBDiidJoNEWFXj09PaZNm8b3339Pamoq1tbWpKamKpxQCCGEEKLiuHTpEhqNBoA33niDbdu2cfHiRWxsbOS+SZSbFIGFEEIIIcQTlZGRwauvvsqUKVNISkoiICCALl268NNPP2FjYyMzWoQQQggh/uPLL79k7NixXL16FVtbW15//XWmT5+OlZWV3DeJcpMisBBCCCGEeKKsrKzYsGEDDg4O9OvXj/Xr1/Pmm2+yadMmkpKS5MOMEEIIIcR/zJs3jx49evDMM8/w+++/M3DgQHJyctixYwdarZacnBylI4pKQKXVarVKhxBCCFExqFQq6QksxD04T1qC3D49OoGBgXz44Ye4urrSqFEj9u7dy40bNzhy5IjS0YQQolxUKpX0BBaiBGMbWMo90yMWFRXF//3f/5GQkMDYsWP54Ycf0Gq1rFq1CmdnZ6XjiQpOZgILIYQQQgjF1K9fn1WrVtGwYUP+/vtvoqOjSUlJkQ+NQgghhBB3cXd3588//2TcuHF89913WFlZkZubK09RiXKRIrAQQgghhFCUkZERkyZNYsmSJdjY2KDRaMjIyFA6lhBCCCFEhaNSqRg4cCAbNmzAx8eHtLQ0rl69qnQsUQkYKB1ACCGEEEIIAF9fX9atW8fSpUuxsLBQOo4QQgghRIVlb2/PL7/8wtq1a2nZsqXScUQlIEVgIYQQ9+3zbdfYejFRZ5ueCiyM9fFzMGNIQyfa+9nq7N8cnMAXO64zroUrL7RyL3XbvV7zxyE1aeRp9Wgv6Jb1QfF8vSscS2N91rzYAGOD4g/MnIlI4/VVVwDoVMOWz/r43jMzwIE3mwLQ7vuT5c5z+5j/KtBombj0Ir4OpnzYo1q5z/WklPS7cbcPuvnQO8ABKLye5adi2XQhgbi0PBwtjOhb14GRTVww0FOVeg61Rsvr/14mMDqjxPepJLP3hLP6XHyJ+z7qUY0ete3LdR7xeOnp6TF69GilYwghxCOz7uePCdy3UWebSqWHibklzt41aNZrBP4tOuvsP7dnPet//ZT2wybSYfhLpW6712uO/WQOPgHl+zt5v07vXM2mOZ9jYm7FW3O3YmBkXGzM9eCTLPykMGvtVl0Z+r9Z98wMMG3lKQCmD2tS7jy3j/kvjbqAeR88i5NXDQZM/rTc53pSSvrduFv/SR/ToFN/fpzUl9T4mDLHvvbLBmyc3AC4GRHGnqW/EB58CrRaXKr502rAs9Ro3LZc2fKysziw6k8uHN5BRkoiti4eNO81ksbdBpfv4sQTM3DgQKUjiEpCisBCCCEe2NhmLnjbmQKFxbyU7Hx2X0nmw41hvN/Nhz63Cn2VxZYLCZga6pGeq2b3lSR61Sk7/5FrqeTkqzEx1C+2L69Aw8GwlGLbP7qrcLs/NJn9YSn0r+tAfXfLMl9PrdEyc9s1QuKz8HUwvfcFKWBAPUeaehUv0mfmqflpXwRmRno08rhznV/vvM7mC4l0qWnHiEYunItKZ86hKMKTcsosci88EUNg9P21CwhLyMbJ0oiJrYt/4VDPTWadCiGEeLzaDHoBB4/Cv20adQFZackEH9rOym/eod+kj2nYqb/CCe/PuT0bMDIxIyczjeDD22nQsV+Z40NPHyQ/NxtD4+L3MAX5eVw+sbfY9gGvTdf5+dKx3Vw+vofGXQfjWbtRma+nUatZ9/PHxF67jJNXjXtfkAKadBtCtfotim3Pzcpg+9/fYmxqjvetIn7356eQl5NdbGzs1Usc27QY9xp1sbRzBCA67AILP3mJgvw8mnYfip2rF1dO7WfZF2/Q/fkptOhT9kLQWo2GFV+/zbXzx2ncdTCu1fy5fGIvm+Z+Tnpy/D2/hBBCVExSBBZCCPHAmnlZFZuVO7ShM6MWnGfuoSh617FHpSp9NmdFEp6UzfmYTMY0c+HfMzdZFxhfZhHYw8aYyJRcjlxPpVMNu2L7j15PJSNPja2ZAclZBUXb755tGpWSw/6wFAJcLcqciZqQkcf0bdc4HfHoVx9PzynA0uTR3BLUdbOgbgkF1U82X0Wt0fJ/Pavjal04Uyg4JoPNFxIZVN+R/3X2BmBAfUcsTfRZfS6e/vUcSyzOXojN4O9jMRjpq8hTl3/xsLD4bJp4WVbKGb8+3l6E34hQOoYQZfL28uR6+A2lYwhRYVVv0KLYrNxmPUfw6xuD2bPkZxp07Fdp7psSoq4ReSWQNoPGcXzzUk5tX1VmEdjOxZOk2AhCTh+iTquuxfaHnjlEblYG5tZ2ZKYmFW2v3763zrjk2AguH9+DR816xfb9V3pSPGt/msb18yce4OrKlp2RhqnFo3kqzaNWfTxq1S+2ffX3U9Fo1Ax8fUbRzF7/5p2KjcvLzuLAv39gZmXL0ClfoW9gCMDmuTPJy8nimak/4deoNQBNew5n9ewP2LXoR/wat8Xe1avUXMGHt3Mt6Bhdx75Bq/7PAtCo6yCWz3qLg6v/okGn/tg4uj709T8uHl4+REWEKx1DiDK5e3oTeeP6E31NKQILIYR4pIwN9Kjras7uK8mkZBdga2aodKRy2Rxc2MKgVTVrwpNyOBCWQmh8Fn6OZiWOb1PdhjXnbrL3SnKJReBdV5KoZm+CtYkByVkPt8DV/tBkZmy7hlqjZWwzFxaeiH2o8wEUqDUcCEthbWA85sb6zOzn99DnLM3+0GR2XUmiT4ADLXysi7ZvuVD4no9o7Kwz/pkmLqw+F8+2i4nFisBZeWqmb71GC28rsvLUnI0q33sbm5ZLRp6aavYVcwb1vYTfiCAn9IjSMYQok4lfK6UjCFHpGBqb4FGrARcObycrLRlz6+L3FBXR2T0bAKjRuC0Jkde4fGIvceEhOHuXPOO2ZtP2nNz2LxeP7CyxCBx8aDuOnr6YWdroFIEfxKVju1n70/+h1WhoM+gFDq3566HOB6AuyOfyib2c2vYvxmYWDH/324c+Z2kuHd9D8KFtNOw8oKiAW5o9y34lKeYGA16bjpV94f1UakIsMVcvUq1ec53jVSoVbQaN48KRHZzbs57OoyaXet7AfZvQNzCkaY9hOse36jeWkFMHCD64lTaDxj3klT4+URHhvL297PZkQijtm+5PfmKKFIGFEEI8cjGpeVibGGD1iGaXPm5qjZZtlxKxMNKnjrM5nWrYciAshXVB8Uy5NUP1bmaGerTwsebI9VRyCzQ6/YNz8tUcuZbKmGYunAhPe+h8YQnZNPG04pV2HhjqqR6qCHwzPY/15+PZeD6BxMx8nCyNeKFl4QyTmNRchs8Puuc5ytuDFyBfreG3g5FYmejzclvdNgwXYjOxNjHA3cZEZ7uLlTE2pgZciM0sdr4f90WQkavmva4+fLw5rNw5QuMLH5+sfqsInFugwUBPhX4ZfYeFEEKIJyHlZhSmltaYWljfe3AFoFGrCdq/CWMzC9z8AqjdqmthgXT7v/R+8YMSjzEyNce3YStCTh8kPzcHQ+M7f/vzc7MJOXWANoPGcS3w2EPnu3kjlGr1m9N1zBvoGxg+VBE4LTGO0ztWc2b3WjKSE7Cydy5qhZByM5qfXi27BQaU3Ku4NOr8fHYt/AFTC2u6jH6tzLEJUdc5uW0lXrUb68yKTkuMAyixIG/vVjj7NybsQpnnjgo5j5OXX7HWHW5+dQr3hwbf+2KEEBVO5fh0LoQQokLKyFOTkp0PgEYDqTkFbDyfwMW4TN7p4l1pCmxHr6eSmJlPz9r2GOjr0aa6DcYGemy/lMgrbT0wMyre8xegc007DoSlcORaKh1r3FkI7+DVVLLzNXSpafdIisBjmrlgqF9YZI5Jzb3v47VaLSdupLE2MJ7DV1MAaFnNmn51HWlVzRq9W4+e2pgZFOtZ/LA2BScQmZLLy23dsTHVnRUen5GHk2XJM8UdLQyJTdO91n2hyWwKTmBmP1/szO9vhnloQhYAJ26k8dvBSGLS8jDUV9HC25rJ7T2KFaKFEEKIRy03K4OstGQANBoNWekpnN21jujQYPpM/BA9/ZLvNyqa0LOHyEhOoH6HvugbGFKraQcMjIwJOrCFrmPewMi05Keo6rTpzuUTewk9c4jaLbsUbb98cj/5udkEtOn+SIrAbQaOQ9+w8D4h5Wb0fR+v1Wq5GniUU9v+5cqpAwD4NWpD426DqNGoLSq9wnsyMyvbYj2LH9bZPetIio2gy5jXMbOyLXPsvhVz0KgL6Dr2DZ3tRiaF739udvEv07PSUgBIT04o9bz5udnkZKZhZd+42D5DY1NMzC1Jjb//91UIoTwpAgshhHhgUzeUPBOzva9Npeq7ujm48Ea4a63CRzDNjPRpVc2avSHJ7LycRP96jiUe16a6NSYGeuwJSdIpAu+6nERtZ7NHVli8XQB+EMeup/L93htEpuTibGnEcy3c6FvXAUcLo2JjTQ31H+m/m1arZeWZm1ga6zOovlOx/Zl5ajwNS36PjA30yCnQFP0cn5HHVzuv0yfAgXa+ZX8oKklYQuFM4KDoDMY0c8XG1IDzMRn8e/YmQcszmDuyNm7WxVc1F0IIIR6VFV9NKXF7readqFdGf9uK5uzu9QAEtOkBgJGpGTUat+Xi0V2cP7SVxl0Hl3hczSbtMTQ24cKRnTpF4OBD23DzDcDOxfOR5LtdAH4QYWcPs3XeVyTFRmDt4EL7oRNo2HkgVvbF72OMTEzL7Et8v7RaLcc2LcHE3Iom3YeWOTY1IZZLx3ZRrV5z3GvU1dnn6FENUwtrQk4dIDcrA2OzO621LhzdCUBBXk6p58651cbM0KTkFlqGRibk5ZR+vBCi4pIisBBCiAf2ajuPop65Gq2WjFw156IyWB8Uz8SlF/lhaM1isz8rmpTsfA5fS8Xa1IAmXncW+ehay469IcmsC4ovtQhsaqhPy2rWHLl2pyVERm4Bx8NTebGNe4nHPGnBMRlEpuTiYWPM+918aOBuWepYjVZLWk5BqftvK++/6ckbadxIzmF0U5cSZ1NrtVDaXHGVSlW0T6vV8vm2a1gYG/B6hwf7gNjRzxZvOxPGNHXBxLAwS3s/WwJcLfhoYxh/HI7i417VH+jcQgghRHl0ffZNnL1rAqDVasjJTCfi4llO7VjFvA+e5dlP5txz9qfSstKSCTl1ADNLG6rXb160PaBtTy4e3cWp7atKLQIbmZji16gNIacOFLWEyMlMJ+zsETo/8+qTuoQyRV4JIik2AjsXT/pN+hiv2o1KHavVaMjOSL3nOcv7b3ot8BiJ0eG0Hvg8xqbmZY49tX0VGrWaVv3HFtunp29A2yHj2bHgOxbPmEyXMa9jZe9EyOmD7F/5B8am5ujpl1EKurXmrqq0uzSVikqyfqEQ4i5SBBZCCPHAajmZ0chTd3XkzjXt8LYzYfaeG/xzLIbXO5a+8nBFsP1iEgUaLY09LIlPzyva7mNngpG+iis3s7gYm0ltl5JvxjvXtGVvSDJHr6fSwc+W/aEp5Ku1dKlZMRZ26V/fEY0WNpyPZ/LKy1S3N6V/PUd61LbDwlj3NiAuLe+R9gTeF5oCUOrsYlND3dm+/5WTr8HcuLBYu/x0HKcj0pnZz488tYa87MJjCjSFn1JSsvPRV6mwLKMHdZdaJf97dPCzxcnC8JG07RBCCCHK4lq9Nj4Bun9DA1p3x97dh63zZnFg1Tx6jHtboXTlE3RgCxp1Ad51m5KWeLNou6NHNQwMjYm9dono0GDc/AJKPL5O6+5cPLqL0LOHqd2iM5eP70FdkE+d1t2f1CWUqXG3wWg0as7sXMuC/5uAo6cvTboNoV773piY636RnpoQ+0h7Al88thuA+u373HPspWO7sLCxp1q9FiXub9l3NAX5uRz490/++fhFACxtHRn42nR2Lvy+zP7TRrdmAOeXMls4PzcHS7uSJ0gIISo2KQILIYR45Lr72zF7zw3ORKYrHeWeNl8obAWxJySZPSHJJY5ZFxRfahG4dTUbTA312HMlmQ5+tuy+kkR9d4sS2y0owcHciAmt3Xm+hSt7Q1NYc+4m3++9we8HI+lSy47+9Ryo41L4mKCduSGzB9d8JK+r1Wo5eDUFXwdTqtmX/Dihq7UxCRn5Je4r7Bdc+B4eupqCFvhgQ2iJY/vNOYeLpRErx9d/oKx25oZF7SKEEEKIJ61eu15snTeL8ODyLyCmlLN7CltBXDyyk4tHdpY45tT2VaUWgWs0bouhsSkXj+ykdovOBB/egZd/oxLbLSjB0taRTiMn0X7oi1w8tpuTW1ew9a+v2LX4RwJa96Bxt8FF7RcsbOwZPe3XR/K6Wq2WKyf34eRdA0fPsp9MSoi6TmJ0OM17jSyzj3TbQS/QvOdI4m6EYGBkjLNXDTQaNau+ew/3GvVKPc7YzAJTC2vSk+KL7bvTL9i5/BcnhKgwpAgshBDikdPeeoxMr4IvDHc5LpOwhGw8bIx5pa1Hsf2pOQV8tTOcXZeTmNzeo9jMWSjsXdu6mg2Hr6UQn5HHyYh03njAlgWPk4G+Hl1r2dG1lh1hCVmsORfP9kuJbApOoH9dB97p6oOxgR5NvazufbJyiEjJJTEzn+7+pc+Iru1szrqb8dxMv1PwBYhNyyUlu6Coz/Lk9p6k56qLHf/z/gjCErKZPbgmRgal/65l56t5ZfklHC0M+XqgbpG7QK0hMjkXd+kH/FjU7/0cTev589eskleLf1zH3q+s7BzmLlvP1v3HSExOxcPVidH9uzG0V6cHOt8P81cwb+Um5n35Ps3q19bZdy0imh8XrOJE4EVycvPwdndmzMAeDOrevmjMuh0HmDb7zzJfo3/Xtsz434sPlE8IUbFob9043V5wrKKKCbvIzfAQ7Fw86XLXYmQA2ekpbPx9BsGHt9HtubeKzZwFMDQ2oWbT9oScOkBa4k2uBR2jx7h3nkT8+6JvYEjdNj2o26YHceEhnNy2kqD9mzm7Zx2Nuw6mz0sfYmBkTPX6Jc/EvV9JMTfISE6gXrt79xi+ceE0ANUbtip1zIUjOzAwNKZm0/Z41mpQtP1q4FHUBfn4BDQp8zXcfOtw49IZ1Pn5Oj2Wo0KDAYr1IRaPzvrJrbD3a0SbN+//C4aHOfZ+FeRmcWXr30Sf3klOWiLmDh5U6zAMn7YDy318yLZ/iD6zm+yUOExtXfBs0Qu/rmOKtSs59tsU4oIPl3ieNm/9jr3vnd/xmxeOErpzESk3LqIpKMDc0QPv1v2p1mFYhf//2CdBisBCCCEeuS0XEgFo9ogKio/LplsLwg2o70h7v5L7tW2/mMjZqAy2XUxiSMOSZ6l0rmnLritJfLf7BoDOInEVka+DGW938ebltu5svZBIfCmzcR/G5bjCFan9nUvvade1lh3rguJZfjqO1/5TOF96KhaAnrfaSNQq5RyWt9pF3KtwbWqoj6G+iuPhaQRFZ1DP7c4CKQtPxJKRp+bZANdyXJW4XzPfnoi9TemPnD6uY++HRqPhzek/cuzcBYb27EhtP292HznNZz/9zc3EFCaNGXRf5zsZdIn5qzaXuC8yNp6xU2aQl5/PqP7dcHGwY9PeI3z8/TwSklN5cUThY8VN6tZi5tsTSzzHz/+sJiY+kS6tyv4AL4SoPAL3bQR4ZAXFx+XsnnUANOk+FP/mJX9JFrR/C+EXThG0fzPNeo0ocUydVl0JPrSNLX9+CUDtll0fT+BHxNm7Bn0mTqXL6NcI3LeRtBJmyD6s6LALALj61r7HyDtj3XzrlDrm5LaV3AwPZfLP64qK8bnZmexd/juWto5Fi/qVJqBdT8LOHeHUjn9p3vsZoPDLiqMbFqFvYHjP48WDa/TsxxhbPlhbuYc59n5oNRqOz32fhCsn8W49AGvPWsQG7idw2SxyUhPw7zPhHserOT7nXRJCzuDVsg823rVJvhbMpY1zSb52nhYvf6MzPi06DGvPWlTvNLLYuSyc7rQejA06yPE572Dm4E6N7s9hYGxKzLl9nF/1PalRITQa89GjeQMqMSkCCyGEeGAnbqRx8z8FxDy1htMR6ey5koSrlRGjmrqU6zyHr6WSlFXygmTDGzvjZWtS9PPyM3HsulJy24a3u3iXO3tegYadl5Mw0lfRu45DqeOGNXLm7K3F7korArfwscbMSI+DV1No7m2FrVnFXgzvNgtjA4Y2ejyP891ILuwj52xZeluMhh6WdKlpx4ozcaTmFNDQ3YLTkensuJRE37oOBLhalHpsWULjswhLyMbXwbRo4cI3Onjx5uorvLM2hIH1HXG2NOJURDr7QpNp4mnJsFL+bcXD6du5jSLH3o+t+49x9Gww/xs/gueHFM7AGtKzI699+j1/rtjAwG7tcHMu/f8j/istI5MPv5mLgb4BeZriX678s3oLaRmZfPPBq3RvV7ig0pBeHRk+eRpzlqxjZN8uWJqb4eHqhIdr8d/JFZt2E30zgfHD+9KpVeOHuGohhBKunjum00dXnZ/LtaATXDiyExtHN1oPeK5c5wk5dYCMlMQS97XsOxp7tzv3Q0c3LCb40PYSx/aZOLXc2Qvy8zh/cBsGhsY06Fh6H9zmfZ4h/MIpTu1YVWoR2K9RG4xMzblych++DVphbl2xvzy/zcTcsqgg+qglRocDYON47y+lE6PDMTA0xty69GJfuyETWDxjMgs+fpFGXQaCVsvpnWtIio1g+DvfYmh85946LjyEuPAQnL1r4OxdA4D67XpzZsdqti+YTVJsBM7eNbh0bA+hZw7RceQkrB3Kd48v7p9n856KHHs/ok7vJOHyCeoMnIxf19EAeLcZwPE57xCyfQFerfpgZlf673LUqZ0kXDlFrT4TqNVrPAA+bQdhaGrO1b0riL98AsdazQDIy0ojOzkO10ady7w+rVbLuaWzMLF2oMN7CzC8tbhitQ7DOP7H+0Qc3YRPu8HYepf+5cnTQIrAQgghHtjCE7E6P5sY6OFsZcSQhs6MaeaCVRkLdf3X5ZtZXL6ZVeK+LjVtdYrAh66Wvgrz/RSBD4SlkJ6rpmdt+zJztvW1wc3amKuJ2QRGldzj2NhAj7bVbdh+KanCLAintJTswqK+ZQktNP7rwx4+eNoas/VCInuuJOFkacTLbd0Z0fjBP1zsD01m/rEYxrVwLSoC13Wz4PcR/vx1NJr1QfHkFGhwtTJmQis3nmnigoG+PB72tNqw6xCGBgaM6NOlaJtKpeL5Ib3Yf/wsW/YdZfzwvuU614xfFqDRahnWuxOL1xUvutyIjgOgbbM7jy0aGhjQpkl9/lmzlasR0TTw9yvx3HEJSXw7bxnVPF3ve3ayEKJiOLTmL52fDY1NsHZwpXmvkbQZ9DymluV7+iHm6kVirl4scV9Am+46ReCQU/tLPc/9FIEvH99LTmYa9Tv0LTNnraYdsHV2Jz4ijBsXz5Q4xsDImFpNOxB0YDMBbSrGgnBKy0ornOBgbFa8hUZJY03My/6ivFq95oz68CcO/PsHe5f9hr6+Ae4169N/0sfF+jVfOrab/Svn0n7YxKIisEpPj2em/sjeZb9x4ehOzuxci52rJ31fnlZYVBZPtcjjW9AzMMSn/ZCibSqVCt8uo4g7f4iokzuo0f3ZUo/Pz87Ayt0P79YDdLY71GrG1b0rSLlxuagInBZVuCaIlWvZvbLTo8PITUvAt/MzRQXg2zyb9yT23D4SQ8489UVglfZ2AyIhhBBPPZVKRdyvo5SOIUSF5jxpCU/69kmlUpETeuSJvmZZzl0M5bfFawi6fBWANk3qMWZgD8b87zNeHjWwqEh5d1/fF977guS0dL54+yW+n7+CsxdDUVHY/uDNF4bj532nN3d5egL/umgNvy9ZW2bWe/XObTt8Eh4ujiz78VOd7dk5ubQYPJEurZsw+6PXy3wNKCwmf/TdH8z5/B1On7/C70vWFusJ/OXvi1iyfgfLfviEOjWqFW1//bPv2Xv0DNv+/g5XJ/sSzz/1mzls3H2YP2a+R4uGFfMDjIlfqyf+34YQFYFKpWLhuYq/GK4QT9rYBpaK/F1QqVS8vb3k2fJKSboaxOXN80gOL+yr7FS7Jb6dR3LgmwnU7DW+qIXC3X19D30/ibzMVBo/9zEX1v5K0rUgVCqw921I7YGv6hRHy9MT+NKmP7myZV6ZWT1b9KbR2Gml7t/ybnfM7N3p8N58ne0FeTls/l8nXBp0oPmLX5b9hpTg8uZ5XN78J03GTce9SWGbmKt7V3D+39m0e2cett51KMjNRt/QuFh/X01BPlmJ0RgYm2Fi46iz7/rBtQQum0XAkDfwLaGlhFK+6W7/xP/7kJnAQgghhBCi3E4EXuSVad9iZWHGs4N6YmpizLqdB5j8yXflOj4xOZXx739Jx5aNeHvCSK5ci2DF5t1cunqDrX9/i0EZK53frWubJni5ld3Kw9O19JYn2Tm5pGVk4uJYq9g+UxNjLC3MiIpLuGeOyNh4vvhtIWMGdKdlwwBOn79S4rgXhvXh0Kkgps3+k6mTnsXZwY4t+46y9+gZBnVvX2oBOOxGFJv2HKFt0/oVtgAshBBCiJIlXDnN0V/fwtDMEt/Oz2BgZMqNY5s49tuUch2fm5bIoR9exaVeOwIGvUZadCjXD6whNSqErp+uLraQWllcG3bE3LH4gtj/Ze7gXuq+grwc8rPSMfUrfv9lYGSCoakl2Ykx5c6jzs8jKzGamLN7ubJ1PjbedXBt0KFof1pUCACRJ7ZxfM675KYlom9kgmuDjgQMfq2oB7KegSEWzsWfCtWoC7i2byUADjVkPQUpAgshhKgysvLUZOdryjXWUF9V7nYVQog7Zv76DwYG+iz54RNcHApvvIf36czYKdNJScu45/EpaRk6/XcB8vILWL1tHyfOXaRV4/KvOF6zmhc1q3nde2ApMrKygcKCb0lMjY3Izskt8xxqtYYPv5mDs6Mdrz8/tMyxTva2TB47mI9/+Itx784s2t6xZSM+mlx6L9BFa7eh1WqZMKL0PpxCCHG/8rKzyMspuR3X3fQNDMvdrkIIoStoxTeo9A1o/85fmNoWFk992g3iwLcTycssvdXdbXmZqTr9d6Fw5uuNw+tJuHIap9rNy53F2t0Pa/eSW0+VR0F24b2evrFpifv1jYwpyMsp9/luHNlA0IrCheCMLGypP/xt9AzurK+SFhUGQEr4BeoMmIS+kSnxl44Rfng9ydfP0+6deRiZlb5I9Pl/Z5MecxXXRp2w9qhR7lxVlXz6FUIIUWUsOxXL/GPl++a5obsFPw3zf8yJhKhaQq5HEnYjmhF9uhQVgAFMjI14fkhvPvj693Kdp0+n1jo/B9Soxupt+0hIvvcHof/KzsklJzevzDHGRoaYmZqUuO/2I3gqVKUcrUKlKm1foT+Wr+f8lWssnv1/GBuVvhAiwLwVG/nh75V4uTnz7AsjsLe14tT5yyzbuIuJU7/i50/ewtxM90NVRlY2m/Ycob6/L40DapZ5fiGEuB9HNixk/8q55RrrXacJz35avrFCiDvSosNIj72GT7vBRQVgAH0jE/y6jub0gk/KdR6PZj10frbx8ufG4fXkpt9f24uCvBzU9yjS6hsaYWBsVvLOe907qVTc49ZJh613HZpNnEVOchyhO5dw4LuJNB3/Oa712wPg1bofLg3aU6PbWFR6hU+LuTXqhIWTN8FrfiRs52Jq93+lhJhazq/8jusHVmPu5EWDZ94vf6gqTIrAQgghqoyedRyo737vBTUALI3L/8i5EKLQ9cjCL1l8PIov3Ofr5Vbu89jb6M7YMDIsvCXVaMo3k/+2+f9ufqiewLeLw9mlFJJzcvNwsi991frAS2HMXbqeZwf3xNnBjuTU9KLjADIys0lOTcfa0pysnFzmLF2Hk70NS2Z/jJVl4aIlXVo3pY6fD1O/mcufyzfwxrjhOq9x4MQ5cnLz6Nu5TZnXKYQQ96t+hz54+jcs11hT89Jn2gkhSpcRdwOgxFYFlq7Vim0rze22B7fpGRR+8ay9z3un0B2LHqonsP6t4nBphWR1Xg4m1o4l7iuJjXdtbChcP8Glfgf2fD6K8//OLioC+7QteTFcn/ZDuLDuF25ePF6sCFyQl8PpBZ8Qe24fFs7etHrtxzJnCz9NpAgshBCiynCzNsbNuuTHuoUQD0+tLvygYWRoeI+RZdO7azGPB9W/S5t7zo51tLcpdZ+FmSnWlubEJyUX23e7X7Czg10JRxY6eDKQArWav1Zu4q+Vm4rtf2P6DwBsmf8NKakZ5OTmMbBbu6IC8G29O7Zi+s8LOHzmfLEi8J6jpzHQ16d7u2ZlXaYQQtw3W2cPbJ3L7g0qhHg4Wo0aQKfFwYO4eyG0B+XZohf2vvXLHFNWEdfQ1BxDMytyUouvmVDUL9im7PUaSmNq64S9XyPizh8kLyMVI4vSW9DoGxphaGZJQa5uS5vcjBSO//42ydeDsfUJoPnL32BsYfNAeaoiKQILIYQQQohy8XIvXGTtWmR0sX3hUbFPOg4erk54uD7YB43bAmpW5/T5y+TnF2BoeOfW+PyVqwDUq1W9tENLLUKv33WQjbsPM2XCSGpV88LB1prMrMIZM+pSZuxotdqiIvt/nQq6jL+vF3bWMoNFCCGEqGzMnTwByIgLL7Yv4+aNJx0Hcwf3Mhd+Kw8b79okhZ1DU5CvU9xOuX6hcL9PQJnHn/jjA5LDL9Dl45XoG+q20irIzQKVHnqGhmTER3Bi7vvYVatHg1G67Rxy05PIy0jBxvvOgrl5WWkc+ek10qJCcanXjsbjPsPAqOSWYE8rKQILIYQQQLvvTz5wn+CHOfZ+ZeWp+ed4DLuvJJGYmY+7jTFDGzrTv175HrvKyC3g72Mx7A9NJj4jH3MjfRp5WjKhlRvediUv8ACg1mh5/d/LBEZncODNpsX2p2YX8NfRaA6EJZOaXYCbtTH96joyuKETBnr30RhMVGi1fb3xdndh896jjB/WF3vbwhka+QUFLFi9VeF0D6Z3x5YcPhXEis27GT2gO1BYkF2weiuGBgb07tiy1GNLK0KfDr4CQB0/H5rVL3zE0c/bHTcnB3YcPMHLowbiaGdTNH7N9v3k5ObRunE9nfPEJ6UQn5RCp1aNH/YyhRDikZo+rMkD9wl+mGPvV152FgdW/cmFwzvISEnE1sWD5r1G0rjb4Ac6367FP3F47d+M/WQOPgG690PLvniDkNMHSzzuuc/+xKt2oxL3pSfFM+ftETh5+knf5SrI2qMm5k5eRJ7cjl+3ZzGxKnzCSKMuIGzXUoXTPRiPpt2Jv3iM6wfXUL1j4RNMWq2WsN1L0DMwxKNptzKPN7V3JebcXsIPrS06HiAx7BxJYedwrNUMA2MzzOyMyMtMJfLkdny7jcbC0bNo7MX1vwHg1bJP0bYzCz4lLSoU96bdafzs/xX1EBZ3SBFYCCGEAD7qUQ07swf7s/gwx94PjVbLhxtDOXUjnf71HKnpZMaBsGS+3hVOQkYeL7Qq+1v9Ao2Wt9eEcCE2kx617QlwNSc2LY+1gfEcD0/l9xG1qWZfciF44YkYAqMzStyXnlPApBWXiE3LZVADJzxsjNkfmsJP+yOITsvlzY5eD33tomJQqVR8OOlZJv3ft4x4/f8Y3rszpqYmbN5zhLAbUbfGKBzyPvXt1JpVW/byzR9LiYi5SQ0fD3YdPsXBk4FMfnYILo72RWMjY25y9mIInq7ONKh9fytr6+np8fHr45j8yWyeeeMThvbqiIOtNecuhrJh9yGqe7oxYURfnWOuRRT2YHZzcnj4CxVCiEdowGvTsbAuvV3O4zr2fmg1GlZ8/TbXzh+ncdfBuFbz5/KJvWya+znpyfF0GP7SfZ0vPPgUR9b9U+r+uPAQXKr506Lv6GL77N18Ss6o1bL+l4/JTr+/hVFF5aFSqag/fApHf/0f+2c9h0+7wegbmxJ1YjvpMVeLxlQmHs16En5oHcGrfyQzPhIrdz9izu7l5oUj+Pd9CVNb56KxmQlRJF0NwtzBHbvqhV921+zxHHFBBwle/SPpMVex9vQnPfYa4QfXYmRuTb0RUwDQ0zeg3vC3OTnvQw599zI+7YdgaGpJbNB+Ei6fxKN5L9wadQbg5oUjxAUfRt/IFIeaTYg8uaNYbit3P6zd7+/+raqRIrAQQggB9Khtf+9Bj+HY+7H7chInb6QzqZ0HzzQpXJirX10H3l8fysITsfQOcMDFqvSeyJvOJxAcm8mr7TwY2eTOwl6datjy0vJL/Hogkq8H1ih23IXYDP4+FoORvoo8tbbY/j8OR3EjOYeZ/Xxp51u4iNaAeo5MWRPCqrM3Gd3UBUcLo2LHicqpZaMA5nz+Dr8uWsOfKzZiaGBA++YNeKZ/Vz769o+H7hf8pOnp6fHLZ1P4+Z9V7Dh4glVb9+Ll5swnb7zA4B4ddMaeOn+ZabP/pH/XtvddBAZo1bguC7+bxtyl61i8bjuZ2Tk4O9gxZmAPXnpmAJbmuitxJ6emARTbLoQQSqvfvrcix96P4MPbuRZ0jK5j36BV/2cBaNR1EMtnvcXB1X/RoFN/bBxdy3WunMx01v38f+gZGKDOL76YaHZGGmmJcdRp1fW+ru/YxsWEXzxT7vGicnL0b06ryT9yefMfhGz/B5W+Ac5121Ctw1DOLJz+0P2CnzSVnh4tXvmOS5vmEn16N+GH12Hu6EmDUR/g3bq/ztjE0LOcXTQDzxa9i4rARubWtJvyB5c2/0HsuX3cOLIRYys7PFv0olav8ZjY3HnC0a1hR1q//hMh2xYQtmsxGnUBFk5e1Bs2BZ92d2b0x186AYA6L5tzS74oMXfNXuOf+iKwSqvVFv80J4QQ4qmkUqmI+3WU0jFEKd5ec4XTkelsfrkhJoZ3Hm86E5nO6/9e5qU27oxpVvqHmY82hrIvNIVtkxphZqT7eNTzi4KJTs1l+6u6j51n5akZv+QCXrYmZOWpORul2w4it0BD/7lnqeNizuzBtXSOPR+dwbHwVHrUtsfDpur043KetIQnffukUqnICT3yRF+zJFqtlsTkVBz+08rgtq37jvHurF+Z/tYEBnRr9+TDCcWZ+LV64v9tCFERqFQqFp5LVzqGuMuSz1/j+vkTvPP3HgyN7zzpFB58in8+mUjnUZNpM2hcuc61+vupRFw6i3+LzhzfvLRYO4jb5+w36WMadupfxpnuiAsPYd4Hz9Jp5CvsXPjDE2uR8SSNbWCpyN8FlUrF29sTn/jrlkSr1ZKbnoSJVfFJI1GndnJq/jQajvlIp62BeDp8093+if/3ITOBhRBCVGnnozP462g0F2MzAWjuY8XwRs68vPwS41q4FrVQuLuv72srL5GaU8C0HtX57WAk52MyUKmggbslr7T10GmbUJ6ewH8diWL+sZgys/asbc+HPaqVuv9CbCbV7U11CsAAtZ0LZwnevsbSTOnszbPN3YoVgKGwp69+CY+i/bgvgoxcNe919eHjzWHF9l+OyyQrT0MLnzur92blqTE11KOumwV13SzKzCQqn97j36FeLV/mfam7QMfGPYcBqP8AM2SFEEJUDJGXA9m3cg5RIecB8G3YmpZ9R/HX1OdpP2xiUQuFu/v6/vPxRLLSUxj42nR2Lf6RyMuBoFLhXbsRnce8jpOnb9FrlKcn8L4Vc9i/suyCaP0OfRkw+dNS90eFnMfJy0+nAAzg5le4kFRUaHCZ578tcN8mgg9vZ8y0X7lRyqzduPDCXvC3rzMvJxtDI2NUenolji/Iy2XNDx/iUbMeLfuOYefCH8qVRVROuz4egm21urR+/Wed7ZEnCtdTsKtWV4lY4ikkRWAhhBBV1pmINN5eG4KFsQEjmjhjYqjHlguJvLcutFzHJ2UW8Pq/l2nra8Pk9p6EJWSxNjCe0PgsVrxQ/74WPGvvZ4v7PWbDuluX3sohJ19Neq4aJ8vibRVMDPWxMNYnJq3444n/ZWtmiK1Z8cfNdl5KJCEzn3a+Njrb94Umsyk4gZn9fLEzL/kxtetJOQA4Wxjx97Fo1py7SVJWARZG+vQOcOClNu4YGZT8AUhUPiqVigFd27F80y5e/+x72japj1qjZu/RMxw5E8zIvl2o5lG+R2uFEEJULNeDT7Lk89cwNbekZb8xGBmbcm7vBpZ+8Ua5js9ISeCfTyZSq1kHuj37FnHhIZza/i+x16/w+q8b0NMvf/nBv0VnbF08yxxj5+xR6r783GxyMtOwsi++sKahsSkm5pakxkffM0dyXBRb582iRe9RVKvXvPQi8PUQAIL2b2b5rLfISEnE0NgE/+ad6fbcW5jf1QN558IfSEuM45kPfii1UCyqBpVKhWfLPlw/sJrjc97FqU5LtBo1sUEHib90HJ/2Q7Bw9lY6pnhKSBFYCCFElfXdnhvo66n445naRcXTQfWdeHnZRVJzCu55fGpOgU7/XYA8tZaN5xM4E5FGM2/rMo7W5edohp/jg/f1zMhTA2BiWPIHBRMDPXIK1Pd93utJ2czeewMDPRXPt3Ar2h6fkcdXO6/TJ8ChqM9vSdJzC9/HeUeiyS7Q8HwLN2xMDdh1JZkVZ+KISs3hy/7F+wyLyuu9l0dTzdOVtTsOMPuv5QBU83Tj49fHMaRnR2XDCSGEeGBb/pyFvr4B479ciJV94cJOTboPZf6H48q1cFl2eqpO/10AdUEeZ3at5fr5k1Rv0LLcWZy9a+Ds/eD3DzlZhYvZGpqUvOCtoZEJeTk5ZZ5Do1az7qdpWDm40HnUq2WOjbtRWASODrtA5zGvY2RsytVzRzm9aw1RIUG88MU/mFpYARB65hAnti5nwGvTsS5nT2JRudUd+hYWzj5EHN3IhXW/AGDh7FNiD10hHicpAgshhKiSriZkcz0ph0H1HXVmzxob6DGqqQufbb1WrvN089edueHvZM5GEkjMuncR+b9y8tXkFGjKHGOkr1diqwYAbrWLKm3usUoFqlL3liw0Pospa0JIy1HzVicvajoVFqm1Wi2fb7uGhbEBr3coexZOwa2F4hIy81n0bAAOtxaA61TTjmmbwtgbkszx8FSa30fBXFRsBvr6jOrfjVH9uykdRQghxCNy80YoCZFXadpjWFEBGMDQ2IRWA55l7Y8fles8ddv10vnZ1bcOZ3atJSPl/vqz5udmk59bdpHWwNAYI9NSvmAvum8q5d5IpaKELlg6Dq6eR3TYBV6YuQADo9Kf1gJo1GUg/s070XrAc+jpF97L1W7ZBXs3b3b8M5sj6/+h86jJZKYms/6XT6nTqtsTWyBPKE9P34DqHYdRveMwpaOIp5wUgYUQQlRJN5ILPzh42RZvweBjV/KskJLY3dU+wVC/8BODRnN/TfyXnIx9qJ7Aprf6AOeWUkjOydfgYF68VURpjl1P5ePNV8nMU/NqOw8GN3Aq2rf8dBynI9KZ2c+PPLWGvOzC1yy4dc0p2fnoq1RYmhgUzUxu52tTVAC+bWB9R/aGJHPiRpoUgYUQQogKLDE6HAB7t+KPpTt6VC/3eSysdRe/MjAovI/Sau7vaaXD6/55qJ7ARrdmAOfnlVxIzs/NwdLOsdRzR4UEcWDVn7TsOwYreyey0pKLjgPIzcogKy0ZUwtrVHp6NOk2pMTzNO05nJ2LfuTquaN0HjWZDb99ikZTQMeRrxSd8za1uoCstOSyi9tCCPEQpAgshBCiSlLfKlga6j9cnzW9e00TKaeedRyo725Z5hj7UvruApgb62Nlok9CRn6xfXf6BZd+/H9tCk7g612FH/be7+ZDnwAHnf2HrqagBT7YUHLv5H5zzuFiacTK8fVxulX4LSn77W3ZeWXPgBZCCCGEsjTqwiKtvmH5v1AuyaPqb1u/Qx88/RuWOcbStvQirrGZBaYW1qQnxRfbd6dfsHMJRxYKPXMYjVrN4XULOLxuQbH9K76aAsBrv2zAxsmt2P7bDAyNMDW3JDc7C4CQUwcA+PWNwcXGRl4+x7fju95zwTshhHhQUgQWQghRJXnYFj62F56cXWxfRErZjxc+Dm7WxriVsfBbefg7mxMYlUG+WqNT3L4YmwlAbRfze55j4/l4Zu0Mx9RQj+l9fGnhU3yG7uT2nqTnFp+x8/P+CMISspk9uCZGBiqd17yWWPx9jkrJBcDV6uE+UAqxbscBps3+k+lvTWBAt3ZKx3koBWo1o978lJrVPJnxvxfvOf7dWb+ydd8xtsz/Bndn3YLHoFemEhYeVeJx2/+ZjYuDXYn7vvhtIfuOnWXr39/e/wUIIaokO9fC9k+JUdeL7UuMCX/CacDW2QPbMhZ+Kw833zrcuHQGdX4++oZ3vqyOCg0GwL1G3VKPLa0IHbhvE0H7N9H12Tdx9q6JhY09STERrPh6Ch4169P3Zd22GZmpSWSlp+DmFwDA6Gm/lvh6i6dPwsm7Bt2efavM4rYQD+PG0U2cXTSDhmM+wqtlH6Xj3JdD308iMbTkhRlva/36LzjULFwMMisplksb53DzwlHUeblYuvpQrcNwPJv3LBqfcOU0h38su9+3vV8j2rxZ8n+3lZEUgYUQQlRJNR3N8LQ1ZuflJMY0dcXu1qzUArWGZafiFE73YLrVsuN4eBrrAuMZ2qhw9opWq2XZ6TgM9VV0rWVf5vHnozP4ZvcNzIz0mD24JnVcLEocV8u55GKypXFhS4qmXlZF21ysjGnkYcmx8DQuxGYUnVOt0bL8dBz6KuhQo/SF5YR4mqjVGj769g8uhYVTs1rZ/bYBNuw6xNZ9x0rcl5efT3hkLK0b16Vv59bF9ltblPzf8YpNu1m6YSduTg4l7hdCPJ1cqvlj5+rN+YNbaT3weSxsCu8p1AX5HN2wSOF0DyagXU/Czh3h1I5/ad77GaDwvunohkXoGxgS0KZHqceWVoSOuHQWANfqtfEJaAqAjZMr2ekphe/dgOeKCuoAuxf/DECDToWLf1Wv36LU1zQ1typzvxBPsxo9nserhEX0spNjubRhDmYO7lh7FC4mmZUUw4GvJ5CbkYxni97YeNUmMfQMZ/75lNSIy9Qd8gYAFi4+NHr24xJf7+qe5aRGXMK1YafHd1EKkCKwEEKIKkmlUvG/Tt68vTaE8UsuMLC+I6aG+uy4nMi1xJyiMZVJ99r2rD+fwM/7I4hMzcXXwZT9ockcvZ7Gi63dcP7PAnjRqbkERWfgbm1MXbfCwuxP+yNQa7S09LYhIjmXiOTcYq/Ro3bZheSSTOnsxasrLvPW6isMbuCEo4URuy4nERidwQst3fCwKd6XWYinzc3EZKZ+M5fj5y6Ua3xUXDxf/LYQI0ND8vKLt4EJuxFNgVpN++YN6du5zT3Pl5uXx/fzV7J43fb7zi6EqPpUKhW9JrzH0pmv8ee7o2nSYyhGJmYEHdhCfETY7UHKhrxP9dv15syO1WxfMJuk2AicvWtw6dgeQs8couPISVg7uBSNTY6LJOJyIHbOHnjUqn9fr6Onb0DPCe+z6tv3+HvaeJr2HIaJuSVXTuzjWtBx6nfoQ51WXR/15QnxVHGq3bzYNq1GzaEfXkXP0IhmE77A0Kyw9V7w6p/ITU+i/sh38Wk7CIBq7YcQbOdC2M7FuNRvh0ONxphY2enMDL7t5oUjpEZexr1Jtyq3mJ8UgYUQQlRZTb2smD2oBvOORrPoRCwGeipaVbdmSAMnPt9+vWiRt8pCT6Xi6wE1+PNIFHtCktkQFI+HrQnvdfWmb13dRwfPRqbzxY7r9KxtT103C7Ly1Fy41TZid0gyu0OSS3qJByoCe9uZMveZ2sw7EsXG8wlk5anxtjPlw+4+9Kwjsw2F2HX4JFO/mYtGo2HCiH78uXxDmePVag1Tv56Dh6sT1T3d2Lz3SLExV65FAODnfe/HpW9ExzHhgy+JjU9iWK9O7Dt+9oGuQwhRtVWv34LR035l3/I5HFozHz19A2o0aUezXiNY//PHRYu8VRYqPT2emfoje5f9xoWjOzmzcy12rp70fXkajboM1Bl748Jp1v/6KfU79L3vIjBA7RadGfvx7xxcPY8j6xeiKSjAzs2LnuPfo2n3oY/oioQQ/3V130qSws7h33di0SxgTUE+ccGHMHf0wLvNQJ3xNbs/R9jOxYQfXItDjcYlnrMgN5uzS77AyNyaeiPeftyX8MRJEVgIIUSVpNVqScoqoJGnFT97Wuns23U5CQA7szsfZg682VRnzE/D/Es8b+8AB3rftZDa3cc+TubG+rzR0Ys3OnqVOe7unGZG+g+ds7T3BAp7Hk/rWf7Vw8WTl5Wdw+y/VnDoVCBxCclYmJvSpG4tXh41gJrV7vw+5ecXsGjddrYfOM61yBjy8vNxsLWmTZN6TB47BHvbwj7SJwIvMv79L/n6/UmEXI9k3c6DpKSl4+ftwZQJIwmoUY1fFq5m894jZOXkUqu6F29PGEm9Wr5A4UzXXuPe5o3nh2FkaMDidTtITEnF09WZkX27MLxP53te09kLIfyxfANnL4aQm5uPj4cLQ3t2ZETfLjoz/YMuh/HzP6u4fC2CjMxsXJ3s6dqmKRNH9sfUpPRe3bevsSxuTg737K0bcj2Slg0DeOuFERga6t+zCPzn8g1cCL3Osh8/Zf6/m0scc+XqDQD8fNyBwn9fUxPjEp9wiEtIwsbSgk/fGE+rxnU59PyUMl9fCPH00Wq1ZKYk4hPQFJ/PdO8Xgg8VPkFwu0UEwLSVp3TGPPvp3BLP26BT/6JWCKUd+zgZm1nQ44V36PHCO2WOKylnSToMf4kOw18qcZ93QBO8A5o8UM4n+Z6Ih1OQm8WFtb9w8+IxclJuYmBigb1fQ2r2egFrd7+icZqCfK7uXU706d2kx4WjKcjDxMoep9otqdV3IiZWhX37b/elbfLCdNKjw7hxdDN5mSlYufpSZ9BkbLzrcGnjXKJObqcgNxtrjxoEDHoNW5/CPtNZiTHs/Hgwtfu/gp6BIVf3riA3PQlzBw+qtR+CT7viixDeLelqIFe2LSD5ahDq/FwsnL3wbj0An/ZDdO4rkq8Hc2njXFKjQijIzsTUzgW3hh2p0XMcBkalP/lXnt67pnYudPtszT2z3paXkcqVLX9h7uiJb5fR/9megiY/Dys3v2L3RIZmlhhZ2JJy41Kp5w3dsZCclHgajv4QIzOrUsdVVlIEFkIIUWWNmB9EgIs5PwytpbN9+6VEAOq6ltwTV4iqaMrMnzkZdIln+nXDx8OF2PgkFq/fzpHT51k390uc7At7N0/54mf2HTvLgK5tGdKzA7l5+Rw+fZ5VW/cRE5/E79N1Z0V8O28Z5mamjBvam7SMTP5auYk3PvuBmtU80Wg0vDiyH8mpGfy9ajOvffo9G//8Cgsz06LjV27ZQ1JKGqP6d8PB1oZNew4z45cFRN9M4M1xw0u9nu0HT/D+rN/wcnNm/LC+mBgbcfBkIDN/W0hwyDWm31p0LTwqlpc+/Bone1teGNoHMzMTjp+7wLwVGwmPiuW7D18r9TWqe7ox8+2JZb6vZib3bncyflhfDA0Lb7uj4oqvVP9fgZfCmLN0HVMmjMTXy73UcZevRWBibMRvi9ayZd9R0jIysbQwo2+n1rwxbjhm/yluN6xdgxU/T79nTiHE0+2nyf3xqFGPsZ/M0dkedKDwyyj3mvc/Q1aIqubkvA9JCDlNtQ7DsHDyIjv5Jtf2riD+0jE6f7QMExvHonGx5w/i2aI3Xq37oynI4+bFY4QfXkdWciytXv1e57wX1vyMgYk5fl3HkJ+dRuj2hRyf+x7W7n5oNRpq9niO3IxUwnYu4ticd+jyfysxNL3T/z/80Fpy05Op1mEYJlb2RJ7YRuDyr8lKiqXOgEmlXk/0md2cmv9/mDt54td9LPqGJty8cISgld+ScuMSjcYWLnaYcTOCIz+/gYmNIzW6jsXAxIyEK6cI2f4PGTcjaDZhZqmvUVbv3dsMjE3L3H+30F2Lyc9Kp/7Id9E3vNMST9/YDID8nMxix2g1avKz01HnlbxIeF5GKmG7l2LpWh3PSrZwXnlJEVgIIUSVpFKp6F3HnjWB8XywPpQWPlaoNXDoagonbqQxuIEjXnbSq1Y8HZJS0zh0KogRfbrwv/Ejirb7V/fixwX/cjE0HCd7Wy5fvcHeo2cYPaA77710Z1bF6AHdGfXmpxw+FURaRiZW/1l0rKBAzaJvp2F+q7CbkZnNP2u2kpWTw9LvP0FPTw8oXMjsr5WbOH/lKi0bBhQdH3MzkX+++YgGtQtnz4zo05mxU6azYNUWBvfogJebc7HrycrJZfpP8/H39WLB1x8VFVhH9e/GrDmLWbxuO706tqR143rsPnKajKxs5s58l7o1C2erD+3ZEX09PSJibpKXn4+RYcmPONvbWper3+693M53L1nZOUz9Zg5N6/kzqn+3MseGXIsgJzePqLh4Pnr1ObRo2X34FEs37CQ45BrzZ00tet3yvr4Q4umlUqlo2Kk/J7etZPms/+HXqDUatZorJ/dxNfAYTXsMx8HdR+mYQigqNz2ZmxeO4tNuMAEDJxdtt/aowcUNv5MScRkXG0dSI0OIDTpA9Y7DqTv0raJx1TsOZ//X44m/eIz8rPSiHrYAGnUB7abMxcCk8B4rPzuTq7uXUpCbTft3/kJ1635KU5BH6I6FpNy4gGOtZkXHZyXF0vatOdhVrweAT7vBHPh2ImG7luDVuh8WjsUXpC3Izebc0llYe9ak7Vtz0LvV8qV6x2Gc/3c2V/euwL1pN5xqtyA2cB8FOZk0Gvsjtt51APBuMwD09MiKj0Kdn6dTjP2v0nrvPqiCvBzCD63F3MkLt4a6T44Zmppj6eZL8tVAspJiMLNzLdoXc24/WnUBao2mxPNeP7QGdV4Oft3GVrq1Y8pL7giFEEJUWa939MLLzoTNwYn8djASAG9bU97t6k2/u3roClGVWZiZYmFmyvYDx6lV3ZOOLRrhYGdD59ZN6Nz6zqOrtap7ceTf34sKt7clpqRhaV5Y5M3MytEpArdr1qCoAAxQzbPwZrtrm2Y657ldzL2ZqNuPul2zBkUFYCgsWI4b2pt3vvyVvUfP8Ozg4h8ajp4+T2p6JuOGNiMjK1tnX88OLVi8bju7Dp2ideN6ODsUznCe/dcKXhzRj8Z1a2JkaMgX77x8z/ctv6CAjMzsMsfo6+lhZWle5pjy+vL3RaSkZTDvy/fL/PBRoFbzwvA+GBoY6BSLe3Voib2tNUvW72DdzgMM7VW1VrQWQjxePca9jb27D+f2rGfnoh8BcHD3oe/LH9GoyyCF0wmhPAMTcwxMzIk+sxsr9xq41G+HiZU9rg064NqgQ9E4a48a9P5mJ+jp6xyfm56EoWnhk4j5OZk6RWDngFZFBWAASxcfAFwbdiwqAAOYOxauBZCTovtkkXNA66ICMICegSF+3UZz6q9pxAUexKLLM8WuJ/7ScfKz0nBtOLbYzFm3Jt24uncFMWf34lS7BSY2TgBcXPcrNbo/i51vQ/QNjWjy3Cf3fN806gLyszPKHKPS0yt3+4WoE9vIz0qndv9XdN6b2/z7TODEHx9w9Je3qDvkDSycfUgMO8v5f7/H0MyqxJnAWq2W6wfWYGLjhHuTqruQoxSBhRBCVFkGeiqGNnRmaMPiMwmFeJoYGRryyZvj+eT7eXz209989tPf+Hl70LZpfQZ2a0d1LzedsVv2H+XI6fPciL5JVFw8SSlpRUVJjVZ39sTtHsG3GegXfuBxvGu7/q2bdK1Gq7O9hk/xhc1uF5IjYm6WeD3Xo2IB+H7+Cr6fv6LEMVE3EwDo3q45h04FsWHXIU4EXsTE2IjGATXp2LIx/bu21WmbcLezF0IeSU/g8th56CRrdxxg6qRnMTI0JDk1HSicQQ2QlpGFmUk6ttaWGOjr89zgXiWeZ/SAbixZv4PDp89LEVgIcV/09A1o3mskzXuNVDqKEBWSvqERDUdP5ezimQQum0XgsllYulbHKaAVXi37FhVuAfQMjIg6tYObl46TFR9JZmI0eenJcPtLXq3u/ZCxpe7izKpbBWQTK4cSt2vvOt7KzbdYXgvnwjyZCZElXk/GzcJFZi+u+4WL634pcUxWUgwAbo06c/PCUSKPbyHhyin0DY2x822IS/12eLboXWY7h6SwwEfaEzj67B70DAxxa9ylxP2uDTrScPSHBK/+kaO//g8o7AdcZ+Bkok/vIi36arFjkq8Hk5NyE9+uo9HTr7ql0qp7ZUIIIYQQokj3ts1o26QeB06c4/Dp8xwPvMjfqzazcM1WZr33Ct3bNScjK5sXP5jFhdDrNKlbi3q1qjOoezsCalZn4eqtbNxzuNh5bxd971bex+gMDIofX6DWlLoP7nzwmTx2MPX9/UocY2Vh9v/s3Xd0FOXXwPHvpvfeOySE3nvvvYTeRBTxByqKvVAVBURfFcUCooiCdGmh994hQCgJkEB6771t9v1jIRDS6BvC/ZzjOWbmmZk7s0Bm79y5T1F8cz6cwBujfThw4jynLl7F7/I1jvtdZtmGHayY/zmW5qal7qNmNTcWz/mk3Pj19UpvJfGwDp46D8Dc35Yx97dlJdaPeGcmAP7b/yl3P9aWFoC6tYQQQgghniynxl2wq9OK2CvHiQ84RcJ1P4L3ruDm/tU0fXUWTk26kp+dyYmf3yElPBBrz0ZYeNTFtVU/LN1rE7x/NRFndpbYr6KsxOMD3k+Vtr2qUAlQdlLz9oP9Wv0mFE00dz/d29W5Wto6NBk7k5q9XyPG/wjx18+SFHyB+MBTBO9fRfuP/kTfxKLUfZi5eNH67Z/KjV9Lt+yH8vcqyMkk8YYfdrVblVs57Na6H87NupMWGQQKBWZOXmjr6nF9x9Kiaup7xfgfBsClafntuJ53kgQWQgghhKjisrJzuH4rHCd7G3p2aEnPDi0BOHspkAlTv+Wvddvo0b4FKzfv5sqNW8x4+1WG9SleRZqQnPpUYgu9XdV7r1vh6qoTDxeHUrdxtle3c9HX16NVJwnCIgABAABJREFU4+JfWlLTMzh14Sr2tupZt6PjEgmNiqFVo7qMHdyLsYN7kZ9fwHd/rmLVlr3sPHySUf1Lv+E3MzUusf+nZdzQPvTr3KbE8r/Xb+e432W+/ngi1hbq6upzl6/x5c9L6d2xFW+MHlhs/M2wSABcHeUNCCGEEOJJKsjNIi0yCCMrR5ybdMO5ibptQGLQeY4veIcbe/7FqUlXbh1aS0pYAA1GfoJHu+KtVHLSEp9KbJm3q3rvlRETAoCxvXup2xhZq98E09LVx7ZWi2Lr8jJTSbh2FkNL9f1EVlIMmfHh2NZsjmfXUXh2HUVhQT5XNizg1uH/iDq3h2odh5V6HD0jsxL7f1RJty5TWJCPbe2WZY6JvXKC/Kw0XJr3LJbcTo8JITs5BteWJd+mSgw6j56JJeauNUusq0okCSyEEEI8gu1XEvh6TwhTunvQp65NxRtUYgWFKiasCsDTxpBpPauVWD//QCgbLsaXsiVM71mNnrXvvr6Wk69k+ZkY9l5LIiEjDztTPbp6WzGmuQMGusWrOncFJLLGL5bQpGz0tLVo6GLCxLYuVLO++zrZO+sCuRBZfg+xBUO8aez6YD3EXlQ3QiMY+9FshvftwvRJrxQtr+Plga6uDtq3q3mT09TX+v4WDRcDgjh7ORAApbL0yTQe1f7jfoRFxRb1DM7Lz+ef9dvR19Ola+umpW7Tpkk9jAwN+HfTbgZ171CsJ++vyzeweus+Zk4eRzUXR/5cs4V1Ow6wYv5M6tdUvyqpq6tDHS8PALS1Sq82ftY83ZzxdHMusfxO9XWjOjWKkt/VXZ2IjEngvx0HGNW/G+am6v6CBUolvyxbj0KhwKd7u2cXvBBCPKKLB3zx/W0WA976nIadB2g6nMdSqCxgyZSx2LnVwOftWWWOO7FlOXuX/ciMdedK2YeSU9tWcH7fJlLiozGxsKZ2q660G/Qahqbm940t4MSWf7mw35e0hBhMrWxp3HUgrQe8XKVfZ9ektKibHP1hIh7tB9NgxMdFy81da6Klo4fW7fupvIwUoGSLhqSbl0gMUr/5U1hY8ERji754iIz48KIJ4JT5eQTtW4GWrl6xfsX3sq3dAm19I24eWINb637FKmsDt/1ByOH1NBz1GSb27tzY9Q+hxzbR/qM/ixKrWjq6mLvVAu62qXjaUsICALC4fdzSRPntI+LMTiyr1cPYRn1vVags4OqmX9DW1cej/eBi41WFStIig7D2avz0Aq8k5F8GIYQQ4gWmLFQxd9ctbsRn4WlTei+v4IRs7Ez1mNCmZIKqvpNJ0f8XKAv5YOMNLkdl0L+eDd52xvhFpPHP6WiCE7KZ29+zqEXAWr9Yfj4cTnVrQ95s50J6rpJ152N5c00gv4+shbuVOpaxLRzpl1XyJjk2PY8/jkfiZK6Pl63Rk7gUVVrDWl60aVKPtdv2k5GZTdN6NcnNy2fL/qPk5Obxyu3J1zq1bMxK3z1M+e53RvTtgomxEVeu32TLvuPoaGtTUKAkIzPricamUMDLH3zFyP5dMTE2YsveowTeDGPKm2OwsbIodRszU2M+m/gSn//0F4PfmsaQXh2xsbTgxPnL7D12lmb1azGga1sAxgzswY5DJ3n78/kM69MZJ3sbwqPjWLN1H/Y2VvTqUHYlSWVlaW7Ke+OG8+3iFYx+bxZDe3dGR1uL7QdPcuXGLSaO9qGed3VNhymEEC+MQqWSzb98Tsyta9i51Shz3PWzh9m/4pcy12/6eQZXju3CsXptuox+m5yMNE7vWM2Nc0d45cslGJtbFo3d+vscLh7wpW7bnrTq/xJhV/3Yv/IXEiJDyk1Ci0dnVa0etrVaEnJkA/nZGVh7NaIwP4/w0ztQ5ufg2UU9+Zp9/fbcPLQOv39m4dF+MLqGJqSEBhB+egdaWtoolQUUZGdWcLSHpICj30+gWoeh6BgaE35qO2kRN6g/7AMMzKxL3UTPyIz6Q9/nwsq5HJw7Bvc2PuibWREfeIboCwew9mqMSwt11Wz1ziOIPLeHU4s+wqPdQAytHMlKiOTW4fUYWNjh9IwmU8uIDQPA0MqxzDGe3UYTdX4/J36ejEeHIWjrGRB5ZhdJty7RaPQUDMyLF/BkJcWizMvB0Kr0N9CqEkkCCyGEEC+ohIw8vtp1C7/w9HLHBcdn09TNtFjFb2nWno/jUlQGkzu6MqyxuqrTp4EtRnohbL2cwOXoTOo7mVBQqGLpySisjXX5bXgtjPXVlQPN3Mx4a20gS09G80UfdQKrubt5ieMoC1VM/u8aetoK5vTzxNRAbmcexPdT3+bv9TvYdeQ0+0+cQ0dbm9peHvzyxfu0b94QgJaN6vDtp2+yZN02Fq7YhJ6uDo52Nrw9djDVXZ14+4v5HPe7TJ0aJSvGH1XPDi3xcnfm3027SMvIomZ1N36a8S6dWzcpd7uBPTrgaGfD0vXb+XfzbvLy8nGyt+GtMYMYO7g3errqXr3VXJ3465spLF7ty+a9R0lKScPS3ITu7Zrz5phBxaqInydjBvbA0c6KZRt2snCFeiIV72quzPvkDfp0aq3h6IQQ4sWRnhTPpp9nEHL5TJljCpVKTvgu48DqhUV9Wu93/dxhrhzbhVvtJoyZ8Rvat3+P1W7djT8/fYl9KxYw4K3PAYi4fomLB3xp1nMYvV//DICm3YdgYGzG2V1radJ9MK41Gz7hMxUAzV+fQ9DeFUSd30eM/2EUWtpYuNWi5RvfYV9X3dbJtmYzmo77kqDdy7m2fQlaOroYWTlQq98ETB08OLXoI+IDTpVbzfqwnJt0w9SxOjcPrCY/OwMzZy+aT/gGxwYdyt3OrXU/DK0cCNr7LzcPrEFZkIuRlSM1+76OZ5fRaOvqAWDq4EHb937j+s6/CTu5nbyMZPSMzXFq3IWafV4vtz/vk3SnylrX0KTMMWaO1Wn77q8EbltM0O5lqAoLMXf1ptVbP2JXu2RbiryM5Ar3WVUoVPdPKSiEEOKFpVAoiP1ttKbDeC487+0gDgclM3vXLZSFKkY0sWf5mRh61bYu0Q4iJi2XYX9d4tWWjoxvXbIS+F6j/r6EQgH/jq2H1j2TWIQl5bA7MJH2nhbUtDcmITOPQX/406mGJV/1Lf6aXN9F57Ew1GXFK/XKPM6dKuLXWzvxSkunRzj7x2P/1soSMzI/bQqFgpygE8/0mE9bZGw8vcd9xIBu7Zj9wf80HY54Agy8Wj/zvxtCVAYKhYLlF8t/oCpK97y3gwg8tZ9NP89EVVhIy34vcWzjXzTo2K9YJW52Rhp/zxhPQsRNvJt3JD0xjuibASXaQWz57UsuHNjMK1/+iVvt4q+lr/n2A4LPH+ejv/ajZ2jE9sVzObdnPZN+3oSVg2vRuJT4aH5+qx9Nug+h74SpT/fkH8DLDU018ntBoVDw0e6n03u3sslKjGbv54NxbdmHxi/P0HQ44iF818P6mf/9kNIZIYQQGpOVp2TR0QhOhaYRn5GHsZ42DZ1NebWlY7FX/POVhaw7H8eBG0mEJuWQr1RhZaxLS3czXm/tjJWxulLifHgak9dfZ1af6txMyGbH1QRSsguobmPIW+1dqW1vxJ8notgTmER2vpIatkZM6uBCHQf1U9/o1FyGL73ExLbO6Gor+O98HElZ+bhYGDCooS0DG9hVeE6XojJYdjqay9EZ5BUU4mppwID6tgxqYFvUCgHgakwGfx6PIighi8xcJfZmenT0suSVFo4leufe6845lsfBVI914xuUOyY4IZumrma82d4FXS0Fy8+UnJwLICg+G4Dqt/v05hYUoqOlQFur+EzFcel5RKTkMqSRXVECODtfiZ62Fm5WBrx+TysJS0NdzAy0CUvOQaVSFV2XlOx8MnOV5bZ3SM0u4O9TUbhY6DOqadV/ZUsIIYSoSF52FvtWLCD4wnHSEuPQNzLBrXYTOgz7H/bud1sTKPPzObV9FVdP7CExMoSC/DxMLKzxbNSGTiPfxMRC/cZPyJWzLP9iIoPf/5q4sCD8D24lKy0FWzdPur/8Hk5edTi4eiGXj+4kLycbew9vuo99H+ca6ge4KXFR/DypP11Gv422rh6nt68iMyUJSwcXmvcaTtMeQys8p/BrFzm6fgkR1/3Jz8vF2smdJt0G06znsGL3U5E3LnNw9W/Eht4gJysDcxtHarfqQvsh49HVL73N1b3nWB5zW0cm/7a13DFxYUFUa9CCbmPeRVtHl2Mb/yoxJjcrA2V+HoPem0u9tj1Z9vmEUveVlqi+F7P38C6xztrRnetnDhEbdgPXmg2JDLqMoal5sQQwgIWtI0ZmlkQFXS43biHEi0mSwEIIITRm5rZgzkekM6SRHW6WBsSl57HuQhxnwlJZMbYeNiZ6t8fd5NjNFHrXsaZ/PVvyCgo5FZrGlssJxKbn8f2g4jfLvx6JwFhPi1HNHEjPUbLibAxTfYPwtDWkUKXuM5uanc/Kc7F85hvEqlfqF7UkAPC9FE9yVgFDGtlhbazL7sBEvt8fRkxaHm+0Kz5h1r0O3Ehi1o5buFjoM6aZA/q6WpwKSWX+gTACYzOZ2kNdZRuenMMHG25gY6LL6GYOGOlq4xeezr9nYghPzmV2P88yj+FuZcj0UiZvu5ehrlaF135Mcwd0tdXjolNzyxwXlKDu/3omLI2FRyOITstDV1tBS3dz3u7ggrOFAQChSepksaOZHhsuxrH6XAzRaXno62jRxduSyR1dMdFX33Zoayl4v5Mbc3aH8OPBMIY0sic7X8mvhyNQKBSMbVF2j69V52JIz1XyUVd39HQqPk8hhBCiqvvvh08JvXKO5r1HYO3kTlpiLKe3r+Km/0ne+nE9pla2ReOunztMw079adJ1EAX5uQRfOMH5fRtJS4hh9PTivWr3LvsRfSMTWvuMJScjjWOb/mbNtx9g7+6NqlBJu8HjyUpP4fjmf1jzzftMWrARfaO7r1P77dlAZloSzXuNwMTShktHdrD9j69JiYui65jJZZ7P1RN72fjTVKwc3Ggz8FV09Q0IPn+cnUu+ITr4KgMmfQFAYnQYK756C1MrO9r4vIKeoREhl89ybONSEqPCGPbRt2Uew8a5Gj7vfFXuddUzKDuJfEfbgeOK2jakxEWVOsbM2o5JCzai0Cr/vkXXQP0QPDcrE33D4q2KstJTAHXrCYC0xDjMrEt/GG5qZUdKXHSFsQshXjySBBZCCKERyVn5nApNY1ADW95qf7eKwcvWiMXHI7kWl4WNiR5B8VkcvZnCsEZ2TO7kVjRuaGN7JqwK4HRoGuk5BcX6whYoVSwaURsjPXViNzNPyRq/WLLzC/ljVO2iStU8pYoVZ2MIiM2kmdvdPlYxaXn8NrwW9W5PejawgS1vrglk9bkY+tWzweV24vNe2flKvtsXiretIb8Or1WUYB3ayJ4FB8NYdyGObjWtaOFuzpHgFDLzlMzv4U1tB/VN/oD6tmhpQVRKLnkFhWUmOK2MdSvszfsg7sRXkeAEdXL3UlQGY5o7YmGow+XoDP67EMelNRksHlkbJ3N90nPV/e22XEogKSufMc0dcbbQ51RIKlsuJ3ArMZtfh9UqOq9W1czpXceaDRfj2XBR/YVGWwHTelajqWvpPcVy8pX4XorH1VKfTjUsSx0jhBBCvEgyU5MJvnCcZj2H0e3ld4uW23t4c2Dlr0TfCsDUypaYkOtcP3uIFn1G0XPcR0XjWvQZxZIpYwm+eIKczHQMjE2L1hUqCxg3Z2lRQjI3K4OTW1eQl5PF6/OWFyU1C/JyOb75H6KCr1Kt/t1+mykJ0Yz76i9caqrfTmrWYxhLp4/jxJZ/adx1EFaOxatYAfJystm+eC4OHrV49aslRQnWFr1Hsmvpd5zevoq67Xri2bA1104fJDc7kzFvz8LJqy4ATboNRqGlRXJMBAX5eejc7md6PxMLaxp06PNI1/xed+Irj5b2g6Vd3Go14trpA1w+upM2PmOLludlZxF8/higvtag/iysHd1K3Y+uvgH5udkPdEwhxItFksBCCCE0wlhPG2M9bfbfSMbL1oi21S2wNtalg5clHbzuJvi8bI3Y9VZj7us+QHJWPia3q3ez8pTFksCtq5kXJYAB3K3USduOXpbFetW6WOgD6gnS7tW6mnlRAhjUCdPRzRz4fPtNjt5MYWSTkpUXZ0LTSMtRMrqZJZl5SuDupB9da1qx7kIch4NSaOFujp2J+gvDoqMRvNzCkQZOJujpaDGzV/UKr1uBspCMvNInFLlDW6F4YpOldfKyxN3KgDHNHIraVHTwsqSuownTtwbzx/FIPu9dnXylup9VRGouf4ysTQ07dTVLRy9LjPW1WX0ulp0BiQyob0tuQSHvrLtGUEI2XWpY0rGGJbkFhWy7ksBXO2+RnFXA8Cb2JWLZHZhEeq6SiW1din2O4vnkbG+L//Z/NB2GEEI81/SNjNE3NObq8T3Yu3vj3awDJpY21GrRmVotOheNc/Dw5pNlh9G6rxo1MzUJg9vVu7lZGcWSwF6N2xWrSLVxUb+JVLtV12JVrVa3k5FpSXHF9l2jSbuiBDCoE6atfcayYf4Urp89RKv+Y0qcz03/k2RnpNJm4CvkZmfAPbnMum17cnr7KgJPHcCzYWvMrNVtuvb9u4C2g1/DrXZjdHT1GDR5doXXTVmQT25WRrljFFraGJo8m8muABp18eHUtpUcXLMQhQJqNu9MVloy+1f+grKgAAAtHfX9nUqlgjLuhRRQ5jpR9RhZOzLgl6o1b4R4eiQJLIQQQiP0dLT4tLs73+wJ5f/2qf+rbm1ISw8z+tS1wcPq7it4utoK9l1L4nRYGpEpuUSn5ZKcVcCd29vC+/rpWxkV//WmfftG2Nq4eLXGnUTi/dtXtyn5+p+bpTqRHJVSeuuE8JQcABYdjWTR0chSx0Snqbft5G3FqdA0dgYk4heRjr6OFg2cTGjvaUGvOtYYltMT+FJUxhPpCfyguta0KnV5Ry9L7Ex0OROaBtxtQVHf0bgoAXzHoAZ2rD4Xy5mwNAbUt2V3YCJBCdkMbGDLh13ci8b1qm3NBxuv8+uRcFp4mBX7MwBwKCgZXW0FXbylClgIIYQA0NHVo9+bM9m68Eu2LZ7DtsVzsHX1xKtxWxp2HoCty90WUjo6elw+voubF0+SHBNOSlwUmalJRQnD+ycoMrYo/uaRlpb6/sTE0ua+5bcTwoWFxZbbuXmViNfGWR1PcmxEqeeTFB0GqBO7+/5dUOqY1NttF+q07kbwheP4H9pGyJWz6Ojp41a7MTWbd6RBx/7ltnMIv3bxifQEfpIMjE0Z8/kiNv40jb3Lf2Lv8p9QKLSo06Y7ddp0Z8ef84qS0noGRuTn5pS6n/y8nKLEvhBC3EuSwEIIITSmcw0rWrqbczIkldOhafiFp7HqXCxr/WL5vHd1OntbkZmr5L0N17gWm0VDZxPqOBjTt64NteyNWeMXw+7ApBL71bm/bPi2B62JKG175e0vRmXt+873ptdbO1HXsfQbb9Pblcs6Wgqm9azGqy0dOXIzhXNh6fhHpXMmLI3VfrH8PrIWFoalv17oZWvE/MElJwy5l57Os6n+sDLWLWoXYXu7f7OVccm47yTfs25XMAfFq/sM961b/EukQqGgX11bzoalczYsrVgSOCtPyfmIdFq6mz+xKmchhBCiKqjTuhtejdpw4/xRbl44SciVs5zwXcbJrSsY/N4c6rTuTm5WBsu/fJPomwG4126Cs1c9GnX2wdGrDqe2rODSke0l9qutU/rv2we9yyitDUKhsqDMdQCq24nkTiPfxLlG/VLHGBqbFe3D5+0vaT/0f1w/c4hbl04TFniBmxdPcnLLv7w29x+MzEp/cGzv7s1LM34rN35dPf1y1z8N1o5uvD5vOQmRIWSlJWPl4IqJpQ0H1ywCKJoIzsLOiYzk+FL3oe4XXPKNKiGEkG9RQgghNCIrT0lwQjaOZnp08baii7e64vRCRDrvbbjOirMxdPa24r8LsQTGZvFRF3d8GtgW20dSVv5TiS08uWS1b1iSutrC1bJkP2AARzP1FwV9Ha1i/YUB0nIKOBeWht3tRGlsWi7hKbk0czNjZBMHRjZxIF9ZyC+HI9hwMY5915IZ0siu1OOYGuiU2P/Tkp2v5M01gdia6PJ/A4snnguUhUQk5+Jsrj7v6jaGGOhoEZJYsiol8nb19J1rdKcfsfL+EmzuViHdv+5ydAb5ShUt3J/da5kvsjP+AYz/bB5vjB7IW2MGaTqchxYZG0/ve3pedmvbjB+mvQPArfAoFvyznjP+AeTk5uHubM+YgT0Z1KNDufuMS0xm6KTpeLm78Nc3U8ode+zcJd6a+T0TR/k89vXbuPswq7fu5WZYFOamxrRoUIfJrw7FwbZ4hV5ETDwL/lnHGf9AsrJzqOPlwaSXB9Osfq2iMb/9u5FFKzeVe7w3Rg+kbo1qvDNrfrFlz+OfAyFeBHnZWcSG3cDC1om6bXpQt00PAEKv+vHvl29wfNM/1GndndM7VhMdfJU+E6bStPuQYvvISEl8KrHdqeq9V0JkCADWTu4l1gFY2DkDoKOnT/UGLYuty05P5dbl00UJztT4aJJiwqlWvwWt+o+hVf8xKPPz2bNsPmd2ruHKsd007z2i1OMYmpiV2L+mJUaHEXrlLN5NO2Dj7AHOHkXrgi8cx8zaHkt79QTFTl518duznrTE2GIJ35T4aLLSkqndquszjl4kXPfj+IJJePceT62+r2s6nIeWlRjN3s8HF/3s2KgzzV+fW2JccsgVjv4wkdZvL8DGu0mJ9bFXTnBj19+khl8DhRaWHnWo2fd/WHs2LDZOVVhIyJH1hB7fQkZcKAotHSxcvfHqPhb7uq0f61zCTmzh1uH1pMfcQs/IDBvvZtQe8AaGlmU/HIm6cJCzf06h26wNGFkXn6Q6LyOVnZ/1KnU7Ixtnun3xHzGXjnL694+LllfWPweSBBZCCKERtxKzeWttYImWAN52RuhpK9C+XXGbmq2uGLm/RcPlqAwuRKh7uSlVJZOJj+NIcDIRKTlFE8DlFRSy6lwMetqKYv2K79XC3QxDXS3WnY+lb12bYtWqS05EsuFiPB93dcfNyoDlZ2LYfCme30fWoo6DumpYV1uLmrfbKDzgnG1PnaGuNrraCk6HpnEpKoP69/RJXn4mhow8JWPrqm+S9HW06OxtyY6riRwKSqbjPddp5bkYgKI2Dq2rmbPGL5b1F+KKVU0rC1VsuhSPAmjuZl4slsBYdfVwTfvirSaEKE+Tut4M7d0JRzt11XlETDwvfzibvPx8Rg/ojoONFdsOnuDzH5eQkJzK/0b0L3U/KpWKGT/8QUpa+f0jAZJT05nxwx8lXqt+FD8sWcPf67fTqlFdPv7faMJj4li5eQ9+V6+zZsEszE3Vf3/iEpMZ98lcsnJyGD2gO1bmpqzZup//TfmGhV99RKvG6gmTurVtiptTyQdMSmUh3y5eSW5ePh1aNMTWyoK5H03gZng0f67Z8tjnIYR4euLCg/l7+ms07TGUPv+7+4DKsVottHX00NK+PX9CWgpQskVDxDV/Qq+eA6BQWf6cAw8r8PQBkqLDiyaAK8jP44TvcnR09Yv1K75X9Yat0DMw4vS2VTTq7FOsJ+/BNYs4u2stfSdOw8bZg6Mbl+K3Zz2vzf0H5xr1AHXfYcfq6odfd879eZGRFM+23+eQMTyBDsMmFC2/fHQnUUFX6PHqh0XL6rXrhd+e9ZzcsoIer35QtPzE5mUANOzU79kFLqoUK89GuLf1wciq5BwomfERnPlzCqrC0v+tiLpwkLNLpmJoYU/Nvv8DVNw8uI7jC96m9ds/YVPjbtL46uZfCd63EusaTajjMwllfh6hxzZxatGHNHppGm6t+j5S/Fc2/ULw3hXY1GxOvcHvkpkQya1D60i6eZEOnyxFz9i8xDbJIVe48G/ZvcRTI28AUL3TcMzdahdbp2Og/m5i7upN47GfkxETwo3dlXfOC0kCCyGE0Ii6jiY0dzNjk388mblKGjqbkqcsZGdAIjn5hYy8PTFYW08L/rsQx1c7bzKooR3GetoExmayKyARbS0FBYUqMnOf7JcWgDfWBDLk9vF2BiRwIz6b9zq5legrfIepgQ7vdnLjmz0hvPLvFfrXs8XKSJczYWkcCkqmkbMJvWqrK/eGNbZj77UkPtkcxMD6tjiY6RGZmsvGi/HYmejS1bv0Prya8G5HN97bcJ2PN91gYANb7E31OBeezqGgZJq6mjLsnorlN9u5cCEinVk7bjKgvi0eVgacuJXK8Vup9K5jTRNX9Re5pq5m9K1rw7YrCSRm5dPJy5JcZSF7ApO4HpfFmOYOJZL+4cnqCmMH02f/aqZ4frk42tGvS9uin5dt2EFaRibfTZlEj/bqGeyH9O7E8Ldn8PvKzYzs1xVT45IPGpZv3MW5y+X34r7j85+WkJaR9dixX7oWzD8bdtC1TTO+nzqpqOdmrepuTPm/31m7/UBR0nrRyk3EJSbz7w8zqF/TE4C+ndsw+M1pzF24nM2/f41CocC7mhve1UrOJv/j0rWkZ2Yx851XqeetnqCyX5e2nPEPkCSwEJWci3d9qjdsxbnd/5GblYFbnSYU5OXif2gb+Xk5RZOveTfrwOkdq9m0YAbNeg5F38iEqKCr+B/ehpa2NoXKggonSntYChQsnfYqzXqPwMDIhIsHtxIbco1er31Soq/wHYYmZvR87WO2LPyS3z8cQeNugzCxsObmxVMEntqHe52mNOioTnC27DuKK0d3svrrd2nSYwgWtk4kx0ZwdudazKztqXO7Kvp54Va7MR71mnNk/RKy01Oxc69BbMh1zu7+j+oNWtKsx7Cise51mlC3bU9ObVtBVnoK7nWaEHLlLJeP7KBx14FlttIQoiLGNk64tihZ9Rpz6SgX/p1NXmZqmdte2/YH2jp6tH1/YVES2bFRFw7MHkXA5oW0/+gPANJjQgjevwr7eu1oMfFbFLf7knu0G8TBr8dwZcNPODftjrau3kPFnhxyheB9K3Fs2Ilm4+cUTWBp7uKN3z9fEHJ0I949Xy22TdiJrVxa9z3KvNJ7bAOk3U4Cu7Xuj5lzyV7nAIYWdri26EXCdT9JAgshhBClmd3Pk5XnYjhwPZkjwSloaymoaWfENz5etK5mAagThl/0qc6/Z2JYejIKXW0F9qb6vN7GGXcrAz7dHMTp0DRq2huXf7CH0MXbiuo2hqz1iyUjV4mXrSFz+3vS3rP8Ccn61rXBwUyPlWdjWHs+lryCQhzN9HmtlRMjm9qjp6O+EXG3MuTnoTX553Q0O64mkJxdgLmBDp1qWPJaK6dK1fO2npMJi0bU4q+TUfheiifn9jm93tqJUU0d0LmnbNnSSJffR9Zm6ckoDt1IZktOAY5m+rzdwYXhjYu/fvVpN3dq2xuz+VI8vxwOR0uhwNPGkJm9qtG9lvX9YZCSrW79YaL/fFX1iMolLCoWgHbN776SqKujQ9umDVi2cSc3w6NoWKv4zf31W2Es+Oc/3h47mB+WrCl3/2u37efI6Yu8PXYIP/297rFiXb/zEACfThx9d9IloEf7FgSFRlLNRV2Fr1QWsv3gSRrVqVGUAAYwMzFmSK+OLFyxCf9rwSXO646rQSH8vX47LRvVYWjv0ivzhBCV29APv+XE5mVcPbGHa2cOoqWljWP12oz87EdqNGkHQLX6LRj83lyOb/qbQ2sXo6Orh7mtA51HvomNczVWz3uPYP8TOHrWruBoD65Om+7YuXlyautKcrLSsXf3Zvgn31Ozeadyt2vUeQDmNg6c2LyM09tWUpCXh7mdIx2HT6RV/5fRuZ0YsnGuxthZf3Bk/RL8D24lMzUJI1MLarfuRsfhE4tVET8PFFpaDPvo/ziy/k8CTx3Ab99GzG0d6TTyDVr1fQlt3eKFCD6TZmHl6Ib/oa1cPbEHcxsHuo6ZTKt+L2noDERVdW7pTCLP7cHEwQPbWi2IPLen1HGZ8eGYOXkVqyI2tnHC1LEaqRF3H6bHB54BlQr3tj5FCWAAXUNjHBp04NbBtaRHBWPh/nD/HoUe9wWg3tD3ihLAAE5NupIefRMTe49i4498/z+Sb13Gsnp9dPSNiA84Vep+0yKDUWhpY2Jfehub50nl+ZYphBDihWOkp83rrZ15vbVzuePu7Rl8vyPvNSv6/8auZsV+vqNPXRv61C1ZcVLWcoBRTR0Y1bTka1AVbdvU1YymrhV/6ahhZ8Tsfp4VjnsWHM31S71ud3jZGjG3f+kJpPtZGunyQRd3PuhS/k2SQqHAp4FtiT7PZbm/J7G465vfV7Bi826Wfz+DhrWLf06LV/vyy7L1/DH3U1o2qkN+fgH/bt7N7iOnuRURTV5+PjaW5rRtWp+3Xx6CtWXJV+Tu6HX7NdSdf39fbPmdPrNL5n1G8wZ3b9YPn77A3+t3cDUohMLCQrzcXXh5UE96d2xV4Tk16PNKhWN2LP0OZ/sH+/Nzh4eLI8f9LhMSHkWdGtWKlodFq5PDdlbFH/Tk5uXx6beLaFjbk7GDepWbBL4VEc13f67i9ZH9aVDr8f9un/EPwMvduaj3b15+PgoU6Orq8O6rd6vBgsMiyMrOoX7N6iX2caeq9/K1m2Umgf9v8Uq0tbSZ+tbYx45ZCKEZ+obGdBr5Jp1GvlnuuHt7Bt9vxrpzRf/vUbdZsZ/vaNh5AA07D3jg5QCtB4yl9YCy/30pa9tq9VtQrX6LMre7w6FaTYZ99G2F454FCzunUq/b/cbOWlzmOgNjU7qPfZ/uY9+vcD/aurp0GvEGnUa88VBxCrj833xuHlxLuw8WY1W9eNX09Z1/E7j1d1q/8zO2NZtRWJDPzYNriPLbT3psKIUFeRiYWWNXuxU1+03AwKzsN/j2zFT30+/+5cZiywO3/cn1HUtoM/nXYn11Yy8fI2jvClLCr0GhElMnTzw7j8S5WfcKz8n37Yr76JbW7/ZBpMeEUKvfBDy7jubG7uVljjOxdycrKRplfi7auuq395T5eWSnxGFgcfeeza1VH2xqNMbIxqnEPvLSkwFQPEI7l4Tr5zB1rF7U+1eZn4dCoUBLR5faA0r++5iVGE394R/h0W4QF1bMKXO/aZE3MLFzQ0tHF5VKhTIvBx19wzLHV2aSBBZCCCGEeI4N6tGeFZt3s3X/8RJJ4K37j+NkZ0OLhurk7Idf/8KhUxfw6daOIb06kpuXz3G/y6zfeYjo+CQWffVRaYd4aMs37eL/Fq+kQS1P3npJ/QVo77EzfPrNQm6FR1c4ydjcjyaUux7A0vzhK7xeG9aXY+cuMWP+n0x9ayz2NlbsOHSSgyfPM6hHBxztileh/7BkDXEJyfw268Ni1bj3yy8o4LNvF+Ll7szEUT6cv/JgrSPKkpefT0RMPB1bNuL0xQB+XLqGy9dvoaWloFn92kx5cwyebuqHZ7EJ6i9LDjYlK+jtrNVJ7ajYhFKPc/SsP+cuX2N43y5FlcVCCCFEVefauj83D64l4szOEkngiDM7MLRywMa7KQBnl0wj5vJRXFv2wa3NAAoL8ogLOEXo8c1kJcfQetKPTySm4AOrubL+Jyw96lGrj3pCsaiLBzj390zSY0MrnGSs8djPKzyGnonFI8XW4eMlaOmU3hLvXvWGvM+p3z/Gb9ksavZ5HQUKrm3/k7z0ZOoOfLtonI6BcaltFbISo4m+eAg9E0tMHUs+3C6PMj+PrMQoHOq1I+H6Oa5u/pWU0ABQaGFTozH1h32IqWO1Ytt0/3JjhedVqCwgPSYEEwd3Ti36mIRrZ1Dm52Jo6YBn19FU7zSs3O0rG0kCCyGEEEI8x7yruVHHy4PdR07z6RsvoXO7csI/MJiQiGjeGD0QhULBtZthHDx5npd8evDpxLuvir7k04PR783i+LlLpGVkYmbyeK1VouMSmb9kDZ1bNeHHGZOLXvMbM7AHH879hcWrN9O7Y0uquZas/rjj3j6+T5KdtSVvvzyYz3/6i3Gf3J3xulOrxkx/u3j18ZEzF1m1ZS9zP5pQIjl8v1+WredWeDRrf/6q6Po/jvTMbFQqFTfDopj0+fcM69OZ8cP7ERwayZ9rtzL2o9ms/ukLXB3tSc9U9x82NCjZN89AX70sOze31OMs37gTHW1txg3t89gxCyGEEM8Lc2cvzF1rEXV+P/WGvo+Wtjo1lhxyhYzYMLx7j0ehUJAacYOYS0eo3mk49Yberc6u3mk4h/9vPPEBp8jPSkfXyPSx4slKiuHqxl9waNCB5v+bV3TvVL3zCM4smcr1nUtxbtoNUwePMvdRWh/fJ+VBEsAAltXq4dl5BNd3LiX6/IGi5bV93sK1Zfn3GgW5WZz9axqFBXnU7P1O0WfyoApyMkClIj3mFicXfoBHu0HU6DGW9Ohb3Ni9jKM/TKTDJ39hbOvyUOeVERdGYUEe6VE3saregKbjviI3I5mQIxu5/N8PZCVFU2/w5IeKVZMkCSyEEEII8Zwb2L09cxcu59jZS3Rs2QiALfuOoVAoGNBN3Q+yZnU3Tvy3qERFa2JKGqbG6lfaMrNyHjsJvPf4WQqUSnp1bElKWvFJhnp3bMW+4+fYf8KP8eUkgZNT0ys8jrmpcbnVuaVZsnYrP/29Djcne8a+NgJrSzPOXb7G6q37mDD1W3754n2MjQxJSk1j5o9L6Nm+RYUJ6TP+AfyzYQdT3nwZD5eyW8g8jPyCAkDdw3jqmy8zsn83ALq2aUYtT3fe/mI+vyzbwDefvolKpd7m3p56d9xZpqDkutDIGE6cv0Kvji0fuq2GEEII8bxza9WXS+u+J+7qSRzqq++Vwk9tB4WiKGFp7lKDPt/tBa3iD3hz05PQNTQBID8n87GTwNEXD6IqVOLcpFuJiddcmnYn5uIhYvwPl5sEzs1IqfA4ekZmxXrlPmmnF39CfMApbGu1UF9DhYIov30EbP6N3LQk6g15t9Tt8rPSObnoQ1JCA3Bq3IVqHYc+9LELb987ZcZHUH/Yh0X7cGzYCXMXb04t+ojArYtpOu7Lh9qvjr4RtfpNxNylBvb17t4Turbsw5H/G8/NA6txbzOg3M+mMpEksBBCCHFbRb1xhais+nRqzXd/rmbbweN0bNmI/IICdh05RbP6tXBxuJvg09PVZcfhk5zwu0xYVByRsfEkpaQVJQsLVYWPHUtoZAwAn36zsMwxZbUnuKPjqLfLXQ8P3xM4Iyub31dtxs7agpXzP8fMVJ3s7tqmGXW8PJj63WL+XLOFd8cNZ+b8JSiVSia9PKREQrqgoIDk1HT09XQpUCqZ9v1iGtWuQY92LYrGZmRmA5CTm0dyajrGRgbo6T5YFQ2Akb66j56OtjZDencqtq5Di0Y42Fpx8vwV9VhD/aJj3S/ndgWwiXHJvnV7j50FoP9TqroWQry4HrQ3rhCa5Ny8B1c2/kzk2d041G9HobKAKL992NRogvE9vWq1dPSIPLeHuMDTZMVHkJkYpe5be+fh652nsY8hMy4cgHN/zyxzTFZidLn72PVZ7wqP86g9gR9EfOBp4gNOYVe3Da3evDt/hEuzHlxYMZebB1ZjV7sldnWKzw2RmRDJqUUfkRETgmPDTjR5ddYjHV9b3wAAhZY27u0GFltnX68thpb2xF8789D7NbJywLvXqyWWa2nr4NFhCBdXfk3CtTOSBBZCCCGEEM+GmakxXVo34eDJ82RmZXPywhVS0jIY1KN90ZiMrGz+N+UbrgaF0LReTerXrM6gHu2p612d5Rt2svXA8Uc6tlKpLPazqlD9ZWjm5HG4lJGktbW2KHefi+d8UuFxbcqZxK40oREx5OTmMbB7+6IE8B19OrXmq1/+4fj5y7w7bjiHT18AYMCET0vs50JAEB1Hvc2Abu3w6daOmPgkYuKTSk1c/71+O3+v385X77+OT/f2JdaXxdTECCNDAwwN9NHVKXm7bmNpwfVb6i+MdxLh8YnJJcbd6Rdsb1Ny0poDJ/2wMDOhVeO6DxyXEEIIUVXoGZnh0KADMf6HKcjJJP7aGfIyU3Ft1a9oTH52Jid+foeU8ECsPRth4VEX11b9sHSvTfD+1USc2flIx1YV3n/vpH4I33DUZ2UmaQ3My3/w3frtnyo8rn45k9g9rtSIG4C6wvp+7m19CDuxhbjA08WSwClhgZxc+AF56cm4t/WhwYiPUWg9WlstXUNTtPWN0NEzKLWVhL6ZNWmRQY+077IYmKnbhRXkZD3R/T5NkgQWQgghhKgCBvZoz87Dpzh46gL7jp/FxMiQrm3uVrav3LybKzduMePtVxnWp3OxbROSU+/fXQna2lpk55TsLXv/ts63K4/NTYxLJBij4xK5GnQLd4Py2yY8jcSkrq76tldZWHq1s0qlQqlUrysrCT1h2rd4V3Plo9dHYWttga2lRaljr90K4/s/V9OvSxsGdG2Hp7vzQ8WqUCioW6Ma5y4HkpSahtU9k+CpVCoiY+NxdrABoJqrI8aGBly+cavEfi5duwlA/ZrFJ1fJyc3jyvVbdG3TtNQksxBCCPEicGvdjyi/vcRcOkr0xUPoGBjj2KhT0fpbh9aSEhZAg5Gf4NGu+KS2OWmJFe5foaWNMi+nxPLctOJvRBndrjzWNTLDtlaLYuuykmJIDQ9EW7/kWz33un+7Z01LRz0PgaqU+6w7y+5NfqeGX+P4z+9QkJ1BrX4T8O417rGOr1AosHSvTcKN8+SmJ6Nvann3+CoVWYnRj1QFHbx/FbcOr6fhyE9KXOP0GPW91719his7uesTQgihUefD05i8/jrjWjryWuuHS5RUBtGpuQxfeqno545elszu51li3NWYDN5aE8j8wd40djUrsd4/Mp2/TkZxIz6LQhU0cDLhtdZO1LQrvz/rqZBUPt50g1dLuX5ZeUr+PRPNgRvJxKXnYW+mR6/aNoxuao+Otlap8ZfGwVSPdeMblDumLHsDE/nvQhzBCdmogGrWBgxtZE/P2sUn2sorKGTluRj2BCYRnZaLvrYWdR2NGdfKibqOJsXGzj8QyoaL8aUeb3rPasX2HZWay+JjkZyPSCM7vxBvOyPGt3amsYu6d1tWnpKev50vGt/I2YSfh9V6pHPVtFaN6uJga8WW/cfwu3yNvp1bY2igX7Q++XZ/3hoexW9ULwYEcfZyIEBRErQ0tlYWXAi4QWxCUlFlaVp6JoduV83e0aV1Uxb8s44la7fSoUVD9PVufylQqZi7cBmHTl3g99kfVzjZ2pPm5e6Mk50Ne46e4Y3RA7G1sihat3H3YXJy82jTRD1DeHlJaLP7ktuljdW+/ffLxcHukRPaA7q144x/AItX+fLZG2OKxZqcms7wvl0A0NXRoXu75vjuO0pAcCi1Pd0BSMvIZOPuw3i6OVHPu3gSODA4FGVhIXVqFJ8lWwhR9YVcOcvyLybSYdgEOg6fqOlwHlpKXBQ/T+pf9HOtll0Z9tG3JcYVKpUs+2IC4YEXKmxNcWLLcvYu+/GBWlgEXzjOyrmT6TD0f498/b4a1rTCMTPWnSv6rMrjXqcpY2ctLvo5NT6ag2sWEnzhBHk5WVg5utOq30s06Kiuzrx+7jBr5t2d4Ox5/XPwpNjWbI6hpT3hp3eQFHQBl+a90NEzKFqfd7vPrplT8Xv7pJuXSAxS3z8WFhaUuX8Dc1uSbvqTnRKHoYWdep9ZacRcPlZsnGODjgT4LuLG7mXY12uDtq76/k2lUnFp7ffEXj5Kq0k/YWT1ZOYeeBrs6rRCoaVNyJH1ODXuXKyi99aR9eoxtVsC6mtw6vdPKMjOoMGIj/FoP/iJxODSojcJ189xfedS6g/7oGh52Ikt5GUk49F+UDlbl87Ezo2shEiC96/CpmbzohZquelJBO9fjZ6pJXb1np/WWpIEFkIIIZ6Ahs4m9K9ni4OZXol1kSk5TN8ajLKMlmFnQtWJXEdzfV5u4Qgq2HAxjjfXBPLTkJrUdzIpdbuU7Hzm7g6htN0qC1VM8Q3iQmQ6vevYUNvemCsxGfx5PJIr0Rl841MDAAsjHab3LD0RtCcwkVOhaXSsYVnq+opsuBjH/ANh1LA1YnxrJxQK2B2YxOxdt4hKzWVcq7v91mbvusWBG8m097RgaCM7UrIL2OQfx9vrrvF/A2vQzO1u4jw4IRs7Uz0mtCn50ODea5WQkcc76wLJyi9kaCM7LAx12egfx/vrr/HdIG+auZmhp6NVdP6zd5WspHyeaGlpMaBrOxav9gVgYPcOxdZ3atmYlb57mPLd74zo2wUTYyOuXL/Jln3H0dHWpqBASUZm2a+zDejWDr8r13lj+v8xom9XsnPz+G/HAcxNjUlKSSsa5+HiwMRRPixcsYnh78xkQNd2GBsZsvfYWU5fvErvjq1o3aTe07kI5dDS0uLzyeN4+4v5jHr3C4b27oSNpTkXA4LYsv8Y1V2deH1Ev4p39Ji27ld/8atowrkBXduy99gZVvruIToukXbNGnAjJJx12w/g6e7MuKF3X7d86+XBHDp1gYnTvuXlQb0wMTJg7bb9JKWkMefDCSUmjQuJUPcVdLKzecJnJ4QQz4Zb7cY07jYYC9vSK/uObvyL8MALFe7n+tnD7F/xywMdMystGd9fv3jsHrA+73xV6vKbF05w6ch2arVUP+Szca5W5tjT21YSfTOAWq26FC1Ljo1k6bRxKAvyaN5rBMbmVvgf3sbmX2aSlZZMq/5jcPCohc87X5EQcYtjG/96rPOoChRaWri27M31nX8D4HpfKwP7+u25eWgdfv/MwqP9YHQNTUgJDSD89A60tLRRKgsoyM4sc/+uLXuTFHyBk7+8h0f7wSjzcwg5ugk9IzN1X+HbTOzdqNn7Na5t/5ND817BtWUfdAyMib5wgITr53Bu2h272pqt9K2IiZ0r3r3HcW3bnxz57n+4NO8JCgUx/oeLzsG+bhsAgvb8S05KHKZOnmjrGxF+umRbDdtaLTC43b7iznrXFr3KjcG1ZR+iLxzk1qF1ZCfHYlenFWlRwYQe3YSpY3W8uo0pd/vS2Ndri0PDjsRcPMTxBW/j1KgzeVmphBzZRH5WGi0mfFPswUFlJ0lgIYQQ4glwNNMvUd0KcOxmCl/vDiE1p+wqgR8OhGFqoMOiEbUxN1T/au5a04rRf19m0dEIfh1eemXqN3tCycgtfb/7ryfhF5HOa62cipKtPg1sMdHTZt2FOM6GpdHMzQxDXe1S4w6Kz+LbiHQaOpvwRruHf8UpPaeAXw+HU8PWiMWjaqOjpU5CDWlkz1trAlh2Opr+9WywMdHjVEgqB24kM7ihHe93divaR5+6Nry6/Ao/HQxj+di7ScPg+GyaupmWGve9lp6MIj4jn0Uja1HHQZ0c7lHLilf+vcL8A2H8O7YuOlqKov0870lggIHd2/PHmi24OzvQsLZXsXUtG9Xh20/fZMm6bSxcsQk9XR0c7Wx4e+xgqrs68fYX8znud7nM6tBBPTqQkZnNuu37+faPlTjYWDG0d2dcHe34cG7xL9BvvjQIT3cXVm7ezR9rtqBSqXBzsufTiS8xol/Xp3b+FWndpB7Lf5jB4lWbWbF5N5nZOdjbWDFmYE8mjvLB1Njoqccw9Tt1xVZFSWCFQsEP095h+cZdbNpzhKNn/bE0N2VEv65MenkwRvdUeTvYWLHs++n8uHQtS9dtA6CWpzvT336VpvVqlth30u0J7ExNnv75CiHE02Bh50yDDn1KXRd54zJH/vsDbV09lPklJ80EdaXwCd9lHFi9sER/1rJsWfgV2ZnpFQ+sQGlxpyXGsfvv77BydGfApC8AMLGwLnVs0PnjRN8KpG7bnrToPbJo+c4l35Cdkcq4OUtx8qwDQJPug1n88WgOrf2d5r1GYGZtR4MOfQi5claSwLe5turH9V3/YGLnilX1+sXW2dZsRtNxXxK0eznXti9BS0cXIysHavWbgKmDB6cWfUR8wCks3Eq/V3dr3Z+C7ExCjm7k8oafMLS0x72tD8Y2LpxdMrXY2Jp9xmPqWI2bB9dxfdc/oFJhbOtCvaHvP7FK2aetZu/xmDpUI/jAagK2LEJVWIiJvTv1h32AR/shRePiA9UTtKVHBXN+WekTwbWZ/GtREvjOmIqSwAqFgub/+5rg/asJP7mFuKsn0DOxwKP9YGr1m4BOBS01ytLstdnqfZ7axuUNP6GjZ4iVZwNq9h6PhXvtR9qnpkgSWAghhHhKZu24yd5rSbhbGdDc3Yy915JKjEnNLsDRTB9vO6OiBDCArYke7lYGXIsrvTJzk38cJ26l8HobZ34/FllifUauEk8bQ/rXL17p19TNjHUX4rgWl1msuvZehSoVX+8JAeCz7h5FCdyH4R+VQZ5SRd+6NsW219FS0LWmFQGxEVyKzqBzDSvOhKmrSH3qF5/wwt5Uj0Yuphy9mUJKdj4WhrrEpOWSkaekmnX5N3HKQhV7riVR38mkKAEMYGqgQ/96tvx1MoqrMZklWk0871wc7bi47e8y1/fs0JKeHVqWus5/+z9F/9+8Qe1iP4P6xnrs4F6MHVzyBvz+sQA92jWnR7vmDxj5s1PHy4MfZ7z7SNuWdp6lKe363XF0zW90GfNgx9fV0eG1YX15bVjJSVbu5+7swPzpkx9ovw+6TyGEeN7kZWexacF0PBu1IS87i9CrJVs8ZGek8feM8SRE3MS7eUfSE+OIvhlQ7n7P7f6PG35H6TzyTfavfLDK4YexY8k8stNTGf7x9+gblt0KLC8nm62LvsLIxJzer39WtDw1IYagC8dp3MWnKAEMoK2jS7eX3yU6+Co5WRkYmz/a211VmbGNMwN+LntyXOcm3XBu0q3UdQN+OVH0/zbeTYr9DOp7J8+uo/DsOqrcbe9watwFp8ZdSiyvTGr1fZ1afV8vc/2DnEPHT5c+1DF7fbuL3VP7VzwQ0NLWoUb3MdTo/nBVv41fnkHjl2c80X1WRpIEFkII8cAWHAxj3YU4Fg6vRb37WhQsOx3FH8ej+HGIN01dzchXFrLufBwHbiQRmpRDvlKFlbEuLd3NeL21M1bGumUeZ9gSf4ASfWj/OhHJ0lPRLBhSvK/u8VsprDobw7U4dT/d6taGDG9iT7eaFc+A2/7HsxWOWTuuPo7m+hWOu19IYjavt3ZiZFMHVpyJLnWMuaEOPwz2LrE8M1dJREouDqYl20uEJeXw6+EIXm7hSF2H0r8oDGpox6CGdiWWX7+dVHYwLft8dlxN5HpcFq+2dMTF4tFeb2rmZsayl+tiZVTyc07JVlcva99+Rf2VFo70qGWNm2XJmFKy84uNDYrPBtSfMUBuQSE6Wgq070tU30rMJju/kDqlXJ9a9urqx4AqmAQWlZtKpWLp+u00qVvy77wQQtxv19LvOL19FeNmL8WlZvF7oiPrl3Bw9W+MmbmQavVboMzP59T2VVw9sYfEyBAK8vMwsbDGs1EbOo18ExOLst+eWfCWuhXO5N+2Flt+aO3vHF63mJe/+B2PuncnGr1x7ggnfJcTfTOAwkIldm5etOz3EvXa9qzwnB6kH+47v27Bws6pwnGl2bX0O3KyMuj3xgw2zJ9S6pjcrAyU+XkMem8u9dr2ZNnnE8rdZ0JkCHuWzafd4Ndw9q5f7thHcevSaa6fOUSDjn1xq9243LHHN/1NelIc/d/6HEOTu/fCoVf9QKXCs/Hdt0xyszPRNzSmRpN21GjS7onHLcSzoFKpCN67AivPhpoOpUqQJLAQQogH1qeuDesuxLE7MLFEEnhXQBIOpno0uT3h1sxtNzl2M4XedazpX8+WvIJCToWmseVyArHpeXw/6MkkQdb6xfLz4XDqOhjzWmv1F4ZDN5KZteMmYUnZFU42V1Y/3HtZGD3ar8vFo2qje3uCqAeVlJlPUEIWS05EkZ2vZEJbj2LrC5SFzNp5k2rWBrzS0olLkRW/lphXUEh0Wi6HgpL553Q0te2N6eBlUerYgkIVf52IxNxQh5eaPfrkE/o6WqVW62bkFrD1cgK62oqi/r2mBjqYGpS8xpejMrgcnUkNW8Oi9UEJ6iT2mbA0Fh6NIDotD11tBS3dzXm7gwvOt5PW8Rnq1z/tSkmi25qol0Wnlf6KqKg6IqLj2Lr/GI52NqW2RdAEY0MD5n36hqbDKFdsQhJn/AO4GV76wyshxLPRqIsPp7ev4tKR7SWSwJcOb8Pc1hGPeuo3Lv774VOunztMw079adJ1EAX5uQRfOMH5fRtJS4hh9PQnU716attKdv/9Pc416hdNKBZwaj8bf5xKYmRIhZOMldXj9l5GZo9WrRpwaj8XDmxm+Cffl5v0NrO2Y9KCjSi0Kr5HUxbks/Gnadi6eNJh6OuEPUCf4Yd1YNWvaOvo0nnUpHLHZaWncHLrv9i6etKwU/GqyIQIdUsrYzNLdv/9PRcPbiEnMx0jM0ta9h1N20HjSvSHF6I0mQlRhJ/eiZGVA9ZejTQdDgA6+kY0fbX0thGVRXZKHAnX/ciICdF0KOWSJLAQQogH5mVrRE07I/bfSGZyJ7ei1/yvRGcQlpzDuJaOKBQKguKzOHozhWGN7Jjc6W6P16GN7ZmwKoDToWmk5xSUmvh7GLFpufx2NIJ21S2Y29+z6OZ2eGN7ZmwN5p/T0XStaYW7VdmtAyrqK/s4HjYBDPDKv1eKKmV96tvSysO82Po/jkcRlpTDkpfqPHCbhm1XEvjhQBgAFoY6fNDFrczYDlxPIi4jn9dbO2Ggq13qmEdVUKhi9q5bpGQXMLKpPZalVAnfkZiZz5c7bwIw/p5EfnCCuhL4UlQGY5o7YmGow+XoDP67EMelNRksHlkbJ3N9MnPV/f0MdEuep76OellO/oP1ABTPL78r1/G7cp1ubZtViiSwQqHg9REP9jqjJgUGhxX1LhZCaI69ew0cq9fm6vE99Bz3EVra6vumyBuXSIwKpcMw9cSPMSHXuX72EC36jKLnuI+Ktm/RZxRLpowl+OIJcjLTMTA2fax4UuOj2bv8R7ybd2T4x98X3Xe17Duadd9/wpH//qRu2x7YOJf9gL2sPr6PKy0xjm2LZtOoiw81m3cqd+yd6/ggDqxeSGJUCP/7duVDbfegwq9dJPLGZRp18cHM2r7csX57NpCfm0Obga+WSOjmZKpba239fTZaWtp0H/s+Onr6nN+7kQOrfiUjOYFe4z954vGLqicp+AJJwRdwbNS5UiSBFQoFNXq+oukwKpQafr3M/saViSSBhRBCPJQ+dW2YfyCMUyGptK1uAcCugEQUQK866v6zXrZG7HqrMffnKJOz8jHRVycWs/KUj50EPhSUgrJQRdeaViUmXutW04rDwSkcCU4pNwl8p91AecwMdNB6BtUTKpWKSe1dMNDV5tjNFDZfiudWYjY/DfFGR1uL8+FprPaL4b1ObrhZPnibhtoOxszt70lcej6rzsXw5ppAvuxbnfaeJSttNlyMQ19Hi8GltJJ4HHkFhczacZNjN1Op72TChDZlV2jHpufxwYbrRKflMaqpfdGfM4BOXpa4WxkwpplDUZK6g5cldR1NmL41mD+OR/J57+rcmbe7tE/tzkcpFTFVl7O97QP37xUldWzZSK6fEJVEw84D2LnkG4IuHMe7aQcA/A9tA4WCBh3VbRwcPLz5ZNlhtO6rbM1MTcLASP3WTW5WxmMngQNPH6BQqaRu255kp6cUW1evXU+unT7AtdMHsRlUdhI4Ky25wuMYmpg/UJXuHSqVCt9fP8fA2ISer35U8QYPKOTKWU76LqfX+E+wdnJ/Yvu919mdawFoPWBsueNUKhXndv+HmbU99dr2KLFeWaC+ny3Iy+WNH9ahZ6C+963bpgdLp4/jzK61NOs1HBtnjyd7AqLKMLJ2LLVPsXgwDvXbPRfXT5LAQgghHkr3mlb8ejicPYFJtK1uQYGykP3Xk2nsYorTPX1zdbUV7LuWxOmwNCJTcolOyyU5q6AoKVeoKn3/DyM8JQdQT8BWlpgKXvnv//vFCo/zqD2BH5ZCoShKpHeqYYm5oQ5r/GLZHZhEe08LZu8Oob6TCZ29LYuS1xl56mrWnIJCUrLzMdLVRk+n+BenWvbG1LJX98Zt72nB2OVXWHAwvEQSOCEzjyvRmXSqYfnYCfp7pWTnM3VLMJeiMqjvZML/+dQosxL5Wmwmn/kGkZCZz9BGdrzV3rXY+q5l9Hnu6GWJnYkuZ0LVlTCGtxPEOQWFJcbm5KuXGes92UpnIYQQ4kmr164Xe5bN5/KRnXg37YCyIJ8rx3fjUacplvZ3H6jq6Ohx+fgubl48SXJMOClxUWSmJhU9+VSpHv/GKzEqFICNP04tc0xKfPltZL4fX/oEW/d62J7AJ7f+y63LZxj+8fcU5OdSkJ8LgFKpLhDISktGoaVdrIduRXIy09n880xcazWkTutuRcnr3KwMAPJzc8hKS0bP0Bgd3ZKtpx6EMj+fG+eO4ORZt8LkbOSNy6QlxtJ6wNhSK5L19NVJ34ad+hclgAEUWlo06TaYyBuXuXXptCSBhXjBSRJYCCHEQzE10KGdpwVHb6aQlafkbFgaqTkF9KlrUzQmM1fJexuucS02i4bOJtRxMKZvXRtq2Ruzxi+G3YFJj3Rs5X3fX+58n/m4q3uxBPS9rMuZgA5gfimTst2vvEnsnqbutaxY4xfLtbgsHM30iEvPIy49r9TE9apzsaw6F8uU7h7FPov72Znq0cjFhGM3U0nNLsDc8O6twLHgVFTwQBPqPaio1Fw+3HidiJRc2lQzZ1af6mW2mTgVksqMbcFk5xcyvrUTr7Z8uElhrIx1i9pFOJqrv5AlZJSs9L7TL9jWVDOfqxBCCPGgDE3MqNm8E9fOHCQ3O5Nb/qfITk+lYRefojG5WRks//JNom8G4F67Cc5e9WjU2QdHrzqc2rKCS0e2P9KxC5XF2ybdSST3nTgNC7vS3+gxtbQtd58vzfitwuOW18+3NDfOHgGVirXfflDq+u/Hd8Pc1rHExHfliQm5RlpiLGmJsaUmrk/4LuOE7zIGvPU5DTsPeKh47wi5cobc7Ezqtqt4Qr1rZw4ClDnW9HYridKunYml+r4wLzvzkeIUQlQdkgQWQgjx0PrWtWH/9WSO3UzhUFAKxnradLxnorH/LsQSGJvFR13c8WlQ/MtAUlbF7Re0tRRk55es4EzMLL6to5k60WdmoEMzt+LVHbFpuVyLy8LVovwK3vu3e9b8wtP4encIgxraMfq+idiy8tTXQF9HgZetUakJ66D4LH49EkHP2tb0qm2Nh7W6TcT0rUEExGSy6tX6JSqDs/IK0VKoq7XvdSEyHS3Fk7smMWm5vLMukLiMfHzq2/J+Zze0y+hjfCoklSlbgihUwdQeHvSuUzKRnZ2v5M01gdia6PJ/A4tfiwJlIRHJuTjffhjgbmmAkZ4WgbElv/AExKiX1XEwftxTFEIIIZ66Rl0GcPX4bq6fPUzgqQPoGxpTu2XnovWnd6wmOvgqfSZMpWn3IcW2zUhJrHD/Wlra5Odml1iekZxQ7Oc71bmGJuZUb9Cy2LrU+Giibwag5+hGee7f7knoPvZ9sm/3xL3XnmXziQu9wUszfkNX7+He6LJ39y41YR0bep29y36kfoe+NOjYF1vX6o8cd+hVPwA8G7aucGzYVT+MzCxxrFar1PXONeoBEB9e8u245JgIgDIT90KIF4ckgYUQQjy0Zm5m2JnqsTMgkYuRGfSoZVWsujP19sRm1W2K9+K9HJXBhQj1a3TKcl5LtDHR5VJUBvEZediaqBO96TkFHL+VUmxcey9LFh+P5N8z0bSuZl404ZdKpWL+wTCO3Uzlh0E1sDd7+q0cHlU1a0OSsvLZ5B+HT31bjG/3TFYWqlh5Vv1KZXtPdXuG0pKzd/K4TmZ6xdY7mOlzKCgF30vxDG18d6IR/8h0/CPTaepqhtF97RACYzNxtTQosfxR5CsLmbolmLiMfF5q5sAb7VzKHBuVmsvM7cEUqmBOP89iPYDvZairja62gtOhaUWtJe5YfiaGjDwlY+s6AqCjrUUnL0t2BiRyPS4LbzsjQP3naNuVBDysDKhtL0lgIYQQlV/1+i0xs7bn0qFthAb4Ub99H3T1795jZaWlAGDn5lVsu4hr/oRePQeUrOq9l6mVLeGBF0lLjMPMWj0nQHZGGjf8jhQbV6tFZw6s/JVjG5dSo0k7dG4nVlUqFTuWfMuNc4d5afqvmNs6PvY5PwxHz9qlLjc0Vt8XPUri2dDErNTttLTV90iW9s6PndCODr6Krr5hhS0aCpVKYkOv416naZlj3Go1wtLeBf9DW2nVf0xRwj4/N5vTO9agZ2CEV+M2jxWvEOL5J0lgIYQQD01LoaB3bWv+Oa1OUva5r2qzracF/12I46udNxnU0A5jPW0CYzPZFZCItpaCgkIVmbllfxnpVduai5EZfLDhOoMa2pGTX4jvpXjM9HVIzro7AZybpQGvtnTir5NRjF9xlV51rDHW0+ZgUDJ+4el0q2lFc3fzp3MRnhBLI13ebO/CTwfDmbgmgP71bFAB+64lERibxcgm9sWSnQ9qbAtHjt1M4ZcjEdxKzKamvTG3ErPxvRSPuaEOH3QpXqmjLFQRlZpL8wqqgIPiswhOyMbTxhAvW6Myx227ksCN+CysjXXxsDZkV0DJSqT6TiY4mevz+7EIsvIKaeJiSkaustSxHbwsMNTV5t2Obry34Tofb7rBwAa22JvqcS48nUNByTR1NWVYo7sT2o1v7cyxW6l8sOE6w5vYY6ynzSb/OJKzC5jWs5pMDCeEEOK5oNDSokGnfhxdvwSARve1H/Bu1oHTO1azacEMmvUcir6RCVFBV/E/vA0tbW0KlQVFvWxL06BjP8ICzrNi9iSa9RxGfm4OfnvWY2Bipu4rfJu1kzvth/2Pw2t/Z/Eno2nYsR/6RsYEnNxPyOUz1G3bk+oNWz2di/Ac8D+sbrvRoEOfBxqfGBWKmY19hZPgpSbEkJ+bg7mNQ5ljFFpa9H/rc1bOeZu/pr5K897D0TMw4sL+zSTFhDHgzZnoGz38/aQQomqRJLAQQohH0qeuDctOR+NqaUC9+5KUTV3N+KJPdf49E8PSk1HoaiuwN9Xn9TbOuFsZ8OnmIE6HplGzjErMvnVtyMhTstk/np8PhWNnqseAejY4WxgwY1twsbHjWjnhYW3A+gtxLLudlHY21+fdjq4MbGhX2u4rnaGN7HEw1WfF2Wj+OB6FAvC0MeTzXtXoVuvh+uLdYWagw6IRtfnrZCSHg1LYdjURKyMdetW2ZlwrJ2xMik9ikpZTQKEKTPTLvzU4HJTM0lPRjGvpWG4S+GxYOqBu4TFn161Sx0zp7oGTuT7nbo/1i0jHLyK91LFrnepjaK5NPScTFo2oxV8no/C9FE9OQSGOZvq83tqJUU0d0Llnwjk7Uz0WDq/FoqMRrDwbA0ANW0M+6upOQ+fHmyFdCCGEeJYadR7A0Q1/Ye3ohkvNBsXWVavfgsHvzeX4pr85tHYxOrp6mNs60Hnkm9g4V2P1vPcI9j9RZsVsoy4+5GZlcG7Penb//T3mNvY07jYYKwdX/vv+k2JjOw6bgK1Ldc7sWM3RDX+hUqmwcnCl57iPaNZz2FM7/+fB5p9nAA+eBM5MS8beveK5Ke5MSqdvXP69i3udJoyb8zeH1i7i5JZ/URYUYO/hzcjPfqRGk3YPFJMQompTqJ7ENKFCCCGqBIVCQexvozUdxnMlOjWX4Usv0au2NdN6VtN0OM/ETwfDsDbWZUzzZ/u659PW/sezNHI24edhpffbu8P+rZVPZJb1h+Hh7kZoWPgzPaYQD8vdzZWQ0DBNhyHEM6dQKFh+sfSHmOLJSomL4udJ/WnQsR8+b8/SdDjFZGekMX9CT6auPKHpUCoUcuUsy7+YSIdhE+g4fOJTO87LDU2f+T0TgIubB5Hhoc/8uEI8DGdXdyLCQp7pMaUSWAghhBAPLCo1l4M3kpnV59EnQhEPTxJrQgghROWlUqk44bsMt1qNNR2KgGeeWBPieSFJYCGEEOIJiE7LZVdAIg5melW61UBCRh7vdnKjQRU5x4JCFfuuJVU8UAghhBCVRkpcJP6Ht2Nh64hb7cqReNUzNGLQu3M0HUa50hLjCLlyloSI0lt1CSGqNkkCCyGEEE/AxcgMLkZm0NHLskongatK8veOvIJCZpfRs1gIIYQQlVNYwHnCAs5Tq2XXSpEEVigUtBv0mqbDqFBMSGBR72IhxItHegILIYQoIj2BhaiYJnoCCyGEqLykJ7AQpdNUT2AhROm0Kh4ihBBCCCGEEEIIIYQQ4nklSWAhhBBCCCGEEEIIIYSowiQJLIQQQgghhBBCCCGEEFWYJIGFEEIIIYQQQgghhBCiCpMksBBCCCGEEEIIIYQQQlRhkgQWQgghhBBCCCGEEEKIKkyhUqlUmg5CCCFE5eDh6kxoRJSmwxCiUnN3cSIkPFLTYQghhKgkXN09iAgL1XQYQlQ6Lm7uhIeGaDoMIcRtkgQWQgjxTKWmpjJnzhzOnz/P119/TbNmzTQdkniO3bx5k88++wxDQ0Pmzp2Ls7OzpkMSQgghXmh5eXn4+Pjw4Ycf0q1bN02H81SpVCpefvll+vbty6hRozQdjhBClEvaQQghhHhmDh8+zIABAzAxMWHz5s2SABaPrXr16qxcuZK2bdsyZMgQ1q1bhzzfFkIIITRn+fLlODs707VrV02H8tQpFAqmT5/OggULSE5O1nQ4QghRLqkEFkII8dRlZGQwb948jh8/zpw5c2jdurWmQxJV0PXr1/nss8+wtrZm9uzZ2NvbazokIYQQ4oUSFxdH//79Wb16NdWqVdN0OM/M7Nmzyc/PZ9asWZoORQghyiSVwEIIIZ6qEydOMGDAAAB8fX0lASyeGm9vb9asWUPDhg0ZOHAgmzZtkqpgIYQQ4hn6v//7P4YNG/ZCJYAB3nnnHfbu3cuVK1c0HYoQQpRJKoGFEEI8FVlZWXz33Xfs27ePL7/8ko4dO2o6JPECuXr1Kp9++imurq7MmjULW1tbTYckhBBCVGlnz57lgw8+YMeOHRgbG2s6nGdu3bp1rF+/npUrV6KlJfV2QojKR/5lEkII8cSdPXsWHx8fMjMz8fX1lQSweObq1KnD+vXrqVGjBgMHDmT79u2aDkkIIYSospRKJV999RWffPLJC5kABhgyZAgFBQX4+vpqOhQhhCiVVAILIYR4YnJycvjxxx/ZunUrX3zxRZWfEVo8H/z9/fn000/x9vbm888/x8rKStMhCSGEEFXKihUr2LFjB8uXL0ehUGg6HI3x9/fnrbfeYseOHZiammo6HCGEKEYqgYUQQjwRFy9eZODAgcTGxuLr6ysJYFFpNGjQgI0bN+Lk5MSAAQPYu3evpkMSQgghqoykpCR++eUXpk+f/kIngEF9z9GxY0d+/fVXTYcihBAlSCWwEEKIx5KXl8fPP//Mhg0bmD59Or1799Z0SEKU6ezZs0ydOpWGDRsyffp0zM3NNR2SEEII8VybOXMmenp6TJ8+XdOhVAqJiYn07duX5cuXU6NGDU2HI4QQRaQSWAghxCO7cuUKQ4YM4ebNm2zevFkSwKLSa9asGZs2bcLMzIz+/ftz6NAhTYckhBBCPLcuX77Mvn37mDx5sqZDqTSsra2ZNGkSs2fPRmruhBCViVQCCyGEeGj5+fksWrSIlStX8tlnnzFgwIAX/vU/8fw5efIkU6dOpXXr1kyZMgUTExNNhySEEEI8NwoLCxk1ahRDhw5l2LBhmg6nUikoKGDQoEFMmjSJXr16aTocIYQApBJYCCHEQ7p27RrDhw/H39+fTZs24ePjIwlg8Vxq1aoVvr6+aGtrM2DAAE6cOKHpkIQQQojnxqZNm1AqlQwZMkTToVQ6Ojo6zJgxg2+++YasrCxNhyOEEIBUAgshhHhABQUFLFmyhKVLl/Lhhx8ydOhQSf6KKuPIkSNMnz6dzp078/HHH2NsbKzpkIQQQohKKz09nd69e/Pbb7/RoEEDTYdTaX344Ye4urry3nvvaToUIYSQSmAhhBAVCw4OZtSoUZw8eZINGzYwbNgwSQCLKqV9+/Zs2bKFnJwcfHx8OHPmjKZDEkIIISqtn3/+mY4dO0oCuAKffPIJq1atIjQ0VNOhCCGEVAILIYQom1KpZNmyZfz+++9MnjyZUaNGSfJXVHn79u3jiy++oHfv3nzwwQcYGBhoOiQhhBCi0rh+/Tpjx45l+/btWFlZaTqcSm/x4sX4+fmxaNEiTYcihHjBSSWwEEKIUoWGhvLyyy+zb98+1q1bx+jRoyUBLF4IXbt2xdfXl4SEBHx8fLhw4YKmQxJCCCEqBZVKxezZs5k0aZIkgB/Qq6++yq1btzh48KCmQxFCvOAkCSyEEKKYwsJCVqxYwYgRI+jRowfLli3D1dVV02EJ8UxZWlryww8/8P777zNp0iS+++478vLyNB2WEEIIoVE7duwgOTmZUaNGaTqU54aenh7Tpk1j7ty5ci8hhNAoSQILIYQoEhERwbhx4/D19WXlypW8+uqraGnJrwrx4urVqxe+vr6EhoYyePBgLl++rOmQhBBCCI3Iysri22+/ZebMmejo6Gg6nOdKhw4d8PLyYunSpZoORQjxApNv9kIIIVCpVKxdu5ahQ4fStm1bVq5cSfXq1TUdlhCVgrW1NQsWLGDixIlMmDCBBQsWSCWPEEKIF87vv/9O06ZNad68uaZDeS5NmTKFv/76i+joaE2HIoR4QcnEcEII8YKLiYlh+vTpJCUlMW/ePLy9vTUdkhCVVmxsLDNnziQ2NpZ58+ZRq1YtTYckhBBCPHWhoaEMHz4cX19f7O3tNR3Oc2vBggXcvHmTH3/8UdOhCCFeQFIJLIQQLyiVSsWmTZsYNGgQjRo1Ys2aNZIAFqIC9vb2LFq0iJdffplXX32VhQsXUlBQoOmwhBBCiKdq7ty5jB8/XhLAj+l///sf/v7+nDx5UtOhCCFeQFIJLIQQL6D4+HhmzpxJREQE33zzDXXq1NF0SEI8d6Kiopg+fTppaWl88803eHp6ajokIYQQ4ok7cOAA8+bNY8uWLejp6Wk6nOfe7t27WbBgARs3bkRXV1fT4QghXiBSCSyEEC+Y7du3M3DgQLy9vVm/fr0kgIV4RE5OTixZsoQhQ4bw0ksvsWTJEpRKpabDEkIIIZ6Y3Nxc5s6dy7Rp0yQB/IR0794dOzs7Vq5cqelQhBAvGKkEFkKIF0RSUhKzZs3ixo0bzJs3jwYNGmg6JCGqjPDwcKZOnUpBQQFff/01Hh4emg5JCCGEeGwLFy7k0qVL/Pbbb5oOpUoJDg7mpZdeYuvWrdjY2Gg6HCHEC0IqgYUQ4gWwZ88eBgwYgJOTExs3bpQEsBBPmKurK//88w+9evVixIgRLF++nMLCQk2HJYQQQjyyqKgo/v77b6ZMmaLpUKocT09PBg0axPfff6/pUIQQLxCpBBZCiCosNTWVr776Cn9/f77++muaNm2q6ZCEqPJu3brFlClT0NPTY+7cubi4uGg6JCGEEOKhvfvuu3h6ejJ58mRNh1IlZWRk0Lt3b37++WcaNWqk6XCEEC8AqQQWQogq6tChQ/Tv3x8LCws2bdokCWAhnpFq1aqxYsUKOnTowNChQ1mzZg3yzF0IIcTz5MSJE1y6dIn//e9/mg6lyjIxMeGjjz7iyy+/lDkFhBDPhFQCCyFEFZOens7XX3/NyZMnmTt3Lq1atdJ0SEK8sIKCgvj000+xsLBgzpw5ODg4aDokIYQQolz5+fkMHDiQ9957j+7du2s6nCpNpVLx0ksv4ePjw4gRIzQdjhCiipNKYCGEqEKOHz/OgAED0NHRwdfXVxLAQmiYl5cXq1evpmnTpgwaNIgNGzZIVbAQQohKbcWKFdjb29OtWzdNh1LlKRQKZsyYwU8//URKSoqmwxFCVHFSCSyEEFVAZmYm//d//8fBgwf56quvaN++vaZDEkLcJzAwkE8++QQnJye+/PJL7OzsNB2SEEIIUUx8fDz9+vVj5cqVeHp6ajqcF8asWbMA+PzzzzUciRCiKpNKYCGEeM6dOXMGHx8fcnNz8fX1lQSwEJVUrVq1+O+//6hVqxYDBw5k69atUhUshBCiUvn+++8ZPHiwJICfsXfffZddu3YREBCg6VCEEFWYVAILIcRzKjs7m/nz57Njxw5mzZpFly5dNB2SEOIB+fv789lnn+Hl5cUXX3yBlZWVpkMSQgjxgvPz8+Pdd99lx44dmJiYaDqcF86aNWvYtGkTK1euRKFQaDocIUQVJJXAQgjxHDp//jwDBw4kMTERX19fSQAL8Zxp0KABGzduxMXFhQEDBrB7925NhySEEOIFplQqmT17Nh9//LEkgDVk6NChRW/2CSHE0yCVwEII8RzJzc1lwYIFbNq0iZkzZ9KzZ09NhySEeEx+fn5MmTKFevXqMWPGDCwsLDQdkhBCiBfM6tWr8fX1ZcWKFVKFqkEXLlzgnXfekWpsIcRTIZXAQgjxnLh8+TKDBw8mLCwMX19fSQALUUU0adKETZs2YWlpSf/+/Tl48KCmQxJCCPECSU5O5qeffmLGjBmSANawRo0a0a5dO3799VdNhyKEqIKkElgIISq5vLw8Fi5cyJo1a5gyZQr9+vWTG3QhqqjTp08zZcoUWrRowdSpUzE1NdV0SEIIIaq4L774AoVCweeff67pUASQkJBAv379WLFihUzQJ4R4oqQSWAghKrHAwECGDx/O1atX2bhxI/3795cEsBBVWIsWLfD19UVPT4/+/ftz9OhRTYckhBCiCrt69Sq7d+/m3Xff1XQo4jYbGxveeOMNZs+ejdTsCSGeJKkEFkKISqigoIA//viDf/75h48//pjBgwdL8leIF8yxY8eYNm0aHTp04JNPPpHegEIIIZ4olUrFqFGjGDx4MMOHD9d0OOIe+fn5DBo0iMmTJ9OjRw9NhyOEqCKkElgIISqZ4OBgRo4cyZkzZ9i4cSNDhgyRBLAQL6C2bduyZcsW8vPz8fHx4dSpU5oOSQghRBWyefNm8vLyGDJkiKZDEffR1dVl+vTpzJs3j+zsbE2HI4SoIqQSWAghKgmlUsnff//NH3/8wXvvvceIESMk+SuEAODAgQPMnDmTnj178uGHH2JoaKjpkIQQQjzHMjIy6NWrF7/88guNGjXSdDiiDO+99x7Vq1dn8uTJmg5FCFEFSCWwEEJUAiEhIbz00kscPHiQdevWMXLkSEkACyGKdO7cmS1btpCcnIyPjw9+fn6aDkkIIcRz7Ndff6V9+/aSAK7kPv30U1asWEF4eLimQxFCVAFSCSyEEBpUWFjIv//+y6+//sqkSZMYM2YMWlryfE4IUbbdu3fz5Zdf0r9/f9577z309fU1HZIQQojnSHBwMC+99BJbt27FxsZG0+GICixatAh/f39+++03TYcihHjOSaZBCCE0JDw8nFdeeYXt27ezevVqxo4dKwlgIUSFevToga+vL5GRkQwaNAh/f39NhySEEOI5oVKpmD17Nm+++aYkgJ8Tr732GkFBQRw+fFjToQghnnOSbRBCiGdMpVKxevVqhg0bRseOHVmxYgXVqlXTdFhCiOeIlZUVP/30E5MmTeKNN95g/vz55OXlaTosIYQQldzu3buJj49n9OjRmg5FPCA9PT2mTp3KnDlz5He9EOKxSDsIIYR4hqKjo5k+fTopKSl88803eHl5aTokIcRzLi4ujpkzZxIVFcU333xD7dq1NR2SEEKISig7O5s+ffrw9ddf06pVK02HIx7SG2+8QZMmTZgwYYKmQxFCPKekElgIIZ4BlUrFhg0bGDx4ME2bNmXNmjWSABZCPBF2dnYsXLiQcePG8dprr/Hrr7+Sn5+v6bCEEEJUMosXL6ZRo0aSAH5OTZ06lSVLlhAbG6vpUIQQzympBBZCiKfsTpVedHQ033zzDbVq1dJ0SEKIKiomJoZp06aRnJzMN998Q40aNTQdkhBCiEogPDycoUOHsnnzZhwcHDQdjnhE8+fPJyIigu+//17ToQghnkNSCSyEEE+JSqVi69atDBw4kNq1a7Nu3TpJAAshnioHBwf+/PNPRowYwcsvv8wff/yBUqnUdFhCCCE0bO7cubz22muSAH7OTZw4ET8/P06fPq3pUIQQzyGpBBZCiKcgMTGRWbNmERwczLx586hfv76mQxJCvGAiIiKYMmUKeXl5zJs3TyagFEKIF9ShQ4eYM2cOW7duRU9PT9PhiMe0Y8cOfvvtNzZu3IiOjo6mwxFCPEekElgIIZ6wXbt24ePjg6urKxs2bJAEsBBCI1xcXPjnn3/o168fI0eO5O+//6awsFDTYQkhhHiG8vLymDNnDtOmTZMEcBXRq1cvrKysWLVqlaZDEUI8Z6QSWAghnpCUlBS++uorLl++zLx582jcuLGmQxJCCABCQkL47LPP0NHR4euvv8bV1VXTIQkhhHgGFi9ejJ+fH4sWLdJ0KOIJCgoKYsyYMWzbtg1ra2tNhyOEeE5IJbAQQjwBBw4coH///lhZWbFp0yZJAAshKhUPDw9WrFhBly5dGDp0KCtXrkTqAIQQomqLiYlhyZIlTJ06VdOhiCfMy8sLHx8ffvjhB02HIoR4jkglsBBCPIb09HTmzp3L6dOn+frrr2nRooWmQxJCiHIFBwfz6aefYmpqypw5c3ByctJ0SEIIIZ6CDz74ADc3N9577z1NhyKegvT0dHr37s1vv/1GgwYNNB2OEOI5IJXAQgjxiI4ePUr//v3R09PD19dXEsBCiOeCp6cnq1evpmXLlgwePJj//vtPqoKFEKKKOXXqFOfPn2fixImaDkU8Jaampnz44Yd8+eWX0vNfCPFApBJYCCEeUkZGBt9++y1Hjhxh9uzZtG3bVtMhCSHEIwkMDOSzzz7Dzs6Or776Cnt7e02HJIQQ4jEVFBQwaNAgJk2aRK9evTQdjniKCgsLGT16NEOGDGHYsGGaDkcIUclJJbAQQjyEU6dO4ePjQ0FBAb6+vpIAFkI812rVqsXatWupV68eAwcOxNfXV6qChRDiObdy5Uqsra3p2bOnpkMRT5mWlhYzZszgxx9/JDU1VdPhCCEqOakEFkKIB5Cdnc3333/P7t27+fLLL+nUqZOmQxJCiCfq8uXLfPbZZ3h4ePDFF19gY2Oj6ZCEEEI8pMTERPr27cu///6Ll5eXpsMRz8jMmTPR09Nj+vTpmg5FCFGJSSWwEEJUwM/PDx8fH1JSUvD19ZUEsBCiSqpXrx4bNmzAw8MDHx8fduzYoemQhBBCPKTvv/+egQMHSgL4BfP++++zbds2AgMDNR2KEKISk0pgIYQoQ25uLj/++CNbtmzh888/p3v37poOSQghnokLFy7w6aefUqdOHWbOnImlpaWmQxJCCFGBixcvMmnSJHbu3ImJiYmmwxHP2KpVq9i6dSv//vsvCoVC0+EIISohqQQWQohS+Pv7M2jQICIjI/H19ZUEsBDihdKoUSM2bdqEnZ0d/fv3Z9++fZoOSQghRDkKCwv58ssv+eijjyQB/IIaPnw4WVlZbNu2TdOhCCEqKakEFkKIe+Tl5fHrr7+ybt06pk2bRp8+feRJuhDihXbmzBmmTJlC06ZNmTZtGmZmZpoOSQghxH3WrVvH+vXrWblyJVpaUuv1ojp37hzvv/8+O3bswNjYWNPhCCEqGfntIIQQtwUEBDB06FCuXbvG5s2b6du3rySAhRAvvObNm7N582aMjIzo378/hw8f1nRIQggh7pGamsr8+fOZMWOGJIBfcE2bNqVVq1YsXLhQ06EIISohqQQWQrzw8vPzWbx4Mf/++y+ffPIJAwcOlOSvEEKU4vjx40ybNo127drx6aefyivHQghRCXz11VcUFBQwa9YsTYciKoG4uDj69+/PqlWrqF69uqbDEUJUIvKYUAjxQrtx4wYjRozAz8+PjRs3MmjQIEkACyFEGdq0acOWLVtQqVQMGDCAEydOaDokIYR4oQUGBrJjxw7ee+89TYciKgk7OzveeOMN5syZg9T8CSHuJZXAQogXklKp5K+//mLJkiV88MEHDBs2TJK/QgjxEA4dOsTMmTPp2rUrH330EUZGRpoOSQghXigqlYoxY8bQr18/Ro0apelwRCWSn5+Pj48PH3zwAd26ddN0OEKISkIqgYUQL5xbt24xevRojhw5wn///cfw4cMlASyEEA+pY8eO+Pr6kpGRgY+PD2fPntV0SEII8ULZunUrWVlZDB8+XNOhiEpGV1eX6dOn8/XXX5OTk6PpcIQQlYRUAgshXhiFhYUsW7aMRYsW8c477zBq1CiZPEMIIZ6AvXv38sUXX9CvXz/ee+89DAwMNB2SEEJUaRkZGfTu3ZuffvqJJk2aaDocUUlNnjwZb29v3n77bU2HIoSoBCT7IYR4IYSHhzN27Fh2797NmjVreOmllyQBLIQQT0i3bt3w9fUlJiaGQYMGcfHiRU2HJIQQVdrChQtp06aNJIBFuT799FOWL19ORESEpkMRQlQCUgkshKjSVCoVq1atYsGCBUycOJGxY8eira2t6bCEEKLK2r59O3PmzGHIkCG8/fbb6OnpaTokIYSoUoKDgxk9ejRbt27F1tZW0+GISu63337j6tWr/PLLL5oORQihYVIGJ4SosqKionjttdfYuHEjK1asYNy4cZIAFkKIp6xPnz5s3ryZoKAghgwZwtWrVzUdkhBCVBkqlYq5c+fyxhtvSAJYPJDx48cTGBjI0aNHNR2KEELDJAkshKhyVCoV//33H4MHD6ZVq1asWrUKT09PTYclhBAvDBsbG3799Vdef/11xo8fzy+//EJ+fr6mwxJCiOfevn37iI6OZsyYMZoORTwn9PX1mTp1KrNnzyYvL0/T4QghNEjaQQghqpTY2FhmzJhBfHw88+bNo2bNmpoOSQghXmixsbFMnz6dxMRE5s2bh7e3t6ZDEkKI51JOTg59+vRhzpw5tG7dWtPhiOeISqVi4sSJtGzZkvHjx2s6HCGEhkglsBCiSlCpVPj6+jJo0CDq16/P2rVrJQEshBCVgL29PYsXL2bUqFGMHTuW33//nYKCAk2HJYQQz50///yTevXqSQJYPDSFQsHUqVNZvHgxsbGxmg5HCKEhUgkshHhu/PPPP3h6etKuXbtiyxMSEvjiiy8IDQ1l3rx51K1bV0MRCiGEKE9kZCTTpk0jKyuLr7/+ukSrnpCQEFasWMG0adM0FKEQQlRO4eHhDB06lI0bN+Lk5KTpcMRz6ocffiAqKorvvvtO06EIITRAKoGFEM+FqKgofvvtN7y8vIot37FjBz4+PlSvXp3169dLAlgIISoxZ2dn/vrrL3x8fBg9ejRLly5FqVQWrXd0dGTfvn2cOnVKg1EKIUTlM2/ePF555RVJAIvHMnHiRM6cOcPZs2c1HYoQQgOkElgI8Vz48MMPcXd3Z/LkyQAkJyfz5ZdfEhAQwDfffEPDhg01HKEQQoiHERYWxpQpU1CpVMybNw83NzcAtm/fzh9//MF///2Htra2hqMUQgjNO3LkCLNmzWLbtm3o6+trOhzxnNu+fTuLFi1iw4YN6OjoaDocIcQzJJXAQohK7+LFi5w+fbpoEoN9+/bRv39/7O3t2bRpkySAhRDiOeTm5sby5cvp0aMHw4YNY8WKFRQWFtK7d2/09fXZvHmzpkMUQgiNy8vLY/bs2UybNk0SwOKJ6N27N+bm5qxZs0bToQghnjGpBBZCVGoqlYpRo0YxbNgwunfvzpw5c/Dz8+Prr7+mWbNmmg5PCCHEE3Dz5k0+++wzDA0NmTt3LgkJCbzzzjvs3LkTIyMjTYcnhBAa8+eff3L69GkWL16s6VBEFXL9+nVeeeUVtm3bhpWVlabDEUI8I1IJLISo1Hbs2EFubi7W1tb0798fY2NjNm/eLAlgIYSoQqpXr87KlStp27YtQ4YM4fr16zRv3pwlS5ZoOjQhhNCY2NhY/vjjD6ZOnarpUEQV4+3tTb9+/Zg/f76mQxFCPENSCSyEqLRyc3Pp1asX3t7e3Lhxgzlz5tC6dWsyMjIIDw8nIiICCwsLmjdvrulQhRBCPIJr164RHByMi4sLLi4uWFpacuPGDT777DOMjY0JDAxk69at2NvbazpUIYR45j788EOcnZ354IMPNB2KqILS0tLo06cPCxcupH79+poORwjxDEgSWAhRaX3++eesXbuWGjVq4ObmRnR0NBEREeTm5uLq6oqzszNdu3Zl2LBhmg5VCCHEIzh8+DD//fcfERERhIWFUVhYiKurK05OTiQlJXHp0iUaN27MihUrNB2qEEI8U2fOnOHjjz9m+/bt0hZHPDXr169nzZo1rF69Gi0teVFciKpOksBCiErrzTffJC8vj2bNmuHq6oqLiwuurq5YWVmhUCg0HZ4QQognLDU1lYiIiKK3Pfz9/YmMjGT9+vWaDk0IIZ4alUpV7N62oKCAwYMH88Ybb9CnTx8NRiaqusLCQkaOHMmIESMYMmRI0fI7aSL5ziVE1SJJYCGEEEIIIYQQQkNefvll5syZg5ubGwArVqxg165d/PPPP5KEE0/dpUuXePPNN9m+fTtmZmYAfPnll7Rp04Zu3bppODohxJOko+kAxLPh4eZCaHikpsMQotJxd3UmJCxC02EIIUSl5eHqTGhElKbDEKLKcXdxIkTuzwUQERFR9Cp+UlISP//8M8uWLZMEsHgm6tevT+fOnfn555+ZNm0aoK4Ajo6O1nBkQognTZLAL4jQ8EgSV3+i6TCEqHSsR36r6RCEEKJSC42IIvZn6b0uxJNm/846TYcgKomUlBQsLCwA+OGHH+jfvz/e3t6aDUq8UN5//3369u3LsGHD8Pb2xtzcnJSUFE2HJYR4wqTztxBCCCGEEEIIoQF5eXnk5eVhbGyMv78/Bw8e5J133im2XoinQaVSFf35srKy4u233+arr75CpVJhbm5OamqqhiMUQjxpkgQWQgghhBBCCCE0IDU1FXNzc1QqFV999RUffPABZmZmJCYm8v777zN58mRNhyiqqJs3b9KpUyc2bdqESqVixIgRpKWlsWPHDiwsLKQSWIgqSJLAQgghhBBCCCGEBtxJAm/cuBEtLS18fHzYtGkT/fv3x8nJiR9//FHTIYoqytPTkz/++IO///6b119/nZiYGGbMmME333yDgYGBVAILUQVJT2AhhBBCCCGEEEIDUlJSMDEx4YcffuDLL79kwoQJJCYm8scff1C3bl1NhyequLp167Ju3Tr++usvhgwZwltvvUWzZs3Yv3+/JIGFqIIkCSyEEEIIIYQQQmhAamoqCQkJuLq6MnXqVMaPH8+4cePQ1dXVdGjiBaGrq8vEiRPp3r07M2bMICsri5CQkKLJCoUQVYe0gxBCCCGEEEIIITTg+vXrREVFoVKpWL16NRMmTJAEsNCI6tWrs3z5coYPH45SqSQuLk7TIQkhnjBJAgshhBBCCCGEEBpQt25dxo8fz6pVq6hWrZqmwxEvOC0tLUaNGsXWrVvp0aOHpsMRQjxh0g5CFPlqwzm2XwgvtkxLASYGutRwMGdoy+p0quNUbP2286HM3nie8Z1q8nqX2mUuq+iYv45rS5Nqtk/2hG7bfDaEeb4XMDXUZctHvdDX1S4xxu9WPJOWHgOgS10n5oxoUWHMACe+HAhA65mbHjieO9vsvxLJtDVnSh3Tv4k7Uwc2fuB9PgvKQhWrjwfh6xdKTEoWVib6dK7jzCsdvDE30is2dsXRG/yy+0qp+5nQpTbjOtUkOjmTwfP3lHtMBwtDNn7Qs9wx3229yPrTt0pd9/mQpvRq6Fru9kIIIR7fnD1h7AxMZsEgTxq7mJQ79q9TMSw9HcuUbq70qW1VbBnABx2dGdTApszth/8TQHRaHo2cjfl5sFfR8vY/X8TBVJd1r9YBYNjfV4lJz3+g+Ne+UhtHM72KB962PSCJr/eGl1hupKuFs7ke3WpaMryhLTraijL38eZ/N7gcnYVPPWs+6uxS6piHua6a9s6GIC5EZpY75t7zmH8wgg2XEksdN727Gz1rWRb9HJWay+ITMZyPzCA7vxBvW0PGt3KgsXPJa3I+MoO/TsVwIz4bLYWC5m6mvNnGEYdSPl/fy4ms908gMjUXcwMdunpb8FoLBwx0H6xO5nBwKv+ejSUkORcDHS3aVzdjQmtHzA3lK5Z4cB06dKBDhw6aDkOIYtzc3Jg/f76mwxBCPGFyhyJKeKWDNx62pgAUKAtJycxj7+UIpqw+zbSBjenXxF3DET6crefDMNLTIT07n72XI+nb2K3c8cevx5KTV4CBXsm/HnkFSg4HRpdY/vmQpsV+Png1ikMB0fg086CRu3WpxwmKSQPgk/4NMbzvWC5WxuXGqAmz1p9lz6VIajlZ8Ea3OqRn57PuVDDHrsewaHx7LI31i8YGxaahr6vNZwMaldhPDQdzACyM9Utctzt2XQznZFAcne976FCa4Ng07M0NeaNbnRLrGrhZPeDZCSGEqCwOBKWUmQS+EpNJdFreA+3nnfbOZOcXFv3sH5WB75UkOlQ3p4OnebGxFoYlHxA/iHv3VahSkZmnxC8ig4XHormVmMO07qXfc4Ql53I5OgtDXS32XEvmrbaOGOk9WgyVxdhm9vSrU1BieWx6Hn+cjMHJTA8vW8Oi5cGJOdiZ6DKhtWOJbeo7GhX9f0JGPu9sCCYrv5ChDWywMNRh46UE3t8YzHcDqtPMzbRo7NnwdD72vYWLhR6vtnAgt6CQNefjuRiVwZ/DvbExufuK/dLTMfx1KpaWbqYMrG/NrcQc1pyP51pcNvMHVkdLUXYCH2BnQBJz9oZT18GICa0dSMgsYN2FeC5FZ/H7cC8MSyk6EEIIIYTQJEkCixJaeNqWqMod2qo6IxfsZdHeq/Rt7IaighvjyiIkPp3L4UmMbe/N2pPBbDpzq9wksIuVMRFJmRy/EUuXus4l1p+4EUtGTgGWxvokZ+YWLb+/2jQiMYNDAdHUd7UssxI1KDYVM0NdBjV/Oq99pWXnYWb44FVN5Tl6LYY9lyJp5G7NglfaoqujrpDpUteJVxcd5NfdV5g+qEnR+KCYVDxsTMqtwjXU0yl1/Y2YVPxCEmjkbs1b3SueETkoNpVm1W2l4lcIIaoAF3M9LkZlkpSVj5VRyZ6Y+66nYGmoQ3J2yWTj/e5P9CpVKnyvJOFpY1CsyvRxlLavoQ1tmb49hJ2ByYxuakc1K4MS222/qq6AHd3EjiWnYthzPQWfeqU/NH6a0nOVmOo/mWRl83uSsXcoC1VM3hCMnraCOX08ih0rOCGbpq6mFX4WS0/HEJ+Rz6JhNajjoE4O96hlySsrrjH/UCT/jqmJQqGgUKXix0ORWBrq8NvQGkXHau5qyhvrbrD0dAwfd1HfK8Sm57HsTBxtPMyY18+j6L7W0UyP345Fs/9GCt28y44rK0/JL0ej8LY1ZMFgT/S01fdFNe0MmbkjlP8uJvByM/uHuHqa4+HmSmh4hKbDEFWEu6sLIWEl35KoClzdPYgIC9V0GEJUyMXNnfDQEE2HISopSQKLB2Kgq019Vyv2Xo4kOTMPKxP9ijeqBLadDwOgjbc9IQnpHA6IJigmFS8H81LHt6/lwPrTt9h/JarUJPDeS5FUtzPF3EivWBL4UQTFpFHdzuyx9nG/AmUhhwKi2XjmFsYGunwzquUT2e+hgCgA3uhWpygBDODlYE5bb3t2+0fwQZ8GGOnrUKAsJDQhg271S16/ihQWqpiz0Q+AqQMbo6Nd/uuY0SlZZOQUPPHrKIQQQjM617Bg+dk4DgWnMqh+8WrgQpWKA0EpdPYyL7ONQGXRzNWEQ8Gp3ErMKZEEVhaq2HUtGRdzPQbVt+bvMzFsvpz4zJLAOfmF7LuRzKZLidR3NGZyh4f/ff2g1vsn4B+dyeutHIpVAcek5ZGRV1hqgvxeykIVe66nUN/RuCgBDGCqr03/ulb8dTqWq7FZ1HUwJiA2i9DkXF5pblcs2VzHwYhGzsbsu5HCex2d0dXWYu/1FAoKVQxrZFOssGFwAxv+PBnDjoDkcpPAJ0LSSM1R8kYb66IEMEBnLwsczaLZGZD83CSBQ8MjSD++QtNhiCrCtM1Lmg7hqYkIC2WJX5qmwxCiQuObyHdjUTZJAosHFpWchbmRXon+r5WVslDFzovhmBjoUNfFki51nTgcEM3GMyF83L9hqdsY6enQqoY9x6/HkJOvxOCeV/ly8go4dj2Gse29OR38eDOlZuTkE52SRWtv9ReEAmUhhSoVejqPVo0Tl5rNprMhbPELJSE9B3tzQ8Z3rgXwQL134W6v4rL2D+DlUPIXiqu1CYcDYwiKTaWBmzUh8enkKwuLErO5+Uq0tRQVJnQBtl0I41p0Kq91qomrdcW9D4NiUgGKjpWTr0RXWwttreejUl0IIURxTVxM2HIliYNBJZPAFyMzScgsoKu3ZaVPAt9pWeFqUfKh+anQdBIyCxhU3xpzQx2aOJtwJjyDqzFZxRKdT1pYcg6bLieyIyCZjFwlbhb6NHVV/649H5HB5I3B5W5/b7/lB5GaXcDfp2NxsdBjVJPib5gFJajvK6pbq5PAuQWF6GgpSvz+vpWUQ3Z+YanXpZa9elnAPUlggNr2JVtq1bI34nxkJqHJuXjZGHI1Rj22jn3x/erraOFpbVC0r7Jcvb2+1LjsDDkQlEpGrhKTJ1RlLYQQQgjxJEgSWJSQkVNAyu0q10KVipSsPLacC+VqZDKfDmj03CTYTtyIJSE9hz6NXNHR1qJ9TUf0dbXZ6R/OpB51MdIv/Y9/t3rOHA6I5sT1GDrfUw185FoM2XlKutV3eewk8J3kZXxaNq8tOsj1mFQKVSpqOVrwRvc6tPC0q3AfKpWK08HxbDhzi2PXYgBoXcMen2YetKlhj9btz6m83rsP6k7P4szcAoz1i7+em5Kl/qKbkJ4DqNs53DnHUT/vIzQhHS2FgoZu1rzTqx61nCxKPUaBspA/9wdiYaTHy+1qPFBcd451JjiOX3dfITolC11tLVrVsGNyr/qVsreyEEKIsmkrFHT0NGfrlUSSs/KxvKclxL4bKdiZ6BbrF6tpOQWFpNxuTaECsvOUnAnPYL1/At28LahxT/XrHdsDkgB11ShAlxoWnAnPYPPlBOo4lD9vwcMqKFRx9GYqmy4lci4iAz1t9fXtX8+62KRq7lb6TC+jf/Edhg84Wdodq/ziSM9V8lFnl2LVsgBBCep7hjPh6Sw8Hk10Wh66WgpaupvydnsnnM3VyfP4DPXEfnYmJVuD2Bqrl91JuMfdHmtfztiYtDy8bAyJz8jDRF+71D7MNia6BMRlk5mnxLiMPs134ypZGHGn73BMeh5e+iU/fyGEEEIITZEksCjh01WnSl3esbbjc9V3daufumdTt/rqGbeN9HVo423PgStR7LkUgU8zj1K3a1fTAQNdbfZfiSqWBN57KYLazhZPJLEYFKt+lehiaCKj29ZgXKeahCdmsOJYEO8vO87ckS3oWLvsSdFO3ojl+23+RCRl4mBuyLiONenf1B07s5JfNsrqvfswGrpZcyggmt3+EYy5J0GblVvAyRvq2dxz85XFzu1CaCKj2njhZGnEjZhUVhwLYuKfh/nttfbUdSn5iuW+K5HEpWUzoUvtUiflK82dY/mHJTG2gzeWxnpcCkti7cmbXAo7xJKJHXGylESwEEI8T7rUsGDz5UQOBacy8HY1cEGhioNBKfSpY1Wp5iVY5RfPKr/4EstdzPV4s23JCc9Ssws4fisNG2MdGjqrfz919LLgh4OR7L+Ryjvtn1z16K7AZBYdjyIhswB3S30mtXOidy1LzA1L/o61MtJ9Yn2SQd1ywvdKEq4W+nTyKtmCKzhRXQl8KTqTMU3tsDDU4XJMJv9dSOBS9A0WD6+Bk7k+mXnqewuDUhLQ+reX5dye/C8zt5yxt1tZ5RSox2bkFWKoU3pS2+D28uz8wjKTwBnlxHVn+5x7JiUUQgghhKgMJAksSninZ92inrkqFaTn5HMxNIFNZ0MY//shfh3XFgvjyt0TOCUzl2PXY7Aw0qN59buvIPao78KBK1FsPHOrzCSwoZ4Orb3tOXZPS4iMnHxOBsXxRtcHfw2yPDUdLXi1oze9GrribnN3IpXOdZ0Z88t+vt/mT/uajkXVvPe7HJ5ERFImLlbGTBvYmEYepc+iDuo+u2nZFc+kXt5n2r+pO2tOBvPH/gAUQIfajqRk5rFw7xXyleovOXfaPTSrbouuthbDW1XHykT9mmf7Wo60qmHPhD8O8+OOS/zxvw4ljrH+1C30dbUZ2vLBJ8rrUscJDxtTxravUZQ47ljbiXquVkxZfZpFewP4clizB96fEEIIzWvkbIyVkQ4Hgu4mgc+Fp/8/e2cZHtW1tuF7ZjJxdxciJCEBggR3l+La0lKlpa6nPfWeyvlqp0qFQoW2QPHiVjxAcIkSd3efjHw/JpkwZCJYKe26r6tXyVrvWnvtyUxm72e/63mpqFcxKtD21i7uCsZ1tWP8ZeJpbaOatNJ61p8vZuGviXw0tQvdXFseRu5OLKNRrWFEgC3SJjHbykRGpI8VUWmV7EwoZVYPp1bHuRZOZWltJ0KczfjXKC8CHNvOSlWqNDphsy1kErAy7dytw+7EMqoaVDw80E13npcz3N8GHztTFvR21gmpQ/1t6OZizqs7MvjueD5vjPNBo9HGG7oaam5rfiig0f1sIPaKNg0ag3GXT9zuo4ZOrau9CQQCgUAgEAj+fIQILGhFsLstvfz0b0BGh3ng42jFx9su8MPBRJ6Z2P0Wra5z7DyfhVKloZefI0WVdbp2XycrjI2kJOZVEJdTRqiH4ayX0WEe7I/N5XhSAcND3TkYn0ujSs2oayh2Zohwb3vCve1btbvZmjM0xJWd57NJK6rC38Wwqfu0vn7aKuenM1j8/RG6OFszra8vE3p4YWmqvw2yoKL2uj2BLU3lfHHvIN5Ye4ovd8fy5e5YpBIYFebB6DBPPtx6Hmsz7ZbIAYEuDAhsXQwl1MOOME97LmSVUNPQqGcrUVxVT0x2KSNC3bEy67zndHOW95UMD3XH2drsum07BAKBQPDnI5VIGO5vw+8xLZYQey+V42lrTFfnv44VBIC7jTF9vK302ob62zDYz5r7V1/io/3Z/DC/q66v2QoizM1CZ2MA0NPdgqi0SjbH3DgR+N5IF6xMZOyIL+W+VZcIczVnSpgDIwNtdZmxzVzMq7mhnsAHUyqQSyWMNJAFDDCqjaJrwwJscbbM5WRmFdBiQdGcwXs5zW0WxlL9WAMZuA1NsZZNmb1mchmV9YYfkDc0ahXe9jKyzZpqRjQo1a0sJVrWJfyABQKBQCAQ/LUQIrCg04zv4cXH2y5wNr34Vi+lQ7adzQRgX2wu+2JzDcZsPJnWpgg8MMgVM2MZ+2JzGB7qzt6LOfTwdjBot3Cjac6erW1QthnjaGXKw6NCeWB4MPvjclkfncb/tl3gq92xjA73YFofP53lgr2lKZ8tHHjd6/JysOT7R4aTUVxFeY0CT3sLHKxM+W5fPECnbDLsrUzQaLSeiZeLwIcT8tBoYEwbou614GBporOLEAgEAsHtxYhAWzZcLOFQaiUTQuw4klpxw8TRPwN/RzP8Hc1IKqqjqkGFlYmMxMJanRfuGzszDI5LK63nfE41PTw6Lo7aER42Jjw51IOHBriyJ7GcjReLeW9vFl8czmV8sB1TwhzwtddecwQ4mvLJ1C7tzmfchn3CldQqVJzNrqafj1WnM4cvx97ciJSm18nNWvtguLim9TVRsy+vU5MHb0tsI/5XZD0XXhHrbm1MUlEdDUp1K0G8qFqBtamsVfvlXH4s7yvE3qLqRiSAo0Vrb2KBQCAQCASCW4kQgQWdRt20J8/Qtr6/Egm55SQXVOJpb8Hj47q16q+oVfDf38+x92IOT40Pb5U5C2AqlzG4qytRiQUUVtZxMrWIZ29g9vPra08Rm13KikdHtCq0ll5YhUQCHp0QVY1kUsaEezIm3JPkggrWR6ex60IWW89kMrWPLy9N6YmJXNapQnPtkVVSzZm0YgZ3dcXH0Qqfy9wnjicV4mJjplvv4uWHqWlo5KfFI1r5NqYXVmFhYoTdFdYTZ9NLkEqgr3/nb/DrFEoWfXcIJ2sz/nf3AL0+pUpNVmm1KAwnEAgEtynd3S1wtDBif1I5DuZGVCvUjAqyvdXLuirU6ubrJu3PzVnAU7rZ08+n9U6f/cnl7L1Uzu+xJTdEBG7GTC5jSpgDU8IcuJBbw8aLxWy6WMLa88U81N+Ve/q6YGVq1Cqj+VqJya+lUa0hso356hpVLF6bjJOlnA+n6AvPSpWG7PIGPGy1IquPnSnmcikJBbWt5olvagt10X7Xh7hos8QTCutavb7xBbWYyaU60TvY2YyDKRUkFtbS3b3ltW5QqkkpqaeXZ/uvf7CLWdO8dXjbmer1JRTW4W1ncsO8nQUCgUAgEAhuFEIEFnSa7U3ZtdcrKN5stjQVhJvR16/N4mo7z2dxNr2EHeezmN3PcObLyG4e7LmYw0dbzwMwolvbhdquFidrU3LLalkfncY9Q4N07WfSijmWXMCAQBfsLa/OdznAxYYXp/TksbHd2H4uk6LK+hu23qLKev5v8zkeHBHMAyOCde27L2QTl1PG0xPCdW12liacyyhhb0yOXmbv9nOZpBVVMTPSD9kVXscJuWV4O1q1EsTbw8zYCLlMSnRyARcyS+ju7aDr++nQJarrldw79MZWWRcIBALBn4NUImF4gC0bLxRjbCQhwNFUJ+DdDiQW1pJaUk+oizkWxjIUKjV7L5VjJJVwfz9XHAxkiQY4mrIvqZyDyRVUDFEaLOB2vXR3t6C7uwVlQxrZEluKXHbjH+w3C7ZtWXeYyWXIZRJOZFZxMa+GcLeWB7Y/ny6gWqHmnhCtZZaRTMLwABt2JpRxqaiWICftnFUNKrbFleJrb0JIkyAb6mKOu7UxW2NLmNXDUWfHEJdfy/mcGqaGO+iuP0YG2rLseD5rzhXricAbLhSjUGmYENLasutyBvpaY2EsZf2FYkYF2mLU9DruTy4nr1LBIwNbFwUUdI7e975F764+LP33vX/q2Kultl7Bss2H2HMilpKKajyd7Zg3ph8zhve+pvm+WLuXH7dF8e2LC+kT4qvX99QnKzlyPsnguGUv30dEkDebD5/jreW/t3uMyYN68NZD065pfYLbm/fn9cErpBd3vrH0Tx17tSjqazm6YTkJx/ZQU16CrYsHvcfPo+foGdc038FVX3L89x+Z/9o3eHdru06MWq1i1VsPk514jhdXn9Lr02g0fHr/cBR1Na3GSaQy/rUy+prWJhDcKoQILGjFiZQiCi8TEBVKFadSi9kXm4ObrTl3DQ7s1DxRlwooqW4w2Dd/YADeji0X3auOprDnYo7B2Ben9Oz02hVKFXsuZmNsJGViRNsC4NwB/pxNL2HTyfQ2ReABgS6YmxhxOCGffgHOrbJXr4d7hgRxMD6Pb/+II7Okmm6edqQXVbHxZDpOVma8MLnHNc9taSpnTn//G7ZWgJ4+DvT2c+THQ4lU1inwd7EmKb+CjSfTifR3YkbflmJui0eHcjatmLc3nOFiZil+zlbE5ZSx/Wwm/i7WPDJa309QpdaQU1rT4cOF5PwKkgsqCXCx1hUufGZid574KYpnfz7GjEg/XG3MOZVaxP64XPp0cWLugBv7OggEAoGgfX47V8QfSeUG+54fcXWWPyMDbVl3vphj6VU8PMD1BqzuxpNSXM+uhDLdzyqNhpTiOrbHlyGXSXh8iPYB8pHUSirrVYwMsDEoAAO425gw0NeaI2mVbI8vZX6vlu/FG/m6AtiZy7mnb2v//htBVrn22s/Vuu0Hu08N9eDpTSm8sDmVaeGOuFjJOZ1VzcGUCnp7WjL7MuuPB/q7EpVWybObUpnT0wkLYxmbYoopq1Xyypguul1HEomEp4d58NLWNB5dl8zUMAeqG1SsPluEk6WchZedr7uNCfN6OfPr6UJe3JLG4C7WJBfVsSmmhEhvK4Zf5mVc16jiUEolZnIpQ/217ebGMhYPcuej/dk8uTGZ8cH25FUqWHOuiEBHU2Z0b3kwLbg63l40HXvra9vJdT1jrwa1WsPzX/zGibhUZgzrTbCvGwfOJPDuj1spKqvi4enDr2q+0wnprNh+tM3+pKwCgn3cuGtc/1Z9vq7a91qvrj68vWi6wfFfrd9HfmkFI3oHG+wX/P2Z/Nh/MLdp/+HWzRh7NWjUajZ8/DwZMSfpOWo6Lr7BJJ06wK5l71FdVsTg2Q9f1XyZcaeJ3ryiU7HHNv5AduI5g30VhTko6moIHTyBLj30d59KJJ2zSRII/koIEVjQip8OXdL72VQuw9XWnNn9unD3kCBszDtXuCsht5yE3HKDfWPCPfRE4COJ+W3OczUi8KH4PKrqGpnY06vddQ7p6oaHnTmphZWczygxGGMilzE0WFukbcwNKgjXjI25McseGsp3+xM4nJDHzvNZ2FmYMLGnFw+ODMHR6q+V7SSVSvi/+f34/kAiB+Nz+f10Bq42Zjw0MoR5A/yRX+ab5+VgyfcPD2Ppvnj2xuRQVa/AycqMeQMDuG9Y11b2GxW1CtQasDJrPwv4QFwuyw8k8sDwrjoRONzbXvc6bjqVTr1ChZudOYtGhnDX4ACMZOKLWSAQCP5MotLa9mK/WrEyzNUcZ0s5hdWNbRYSu9UcSq3gUGqF7me5VIK9hRH9fay4s5czgU7aLNXtcVoriGnhjgbnaWZOTyeOpFWyObaEeREtQuiNfF1vNuV1Wv/e9uwQwtws+GZ2IN9H57M5poT6RjVu1sY82N+V+b2cdJm1AM6Wxnw9O5BvovJYeUZb8DXQyYznh3u2ss0Y4GvNB3f48eOJApYcycXCWEaktxWPDHRr5dH78ABXHMyN2HSxhE8O5OBoKWd+L2cW9nXWsz4rr1Pxzp5MXK3kOhEYYGqYA+ZyKSvPFPLZoRxsTI2YEGLPg/1ddYXjBFfPxIHXbr92PWOvht0nYoiOTeWpuWO4Z4K27sb0Yb145tPVfL/1MFOG9MTN0bZTc1XV1PP6d5swkklRKFWt+itr6igorWR039B2z8/T2Q5P59Z/J9ftO0VeSQX3TR7M8F5CBP6n0m3IxFsy9mqIP7abjIsnGH7XU/S7424AeoyazvoPn+XYph8IHz4FG6fO7bKor6li21dvIDUyQtVouAhoM7nJMRzd8B0yubHB2MIMbRZ+yIAxBPQeepVnJRD89ZBoNE1Gr4K/NRKJhJLV/7rVyxAI/nI4zPsA8WdQIBAI2kYikVDwxexbvQyB4G+HyxNrb/k1iEQioeror7d0DbcbT3z8Kyfj09i/5EXMLrMyO52QzqL/+4nHZ43ivsmDOzXXy1+v51xSJiN7h7BqT3QrO4jmOd94YApThkRc1ToLyyqZ8dKXuNjbsPrtR5Ab3fyHE1YD77rl7+mbhUQiYfkZUXT6ZrHmv0+SGXuSp5bvR27SkhCVGXeaVf95mGHzH6f/1Hs7Ndfmz18mO+E8Qf1GcnrHqjbtIBT1tfz40l3Yu/uiqKshK/5MKzuII+uWErVuKQ9//ju2zjc2Mexm8UAv67/t51Bw/YhMYIFAIBAIBAKBQCAQ3DQuJGfx7aaDxKRkAzAwPIA7x/Xn3reXs2jqMJ2FwpW+vov++yPl1bW8vWgGn6/dy4XkLCRARFcfnpwzGn+PFtuUzngCf7vxAEt/P9juWjvyzo1JzSbA01lPAAYI9dMKRLFphi3urmRb1Hl2n4jhqxfu5mxipsGYS1kFALrzrGtQYCKXI5V27Of95do/qGto5MW7J/4pArDgzyfn0gWOrFtKXnIMAH49BtB34l38/Nq9DJr5kM5C4Upf35VvLaKuqpzJj7/NgZVfkHvpAkgkeAVHMOzOJ3DyarHU64wn8JG13xK1/rt21xo2dDKTHn2zzf685BicvAP0BGAAN/9uTf2x7c7fTMyhbcQf28Pcl5eQnXC23di9P35EQ201Exa9yu+f/dtgTFFGEsam5tg4ae2dFPW1GJsa9rwXCG4HhAgs+MtT26CkTqHsVKyRTNppuwqBQCAQCAQCgMp6JY2qzmXNmMmlmBsLQUUg6Cyn4tN54uNfsLIwY8H4AZiZGLPlyDme+mRlp8YXV1Sz6P9+ZFivYJ6ZN5akrALW7TvFpcx8tnz09FXZf43sE4KXS/v+poZsFZqpa2iksqaeXl1tWvWZmcixMjclt7i8w3XkFJXx/i87uHNsfyJDu7QpAidlai3zth+7wDOfraakohpTYzkje4fwzPyxbXogp+YUsf3YBQZ1DyAy1M9gjOD2JjP2FGv+70lMLazoO2kBclNTYg5uZd0HT3VqfE1FCav+8zCBfYYxYsHTFGUmcXbPegoyLrH4i81IZZ2XioIiR2Ln6tVujK1L29ZFjQ311NdUYuXQq1Wf3MQUEwsrKopyO1xHeWEOe374gD4T5uMbHtmuCJx4Yh8XD2xmxvMfYWHbto97YcYlTC2t2brkdZJPHURRX4u5jT09R89k0IwHrup1Egj+Coh3rOAvz8qoJJYfSOxUbISvA1/dP+Qmr0ggEAgEAsHfiVe2p3Mup3Xlb0PcF+nC/f3+moXqBIK/Iu//vB0jIxk/v/EQLvbWAMwa2Yf73l5ORXVdh+Mrquv0/HcBFI1KNh06y6n4NPqHdb4QcKCXC4Fe114UsbpOWzz7yizgZkyN5dQ3NLY7h0qt5rWlG3G1t+axmaPajU3K1vpgx6Xl8uSc0ZgZyzkem8LGg2e4mJrNitcfxNrCrNW4lbuPo9HAfZPFfdHfld0/fIBMZsQ9767A2kH7no4YM4tfXrufuqqKDkZDXVWFnv8ugFLZyIV9m8iIPYVf99aFCNvC2ScQZ5/OFY83RENtNQByk9bvZQC5sSmNDe3/rVCrVWxd8jrWDi4Mm/dYu7FVpYXsXPou3UdMJbDP8DbjFPW1lBflgkYDGg2THn0LRX0NMYe3c3T9dxRnJTP92Q/bPzmB4C+GEIEFf3km9PSmu0/nqixbm4ksYIFAIBAIBFfH44PdqapvXZTJEO424lpDIOgsydmFpOYWMXtkH50ADFqx9J6Jg3j12w2dmmfCgHC9n0P93Nl06CwlFdVXtZ66hkbqFe2LtCZyI8xN2/icN20YkGDYjkHSsUsDy7ccJi4tl59eexAT4/Zvx6cNjWBEr2AWThqETKrNeB7VNxQfV0c+Wb2bFTuO8vgsfSG5uq6B7ccuEO7vSUSQd8cLEtx2FGUlU5KdSsTY2ToBGLRiab877mHLl692ap5ugyfo/ezWJYQL+zZRU264cHpbNDbU09hQ326MkbFxmzYKzf61bX58JJIOP1zHNn5Pfkocd7/zI0bGJm3GaTQatn31Jqbmloxa+Fy7c6qVSobNewxLOyfChk7StYcNncz6D5/l0on9pJ0/jl+PzgvmAsGtRojAgr88HvYWeNgb3uokEAgEAoFAcL10dRb+fgLBzSAjrxgAHzfHVn1d3J06PY+DtaXez3Ij7W2sSn11xY9WbI+6Lk9gsyZxuC0huV7RiJOdtcE+gIsp2SzbfIgF4wbgbG9NWVWt3nzVdfWUVdViY2GGVCph5ojWxawA5ozqy+dr9nA8JqWVCBx1PokGhZKJA7u3e56C25fS3AwA7N18WvU5eHbe/sPCRt8aRWakfX9r1OqrWk/05p+uyxPY2FSbAdyoMCwkKxvqsbJr++9FblIMRzcso++kBVjZu1BbWa43X0NdNbWV5ZhZWnNy+0oyYk8y47mPUCoUKBUKANQqrf1kbWU5UqkUU0trTC2t2yxG13v8PJJPHyLtghCBBbcXQgQWCAQCgUAgEAgEAsENR9UkJhlfZ2GyzhRC6wyTBvWgZwfZsU62Vm32WZqZYGNhRlF5Vau+Zr9gl3ZE4KMXklGp1Py0PYqftke16n/u898A2PLhU7g72bY5j7HcCCsLM2rrFa36Dp5NRCaTMqZvaJvjBbc3apV254qR3LAtSWeRSDvvp90eYUMn4Rncs90Yy3ZEXBNzS0wtbaguK27V1+IX7GxgpJbUc1GoVSqiN/9E9OafWvVv+Oh5AB75fDPJpw+BRsOGjwxnAX+xaDTWjm4s/nJLu+fT7COsqO+clZRA8FdBiMCCfxwDXt90zd7B1zP2aqltUPLjwUT2xuRQUl2Pp70Fs/v7M62Pb6fHrzh8iX2xORRU1OFqY86Enl4sGBzYqoCGUqVmzfEUtpzJJLesBntLEwZ3deXBESGi0J5AIBAIBFcw5Ivz9PSw4IsZAX/q2KulVqFixckC9iVXUFLTiIeNCbN6ODIlrHM2W9UNKn48UcCh1AqKqhuxMJYS4WnJg/1c8bE3bXNcXH4tj65L4pNp/kR4Wrbqj0qr4OdThSQX12EulxHhYcG9ka74ObQ9p+D2xMtF+15Lz2st7mQUXN2W8xuBp7Ndu4XfOkOonztnL2XSqFQhv0zcjk3NASDM36PNsW2J0NuizrPt6AWenjuGIG9XHGwsySoo5bnPV9M9wItX77tDL760sobyqlq6+bU+1unEdIK9XbFro2ic4PbHzk37HirJSW/VV5pnuMjgzcTWxbPdwm+dwc0/lOyEs6iUjciMWsTtvOQYANwDwtoc25YIHXNoG7GHtzNiwdM4+wRiYevAyLufob6mslXsvp8/pSgzibmvLNHZScQf3c3hNd/Qf9q9dB8+RS++JCcNoMOCeALBXw0hAgv+cbwxszf2Fm37BN2ssVeDWq3hpVXRnEorYmpvX7q623IoPo/3N5+juLKOB0eGtDtepdbwr5XHOZtezKQIH0I8bInNLmPpvnhiskr5aMEAvfg315/mj5gcRoV5MHeAP4m55Ww4kUZsdhnfPDDkurM3BAKBQCD4O/HqGG/sza/tMvp6xl4Nao2GV7anczqrmilhDgQ5mXE4tYIP92dTXNPYYXE7pVrD85tTicuvZVywHd1czcmvamTTxWJOZFTxzexAg6JtTkUDr25PR9XGLv1NF4v5+EAO9uZGLOjtgomRhG1xpTyyNokPp3Shu7sQrv5OBPu44uPqwM7jMdw7aTAONtqHAo1KFb/sPHaLV3dtjB8QzrGYFNbtP8X8Mf0Arc/oLzuPITeSMa5f22JVWyL0uUta4S7E150+Ib4AuDnaUl5dx87jF1k4cRBeLi1b979cuxeAKUN66s1TVF5FcXk1w3sFX88pCv7iuPh2xd7Nm7iju+g/9V5dVqpKqeTk1l9u8equjdBB40g7f4yze9bTZ8I8QPu5OrHtV2RGckIGjmtzbFsidHbCOQBc/YLx7qa1VnHtYvg+2tRCuwPAN7yfrs3Rqwtl+Zmc2r6K0EHjMZJrk6MU9XVErf8OI7lJu+sSCP6KCBFY8I9jfI9rf1p3PWOvhr0xOZxMLeLxsd24a7C20urU3j688Otxfjp8iUm9fHCzbdu/cG9MNqfTinlwRDAPjNBeBE7v64eFiRFrjqdyMqWIvv7aLTl/xOTwR0wOcwf48/SElqIbLjZmfPtHPPvjchnXXTzhFAgEAoGgmXHB155JeD1jr4Z9SeWcyqrm0UFuzO+l3UZ7Rzd7Xtqazs+nCpkYYo+rddu7fbbFlRKbX8tjg9yY16tlG+6IABseXpvEV1G5fDili96YqLRK/rs3k4o2iuyV1Sn58kguNqYyls0NwslSm+01NcyBe1dd4v/+yGLFnV0xkt2Yrf+CW49EIuHFuyfyxP9+5a43ljJrZB/MTYzZcfwiKTmFTTG3eJFXycQB3dlw4DSfrNpFVkEpgV4u7D8dT9SFZB6dMQJXBxtdbHZhGReSs/B0tqN7wNVdTxvJpLx090ReXLKWB979ntmj+mJlbsrBs4mciEtj0qAejL7C8qE549rtsjUI/n5IJBLG3P8ia//vSX789wIixsxEbmpO3JGdFGenNAfd2kVeJd0GT+TcHxvZ9/MnlBdk4eQdyKUT+0g9d5Qhcxdj7djy4LK8IJucSxewdfHEI+jmeV87eQXQZ8J8Tu1YxS+v3UfY0MkoGxu4eGAzpflZjH/oFb3CfALB7cCNMYERCAQ3lB3nMpHLpMyMbDH2l0gk3DkoEKVKw54L2e2Or65vJMDFmqlXWEf09dfexCXmluvafj+VjqWpEQ+P0n8qOqW3LwuHBuFgKbZmCgQCgUBwu7Ezvgy5VML08JaCXBKJhHm9nFCqNey9VNbu+JOZWs/TK60jujqb42tnyvlcfR/Et3Zl8NLWNGzNjBgdZGtwzuiMShqUGmb3dNIJwADmxjJm9XAkq7yBMznVV3OagtuAft268NXzd+PlYs8PW4/w7aYD+Lg68PLCyUBLkbfbBalUwufP3sXsUX3541QcH/6yg4LSSl677w4emDJUL/ZMYgavLd3I+v2nr+lYI/uE8M2L9xDg5cLPO47y+Zq9lFXV8uKCCbz5wNRW8c2F5qzMxfX73x3f8H7MfXkJdq5eHP/9R6LWLcXe3YdxD/4bQJe1ersgkUqZ/eJn9Bo7m8ToP9j7w4dUlRYyftGrDJz+gF5sVvxZti55nXN7N9z0dY1a+BzjF70KwIFVX3Bs4/dY2jsz56Uv6DFy2k0/vkBwo5FoNJqrK6kquC2RSCSUrP7XrV7GTeViZinL9ycQm1MKQP8AF+YN8OfB7w7xwPCuOguFK319H/3+MBW1Ct6Y2ZuvdsdxMasUJNDTx4HHxnaji3NLcYfOeAIv2xfP8gOJ7a51Yk8vXpvRu83+sf/dhoedBT88MlyvvV6hZMQ7WxkW4sb/ze9neHA7LN+fwLL9Cbw9uw+jwz1RqTWMfGcLkf7OfHhXf90xjGTSVr7Bf1cc5n2A+DMoEAgEbSORSCj4YvatXsafQkxeDd+fyCc+XyukRPpYM6enI4+sTea+SBedhcKVvr5PbEimok7Fa2O9+fpoLjF5tUiAHh4WLB7ormeb0BlP4O+j8/nhREG7ax0fbMcrY9oucDVxaQzuNsYsmxuk117fqGbMNxcZ2sWGdyf5tjm+rLaRoppGgpxa7zya/n0s9Y1qdjzcsoPovlWJDA+wZV6EE7+eLuSHEwV8Pl3fE3jFyQK+O57P/032ZZCffqbisfRK/rUljYf6u3JP339GZpXLE2tv+TWIRCKh6uivN21+jUZDSUUNjratvaF3R8fw76/X88YDU1vZGghuT6wG3nXL39M3C4lEwvIzrb1kbwUajYaaihIsbR1b9cUf3c3mz19m4iNvED78DgOjBX93Huhl/bf9HAqun9vrsatA0AZn0op45udjWJnKmT8wADO5EdvOZfLcr8c7Nb6kuoHHfjjCkGA3nhgfRnJ+BRtPppGUX8GGZ8ZelSA6PNQdT4fWF7qX42HXtt9dvUJJVV0jzr5mrfpMjY2wMpWTV17b6fUolCpyy2o5EJfLDwcTCfWwY1iIOwC5ZTUolGrc7cz5IyaH5fsTSCuqwkgmYWCQK89N7I6zTet1CAQCgUDwd+NsdjXPb07F0kTG3AhnTOVSdsSX8uKWtE6NL61t5MkNyQzuYsPjg91JKa5jU0wJyUWprLk3BCNp57fmDvW3wcOm/RoEHjZtZ3nVN6qpalDhbNm6crypXIqliYy8KkW789uZy7Ezbz1+76UyimuUDOlirde+dE4g8g6ul8yNtf01CnWrvoo6JQAlNY3tziG4/Zjyr88I9/fk2xcX6rVvP3oRgO4B11dQSiD4J/Ltk1NxDwxn/mvf6LXHHtkBgHtQuKFhAoHgH44QgQV/Cz7aegEjqZTvHx6uEy2nR/qx6LuDVNS2f5MDUFGr0PPfBWhUqdl8OoMzacVEBji3M1qfAFcbAlyv3YerukF7E2QmN1yMzUQuo15h2GvPEFvOZPDR1gsA2FkY8/zk7siNtDdhlXXaG63o5EJ+P53BXYMCWORqQ0xWKauPpZCUV8EPjwzHxvz22k4kEAgEAsHV8r+D2cikEr6bG4izpfZ7b3q4A4+sTWrT4/ZyKupVev67AAqVhq1xpZzNrqavt1Wn1xLgaEaA47U/hK1uuk4wlRsWZU2NJNQ3thZiOyK9tJ5PDuZgJJVwb6R+tm5HAjBAuJv2IfiexDLGdtX3Rt6fXAFAQ1sV5QS3JRKJhCmDe7J23yme/Ww1A8MDUKnVHDybSHRsKnNG9cXXrXU2o0AgaBuJREL48Cmc3b2W9R8+S5eeA1GrVCSfPkT6xWh6jZ2Ng7vvrV6mQCD4CyJEYMFtT0pBJWlFVcyI9NPLWjWVy7hrcCBvruucB9e4K4q+Bbvbsvl0BiXV9Ve1nnqFkvrG9m8WjY1kmJsY/vg1b91oy8tfIgGuwuc/1MOO9+f3o6CyjpVHkli07BDvzolkaIgbSpX2BjCjuJoP7+rP4K7aba7DQ91xtTXn420XWBmVzOIxoe0dQiAQCASC25rUkjrSSxuYHu6gE4ABTIyk3NnLmf/szuzUPGOuEDaDXczZGldKSe3VZbfWN6qpV7Yv0hrLJJgbG35gTJOOKmnjgkHSZk/bJBfX8dzvqVTWq3hmmIdBm4iO6OpszmA/a46kVfJ/f2Qxq4cjMomEjReLic3XegxfTca04Pbg+bsm4OvmyOYj5/h8zR4AfN2dePW+O5g+rNctXp1AcHsyeuFzOLj7cPHAFg6s/AIAB3dfxi96VXjVCgSCNhEisOC2J7NYW7jEx7G1BYOfU+ezbuwt9LddGjdly6qv0k/nlyNJ1+UJbG6s/Vi2JSTXN6pwsup8dlCIhx0hHtp/Dwt2484v/+CTHRcYGuKGadPNo6utmU4AbmZqb18+3XGRkymFQgQWCAQCwd+azLIGALztWlsw+Np3vsCSvbn+pbVcphU01VeZ3LryTOF1eQKbNdkuNLQhJNcr1TgasIpoi+iMSt7YmUGNQs1jg9yY0f3aMzdfG+vNR/uz2R5XyrY4bR2HEGcz3hzvwzObUrEyaUPYFty2GMmkzBvTj3ljrr6ehUAgMIxUZkTv8fPoPX7erV6KQCC4jRAisOC2R9V0Z9WZbYjtIb1BmScTenrT3ceh3Rgnq7ZvKC1M5VibySmubJ2BrPMLtrm2ir/ONmZE+DpyJDGfiloFLtZaMdnBsvV8ciMp1mbG1CiU13QsgUAgEAhuF27YtURb23iukvHBdnR3a7t+AIBDOyKuhbEMa1MZxQb8ddvzCzbEtrhSPtyfBcBLo7yYFGrfqXFtYW4s4/VxPjw62J2c8gbsLeR42ZpwJrsaAE9bYUElEAgEAoFAcDMQIrDgtserqQhbRnF1q77Mkpo/ezl42FvgYd/+jVtHhHjYcT6jhEalWuffCxCXUwZAN4/2b8D+vTqauOxy1j49GmMj/Yya2gYlUon2RtfcxAh3O3MyS6pRqtR6BfBqGhopr22gq9u1+xsLBAKBQHA74GmrzQDOKGv9ADarvOHPXg7uNia4d1AYriOCnc25kFtDo0qtJ27HF2iLy4a4dGznsDW2hPf3ZWMml/L2BB/6+Vh3OKY9qhtUHEqtwN/BlK7O5jhatAjR0RmVAPT0aL+4rkBwI9l8+BxvLf+dNx6YypQhPW/1cq6avSfjeHHJWoN9U4dG8Pr9Uwz25RWXM+fVr3nhrgm35XkLbh0XD2xh+zdvMfGRNwgffsetXs51oVYpWfHKQpy8A5n06Jttxp3Y+gv7f/mUF1efatWn0Wi4eHALZ3b+RmleJibmFvj3GsKQOYuxsNG/Z6+rruDwmm9IOrmf+poq7Fy96D5iGr3HzUEivb6H0AJBZxHvNMFtT5CbDd4Oluy+kE3pZf69SpWaVVHJt3Bl187Y7p7UN6rYeKqlIrlGo2FlVDJymZQx3T3aHe9ma05hZR2bTqXrtZ/PKOF8Zgl9uzjrPIkn9vSmqq6RtcdT9WJ/OZKERgOjwto/lkAgEAgEtztBTmZ42ZqwN7Gc0sv8e5UqDavPFt3ClV07Y4JsqVeq+T2mRNem0WjPRy6VMDrItt3xMXk1fHQgG3O5lE+n+V+3AAxae4z/Hcjmq6g8vfb00no2x5QwpIs1HtcpfgsE/ySSsrS2MS8vnMTbi6br/Td9qGG/5bKqWp76dBW19R0XzxYI/q6o1Sq2ffUmBent2zgmnz7EwVVfttl/ZO237PjmP5ha2TDirqcIH3YH8VG7WPHKQuqqynVx9TVV/PL6A5zdvRb3wHBGLHga94Bw9q34H5s+fVFXF0gguNmITGDBbY9EIuG5yd159udjLPz6ADP6+mFmLGP3hWxSi6p0MbcT47t78fupdD7fGUN2SQ0BrtYciMvjWFIBD48KwcWmJXsnp7SGi1mleNhZEO6tfdq4cGhXjiTm8/nOGFILqwh2tyWtsJJNp9KxMTfmucnddePvGhxIVGI+X+yOIaWwkm6edlzILGHn+Wz6dHFiYk/DfoMCgUAgEPxdkEgkPDvMg+e3pPLA6ktMC3fETC5lT2IZaaXaB8y315UEjA22Y3NsCV8eziW7XIG/oymHUio4nlHFQ/1dcbFqsV3IrWjgYl4tHjbGhDXZUHxxOBeVGvp3sSKrvMFgRvS4YLtWbe3RXGjvhxMF/HtrGgN8rSmtVbLufBEWJjKeHCIePAsEV0NSVgE2FmbMHNGnU/EXk7N5+Zv15BaX39yFCQR/YapKi9i25HUyYk+2GaNWqzix5WcO/fY1GrXhWj2VJQUc3bgcn259mfvyEp3m4OrfjY0fP8/J7asYOncxAEc3LKc0N53+U+9l2PzHdXM4+wax5/v3iT28jbChk2/gWQoEhhEisOBvQaS/M58vHMh3+xJYcfgSRjIJg4JcmdW/C29vOHPdHn9/NlKphP8tGMDSffHsi83l99PpeDlY8u+pPZnS21cv9lxGMe9sPMvEnl46EdjG3JjvHhrGd/viORifx9YzGdhbmjChpzcPDO+Kk3VLYTlTuYwv7xvMT4cusediNrsvZONkbcp9w7py77CgG+aVLBAIBALBX5k+3lZ8MtWf5dH5/HKqECMpDPCzZmZ3R97dm6Ur8na7IJVI+HBKF5Ydz2d/cjlbYlV42prw4khPJnfTr11wLreG/+7NYnywHWFuFtQqVMQ12UbsS65gX3KFwWNcrQgMcG+kC3ZmRmyKKeHzwznYmBoxPMCWhX1dcLqKYnUCgUArAvt7Oncq9ou1e/lpexROtlbMHN6b9QdO3+TVCQR/PS6d2M/WJa+jUasZMO0+jm36oVVMfXUlv7z5ICXZqQT2GUZVaSH5qfGt4ioKc3EPCKfXuDl6SWe+4ZEAFKS1jEk6uR9jU3MGznxQb46eo2dwZO03nN2zQYjAgj8FIQILbns0Gg2l1Q308nPi6wec9Pr2XswGwMGqZWvhsf9M04v56v4hBuedFOHDpAgfvbYrx95MLEzlPDOxO89M7N5unKF1glYIfn5yD56f3KPDY5mbGLF4TCiLx4Re83oFAoFAILhd0Wg0lNYqifC05EvPAL2+Py5p/fjtL/OvPfyE/nfrFzP0xzQzMcSeiSH6noBXjr2ZWBjLeGqoB08NbT/D9sp1mhvLrmud9/dz5f5+rgb7pBIJ07s7Mr274zXPL/hrUVuv4PM1ezh6MYXCskoszUzo1dWHh6YOI9DLRRfXqFSxavdx9pyMIz2vGEWjEgcbSwaGB7B4xggcbLR+0Kfi03n4/Z/476OzSM4qYGvUecqravH3dObpuWMI9fPg6w372Hk8hroGBUHerjwzbyxhXbTv89yicu544TMenzUKY7mMVbujKa2swdPFnjkj+zJrZMdZs+eTsli+5TAXkrNoUDTi4+bIjGG9mD2qr57YE5Oaw1fr95GUVUB1bT1ujraM7BPCA3cMxcyk7YcazefYHm4ONmz9+Ok2+6tq68ktLmdQ9wDd66vRaDCWG77FT8oq4M6x/Xlo6jD2n04QIvAtRFFfy4GVX5B27ihVpYWYmFviGRzBoJkP4ewTqItTKRs5tWMVCcf2UpqbjrJRgYWtA116DGTInEewsNU+0MuMPcWqtx9hylP/pSgzmZiDW6mrKsfRy58RC57Czb8bh9d8TVzUThrr63D2CWLE3U/jHhAGaMXMb56cwrD5jyMzknNqx2pqK0qxdfWk19jZRIyZ1eE5ZSee59im78m5dAGlogF7Nx96jppOxNjZep+Z3OQYDv/2NYWZSTTUVmPj6EpQ5CgGzngAuUnbxc+bz7E9rB3dWPzllnZjirKS8Q2PZPidTyIzkhsUgRtqq1E1KrjjyXcJHTiOlW8tMjiXV0gEd7/9fav2ZsHY2tFN11ZZUoCTdwByY/1zlEpl2Dp7UJAWj0ajue12MAtuP4QILPhbMPPTPYR52vHlfYP12nee11azDvO6vkrWAoFAIBAI/t7MXRFPN1cLPpvur9e+O7EcgDDXjgupCQT/RF5cspbTCenMHR2Jj6sDBaWVrNoTzfGYFNb/93Gc7Kx0cYfOJXLH4J5MH9aLhkYlxy6msPHgGfJLKvjy+QV68366ejeWZibcM2EglbX1/LjtCM9+tpogb1dUajUP3DGE8qpaftoRxTOfrmLj+09gadaS+LHhwGlKK2uYOzoSR1tLdhy7yH9XbCO3uJwn54xu83z2nozj5W/W4+1iz72TBmFqLOfoxWTe/2UHcel5vPngVAAy80t49MOfcbazYuHEQZibGnMqPp0fth4hM7+EDx6f0+Yx/NwdeXvR9HZfVzMT43b7k5v8gAvLqrj7re9IzMhDrdEQ4uvO47NG0a9bF734j5+ch/yKgtGCW8OmT14iK+40vcbPwd7Nh6qSAk7tXE36xWge/HgdVvZOTXEvknzmMOHDJtNj5DSUjQrSLhzj/L6NVJbkM+ffX+jNu/+XTzExs6DflLupr6ni+O8/suGj53D2CUKtVjFg2v3UVZUTvWUF6z98lkWfbMDEvKUY57m9G6itLKXXuLlY2joSe2QHu5f/HxVFeQy/84k2zyfh+F62fPEKdq7e9J+yELmJKannjrLnhw/IS41n0uI3ACjNy+S3dx/Dyt6ZflPuwdjUgszYkxz//QdK8zKY/uwHbR7DwcOPyY/9p93XVW7a8fd0/6n3IjPSPqCpKMw1GGPl4MyiTzZcVbE2jVpNZXE+WQlnObjqC8ysbIicfLeu39jUHEWd4aL1tVUVqJSN1FVVYG5t2+ljCgTXghCBBbc9EomESRHebDiRxr9WHqd/oAsqtYYjCXmcSCliZqQfPo5Wt3qZAoFAIBAI/qJIJBImhtiz8WIJ/96aRj8fK1RqiEqr4GRWNTPCHfC2aztDSSD4p1JWWcPRi8nMHtmHp+aO0bUHebuyZP0fxGfk4WRnxaXMfA6eTWT+mH48f9d4Xdz8Mf245z/LOBaTQlVNPVYWLZ8zpUrND68+gEWTsFtdW8+vu45TW6/g5zce0lmWNTQq+Wl7FHFpOUSGtgifeSXlfP/K/XQP8AJg9si+3PfOcn7ZeZTpw3rh5dI6SaSuQcF7P24l2NuV5a/crxNN543px0e/7mTVnmjG9Q9jQJg/B84kUFPXwFsv3E23pizkGcN7I5VKyC4sQ9GobDMr18HGkokD29/t1xHNReHOXcpkwYQBPDRlKBn5Jfy84yiPf/QLHzw+hxG9g3XxQgD+a1BbWUba+aNEjJ3NiLue0rU7+3bl0OolFKQlYGXvRGHGJZJPH6L3hPmMXvicLq7PhHmseGUhaeePUV9ThalFy32uWqVkwds/YGKm9XZvqK3m5LZfUdTXsvDdFTpRU9moIHrzT+SlxOmsCwAqivNY8NZyPIK0782IsbP45fX7ObH1F3qMnIadq1er81HU17Fr2X9x8e3KXW8t1wmsvcfPY+9PH3N6xypCB47Dr0d/kk4dQFFXw6RH38TNvxsAPUdNRyKVUV6QjbJRgZHc8MMPC1sHug2ZeE2v+eU0r689pLKrl8lKctNZ/rz2wY9EKmP8Qy9j5+qp6/cM7kHymSPkJsfoMrAB8lJiqSjSitFKRT0Cwc1GiMCCvwXPTAjH19GSrWczWbI7FgBfRyuDHroCgUAgEAgEV/LkUA+87UzYHl/G11F5APjYm/KvkZ7ccYWHrkAg0GJhZoKFmQl7TsYR5O3K0J5dcbS1ZETvYD0BMsjblUNfv4T0isy60soaXfZudX2Dngg8uHuATgAG8HPXZkeO6hOiV7PCu0nMLSyr0pt7cPcgnQAMWhH0nomD+PdX6zh4NpEF4we0Op/jMalU1NSxsO8gquv0iyGO6xfGqj3R7D8Vz4Awf5ztrQH4fO1e7p88hIggb4zlRrzz8IwOX7dGparV/Fcik0qwtjBrsz/Y140H7hjCxIHd8XVrsVcZ3TeUOa9+zQe/bGdYRFdR3+MvhrGZBcZmFiQc24OzdyABfYZiaetIUN/hBPUdrotz9gni6R8OIpXqi/c1FaW67F1FXY2eCNyl5yCdAAza7FmArpEj9bJam8Xc6rJCvbn9IwbrBGDQCqaRd9zD5s/+TdLpg0RO0s/WB0i/GE19dQVdp9xDQ61+pmvIwLGc3rGKSyf34dejP1b2Wv/qAyu/YMC0+/AMjsBIbswdj7/d4eumUippqK1uN0YqlWJqad3hXDcDE3NLpj79fygVDZz7YwM7vn2b0twMht/1JAADpj9A6rmjbPz4BUYtfA63LqEUZl5i748fYWphTX11BVIjIc8Jbj7iXSb4W2AkkzK7vz+z+/t3HCwQCAQCgUBwBUZSCbN6ODGrh1PHwQKBAABjuRGv3z+F/3y/mXd/3Mq7bMXf05lB4QFMGdJTJ9w2x+6KjuF4TApZBaXkFpdTWllDswWmRq3Rm7vZI7gZWZOI5Wir394sLKuvGB/g1bpgml+TWJpdWGrwfDILSgD4fM1ePl+z12BMbnE5AKP7duPoxRS2RZ3nVHw6JsZGRAT5MCyiK3cM7tGuncP5pKzr9gTuHuClJ3LrxjnaMjwimO3HLpCaW0RAJwvHCf4cjOTGTHj4NXZ8+za7lr3HrmXv4ejlT5eeAwkfPgXHJuG2OTb+6C7SLkRTlp9FRVEutRWlNH9oNGq13tyWtvoPLJsFZAs7R4PtV37mnLxa+9s7uPsCUF6QY/B8yvIyAK2we2DlFwZjKoq0D1aD+48m7fwxYg5tIzP2FEbGJngGRxDYZxhhQydjbNr2Q4+cxHM3xBP4ZmFl70xwf63NTOjg8fzyWlMG9agZ2Ll64h4QxoznPmbnd+/y+6cvASA3MaXflHupLMnnwr5NmFrcGgFb8M9CiMACgUAgEAgEAoFAILgmRvcNZWB4AEcuJHH8YjKnEtJZseMov+46xruPzGRMZDeq6xpY/MEK4tNz6RXkQ5i/B1OHRhDq586vO4+z/diFVvMaydqyL+hcZquRrLWfp1KlbnfuZiF58YwRhPt7GoyxbspWNpJJ+c9D03ho6lAOnknkRFwa5y5lcDwmhV92HuOn1x/EzsqwR2mQtwtfvXC3wb5mTIyv/VbdwUabDVpb3362seDWENx/NF16DiTlbBRp54+RGXeaE1t+5uS2lUx58l2C+4+mobaa1e88Sn5aPF7BEbgHhNF9+BTc/EM5ue1XYo/saDVvWzYGkk5+ZgxloqrVqnbn1mi0n5khcx7BPTDcYEyzuCmVGTHp0bcYOOMhkk4fICPmJNkJ50i/cJyTW3/h7nd+atMT19kniLmvLGl3/UbGJu32/1lIpTKCB44hLyWWwoxEnS2Ef6/BLP5yC4UZSahUSpy8/DE2NWflW4uwcnBp0wpDILiRCBFYIBAIBAKBQCAQCARXTW29gqSsAtwdbRkb2Y2xkVqfzzOJGTzywQp+2h7FmMhurN4TTVxaLi8vnMTMEX305iipbH+L97WSmd862zc9rxgAHzfDFi8eTrYAmMiNWhVWq6iu40RcKi5NNhB5JRVkFZQQGdqFBeMHsGD8ABqVKj5ZtYvf/jjJ7ugY5o6OvPIQAFhbmLWa/2p5+Zv1xKRks+o/j+jZZgCk5hYjkYCnsyiO/VdDUV9LYUYSNk7uhAwYQ8gArZd2VvwZVr+zmOjNPxHcfzSnd64mPzWOcQ/+m56jZ+rNUV1RclPW1pzVezmlOekAOLj5GBxj4+QOgJGxKb7h/fT66qoryIg5iZWDCwCVxfmU5mXiGx5J5KQFRE5agErZyL6fP+HMrjXEH91F7/FzDR7H1NK61fy3mujNP3Fi26/MeP5jPK4QwBV1tUCLMJ0Vf5aSnDR6jp6Ba5cQXVx9TRW5SRcJHjAGgeDPQIjAAkEHbDubwTsbz/Lq9AgmRRj+8vsro1Cq+OVIErsuZJNXVouJXEqYpz0PjAgmzKv1heGWMxmsj04lragKazNj+nRx5JHRobjY6GcyxGaX8cOBBC5klVKnUOJqY87Y7p4sHBqEcTuFJz7edp4jiflsfHbcDT9XgUAgEAj+imyPL+W/e7P492gvJobc3qKMUq1h0Zok/B1MeWWMd6v+sjolP0TnczyjipKaRrxsTZgW7sDUMAckEv1stNyKBpYey+dsTjV1jWqCnMx4oL8rER6WreYtqFKw7Hg+0RlV1DWq8bI1Zk6EE+ODW7+eF3Kr+fFEAXEFtUglEoKczHiwvythbhatYgXXR0p2Ife/+z2zRvbh3/dM0rUH+7hhbCTTWTiUV2kFkQBPF73xF5KzOJ2QDoDqiq3t18v+0/FkFZTqCsApGpX8vOMoJnIjRvQKMTimf5g/5qbGrNoTzdShEXqevN9s3M+aP07yyr2T8XVz5Icth1l/4DQ/vf4gYU2F4eRGMoJ9taKYTNo6E/lG4mxrRU5ROWv+OMl9kwfr2k8npHP0YhKDugdiby3e8381irJS+PWNB4gYM4uxD7yka3fxC0ZmZIykyaqhtqoCAMcrLBpyLl0gK+4M0JKle6O4dPIAZflZOs9gZaOC6K0rMJKbEBg53OAYv+79MTY159T2VXQfPkXPk/fImm84s3st4x56BQd3X45t+p5zezdw9zs/6oqjyYzkuPppP4/SNrP//5o4ePhRW1HKiS0rmP7sh7r22spyzv+xEVNLG7xCegGQEXuSqHVLcfDw1bUBHFz1BSqVkr6T7vrT1y/4ZyJEYIHgb85b60+zLzaXoSFuzOnfhfIaBRtOprH4+8P8b8FA+vq3eLV9uSuGX6OS6dvFiacmhJNTWsPa46mczyjlh0eGY2Ou3aKSmFfO4uWHsTSVM7e/P3YWJpxMLeT7A4lczCzl03sGGixCseFkGuui03C1bdvvSSAQCAQCwV8TlVrDe3sySSqqw9/BtFV/rULFkxuSya1QMC3cAS9bE05kVvHxgRzSSut5ZljL9vri6kae2JBCbaOaWd0dsTUzYuPFYp7ZmMJHU7rQx7ul2FFuRQOL1yWjUGmY2d0ROzMjdiWW8e6eLMrrVMyLaLmWOZJawas70nG3NuG+SFeUag1rzxfx5IYUvpjpTzdXIYrdSMIDPOkf5s+6faeorm2gV1dvGhqVbIu6QL2iUVd8bWhEV1bvjea1pRuYNbIvlmYmxKXlsi3qPDKZFKVKTXVd/Q1dm0Qi4d53ljN3VF8szUzZGnWexMx8/rVgQitf4WasLcx44a7x/Of7zcx99WumD+uFg40l0bGp/HEqnt5dfZg8qAcA88f2Z2d0DE99spKZw3vj7mhLdlEZa/44iYu9NWP7dbuh53Ml904ezP4zCXy1fh8Z+SWE+3uQmlvE+v2ncbaz5qXLRHnBXwePwHB8u/fn7J51NNRW4xUSgVKhIObwNhoV9URO1hZfC+w9lNM7V7N1yetEjJmFibkl+SlxxBzehlQmQ63quFDa1SJBws+v3UevcXMwMbck5tBWCtMvMfreF7C0dTQ4xtTSmlH3Ps+Ob99m+b/m0WPkNCxtHUi7EM2lE/vwCulF2FDte7HPhDuJi9rFuvefpufomdg4uVFekMOZ3WuwsnchZMDYG3o+N5uA3kMJihzJpRP7WPPfJwnsPZTaqjLO7V1PbWUZU5/6P4xNtYlUPUfP4Nze9Wz65EV6j5+LubU9yWcOk3LmMINmPoSLb9dbfDaCfwpCBBYI/sYcTypgX2wus/r58dykHrr2yb28WbBkH59sv8DKJ0YB2szelUeTGR7qxrtzInUibpCrDW+uP83Gk2ncO0z75fTRlvMYySQsf3gYbrbaL7YZkX58tuMiq4+lsD8ul1FhHrrjNTSq+GpPLGuOp/5Zpy4QCAQCgeAGUlzdyNt7MjmT3bbosO58MemlDbwy2ovxTRnP08Id+ffWNDZeKGF2Dyc8bbVbY384kU9RdSPfzA4k1FV7LTE22I6FvybyycEcflnQVZc5/MnBHCrrVXw9K4BgF23s1DAH7ludyPfR+czs7oBcJqWuUcUH+7JxtTLmm9kBWJtqb3VGBNiw4JdElh/P53/TRBHhG80Hj81mxY6j7DkRy4EzCchkUkJ83fj06TsZ3CMQgMhQP95bPIsftx1h6aYDGBsZ4epow+KZI/Fzc+TpT1dxLCaFkKYs2hvBmMhu+Hs6s3LXcapq6wnyduHjJ+cyvFdwu+OmDInA1cGWFdujWLk7GkWjEjdHWx6ePpy7xw/AWK59X/m5O/LdS/eyfMshtkadp7SyBltLc0b3DeXhacP1sohvBraW5vz42oN8u3E/B88msv3YBeytLJg8qAcPTx+Ok61Vh3MIbg3TnnmfE1t+JuH4HpJOHUAileHaJYRZ//oE/whtVrdPWF+mPPke0b//SNS6pcjkxtg4ujJkzmIcPfxY98HTpF04rmctcL0EDxiDo5c/p7avpKG2CifvIGY8/xGBfYa3O6778CnYOLoRvWUFp3asQqVQYO3kxuDZD9N30gKd162Dhy93vrGUoxuWE3NwK7WVpZhZ2RLcfzSDZi3SyyK+XZj69H85uW0lF/b/zt6fPsLY1BzP4B4MnPEgbv4tD4IsbR25843vOPTbV5zZtZZGRT0OHn5MefI9QgbeXuK34PZGoml28hb8rZFIJJSs/tetXsZtye1sB/H5zousOprCL4+NxN9F/0v1XyuPczghnx0vTsDWwoT//n6WLWcy2PjsWD3rB6VKzXf7EgjxsGV4qDvV9Y2M++82hoW48948fZ+zxLxy7v36ADMj/Xh+slZ0ziqp5okfoyioqGNaH1+iLuUjk0r+MnYQDvM+QPwZFAgEgraRSCQUfDH7Vi/jtuZ2t4M4lFLBO3syUak1zI1w4udThYwPtmtlB7HseD7H0iv5dnYgRrKWHUHrzxfz6aEc3hrvw8hAW1RqDROWxhDoaMaSWfpbnX+Izuf7EwV8MzuAbq4WFFQpmP1jPJNC7XlxlJde7LH0ShIKa5kWrs0O3p1Yxtu7M3ltrDdju9rpxW64UIxCpdHLGr7VuDyx9pZfg0gkEqqO/npL13CjyS0q544XPmPyoB689dC0W72cfxRWA++65e/pm4VEImH5mcpbvYybQkVhLt88OYWwoZOZ9Oibt3o5guvkgV7Wf9vPoeD6EZnAgj+F2gYlS/bEEp1UQGFlPZamRvT0ceT+4V0JcLXRxTUq1fx2PIV9MTlkFFejUKlwsDSlf4Azi0aFYG+p3Xp4Jq2Ix36I4u05fUkpqGD72SzKaxvwd7bm8XFhhHrYsnRfPLsvZFOnUBHoasMT48Po5qm9Icgrq2HGJ3tYPDoUuZGUNcdTKK1uwNPekpn9/JjR16/Dc7qQWcJPBy9xMauUBqUKbwdLpvbxZWakn57nXWx2GUv/iCMpv5KahkZcbcwZHurOfcOCMG2n6m/zObaHq61Zu2LqvcO6Mq6HFz6Orbe8ldUoAJA1ZfyeTiumi7O1TgBWKFVIkCA3krJ4TKhunJmxEb89NQaZgSKz5VfMCVBUWYe1mTEvT40gMsCZ6f/b1e45CQQCgUBgiFqFim+O5hGdUUVRdSMWJjJ6uFtwb6QLAY4tGXeNKjVrzxezP6mcjLIGGlUa7M2N6OdjxYP9XbE3lwNwNruaJzem8NZ4H1JL6tgRX0Z5nZIuDqY8OtidEGdzlh3PZ8+lMuoa1QQ6mvHYYHdd1mpepYI5P8Xz8ABX5DIp684XUVqrxNPWhOnhDkwLN7x19nIu5tWw4mQBMfm1KJRqvGxNmBLmwPRwff/cuPxalh3PI7m4nhqFChcrY4b527Cwrwum8rZ9R5vPsT1creSsvTe03ZiUkjp6e1qyeJA7cpmEn08VGox7sL8rD/Z3bdWeWKT1g3Wx0r72aaX11DWqda/l5TRn+sYX1NLN1YJzOTVogP4+LVmNtQoV5sYyBvhaM8C35SH3mexqJEC/JisJlVqDQqXGTC5jRveOfx8CgUAgEAgEf2eECCz4U3jltxOcSS9mdr8ueDtaUlhRx5rjqUSnFLL6iVE4WWtv3l5Zc4IjiflM7OnNlD6+KJQqopMK+f10BvkVdXx6z0C9eb/cFYOFiRELBgdQWdfIz0eSeHHVcQJcbFBrNCwc1pWKmgZ+iUrmhV+Ps/ap0ViYynXjN51Kp6ymgdn9uuBgZcqu81l8uOU8+WW1PDq2bR+vfbE5vL72FF4Oltw9JBBTuYxjSYV8vO0CCbnlvDpda/aeVVLNUz9F4WRtxt2DAzE3MeJ0WhErDl8iq6S6VSbt5fg6WfHGzN7tvq5mxu2b51ubGWNtZtyq/WJmKTFZpQS62mBlZoxCqSK3rIbBXV05nVrEkj2xxOeUI5VAhK8jz03qjp+z9iZLJpXgaW/YT2/V0WQAevu1ZNmEezmw4tER7a5TIBAIBIKOeH1HBmdzqpnZwxFvWxMKqxtZe76Yk5lV/LogGEdLuS4uKq2SCSF23NHNAYVKTXRGFVtiSymoauTjqV305l1yJBcLYxnzezlR1aDi19OFvLw1DX9HM9QaDff0caGiXsnKM4W8tDWNVfcEY3HZ9+/m2FLKapXM7O6Ag4Wc3YllfHwgh/xKBY8Mantr+/7kct7alYGnjQkLejtjYiQhOqOKTw7mkFBYy8ujtVm2WeUNPPt7Co4Wcu7s7Yy5XMqZ7Gp+OV1IVnkD70z0bfMYPvYmvGqgeNvlmLUjIjezoLczcpk2Lq9S0WE8QH2jmuyKBnbEl7IjvowhXax1frxF1Y0AOFvKW41zspDrHSejTOsTa2duxOeHc9gRX0Z1gwpbMyNm93Dk7j7OOsE8o6wec2MpNQoV7+/L4lh6FUq1Bh87Ex4Z6MbgLjatjicQCAQCgUDwT0GIwIKbTllNA8eTC5kR6cfj48J07YFuNnyzN57EvAqcrM1Iyq/gcEI+c/p34ZmJ3XVxc/r788C3B4lOLqSqToHVZaKmUqVm6UNDsTDR3jDUNDSy6mgKdQol3z88XOdrq1Cq+flIEnE55XqF0PIravn2gaGEe2u3Zs7o68eiZYdYeTSZO3r74OXQOoO2TqHk/c3nCHKz4dsHhiI30t4Uze7vzyfbL7DmeCpjwj3pF+DMwfg8ahqUfD6jF6Ee2izkqX18kUok5JTWoFCqMDYyLOTaW5oyvoeXwb7roaSqnjfXnwJg0Sith1R1vRKNBtIKq3j2l2NM7+vHPUOCSCusYsXhSzy87DDfPzK8TfEX4OfDl4hOLiTUw44hwS1ZQM2vj0AgEAgE10pZnZLozCqmhzvw6GXCaoCjGUuP5ZFYVIejpZzkojqOpFUyu4cjTw5t8aaf1cOJRWuSOJFZRVWDCiuTlu9epVrDN7MDMG8Sdmsa1Px2roi6RjXfzQ1EKmm5lvj1TBHxBbX08WrJSs2vVPDVrADC3LTfkdPCHVi8LpnVZ4uY3M1B54F7OXWNKj7an01Qkx1Cs8A6q4cTnx/KYe35YkYH2RHpbcXh1ApqFGo+meZNSFOW7JQwB6RSyK1QoFCpMZYZ/q61N5czLtjOYN/VIG9j/vb4/kQ+q84UAeBpY8xjg1t+bzUKbUV7Q1nMJk1t9Y1qAKoatLEf7MtGKpHw+GA3TIykbIkt5bvj+ZTUNuoKzlXVa2MfW59MsLM5r47xolqh1gr729J5a4IPIwJsr/pcBAKBQCAQCP4OCBFYcNOxMDHCwsSIfTE5BLraMKSrKw5WpgwLcWdYSMsNQaCrDXtfmYRMou8zUFrdgGVTYY+aBqWeCDwgyEUnAIM2exZgeKi7TgAG8HRoyjypqtObe2Cgi04ABq1gedfgQF5bc5IjCfnMH6TvUwdwIqWQyrpG7g71oKahERpa+saEe7LmeCoH4nLpF+CMc1OG81e7Y7lnaBA9fRwwNpLx5qw+Hb5uSpWa6vrGdmOkUonBTN+2KKio5cmfjpJbVstdgwIY3NVVdyyA7NIanpvUnVn9tFlSw0MhyM2G5389ztI/4vnPbMPr/uVwEl/ticPOwoS35/TR28IqEAgEAsH1YmEsxcJYyr6kcgIczRjkZ42DhZyh/jYM9W/J7gxwMmPXw2E64baZstpGLI214mKtQl8EHuBrrROAQZs9CzDM30ZvnmYxt7ha/7t5gK+1TgAGrWB6Zy9n3tiZwZG0SoMetCczq6msV3FnL1tqFGpAresbFWTL2vPFHEqpINLbSpct+83RPO7u7Ux3DwuMZVJeH9txnQKlSkN1k+DaFjIJWJne+FuCAT7WdHezIKOsgZVnCrl/1SX+N60L3VwtaLYqNHS10NzWfC3RqNIGNyjVrLirK2Zy7e9qZKAti9cms/FCCTO7O+JtZ4pSraFGoaavtxVvT/DVzTnIz5q7fk7gi0O5rX6vgr8f7k62nP7xjVu9DIHgtsHG2Z0XV5+61csQCAR/AkIEFtx0jI1kvDwtgvc2neX9zed4H+jibM2AQGcm9/LRCbcAxjIZe2KyOZFcSHZpDblltZTVNNB8rX6lwblDk0dwM81etI5WhtuvHO/v0npboG+Tf252WY3B88kq1rYv2RPLkj2xBmPyyrXedyO7uXM82Ysd57I4nVaMiVxGT28HhoS4MrGnN2bteAJfyCy5bk/gy0nILeeFX49TXFXPnP5d9LKyTZtufmVSCdP6+OqNG9TVFRcbM06mtPb/U6k1/G/7BTacSMPB0oTP7x2Eu13b2cICgUAgEFwLxjIpL47y4v0/svhwfzYf7ocuDqb087FiYog9vvYt3/tymYQ/LpVzIquKnHIFeZUKyuqUOnFRfUWtFHtz/e/i5msGBwt9q4Lmh8tq9OniYMqVeNtpBePcioZWfaC1eACtsPvN0TyDMc12CMMDbInOqGJnQhlnsqsxMZLQ3d2CIX42jA+x04mihriYV3NDPIGvhQhP7fXUYKCvtyWLfkvim6N5fDEjQGdBUa+88tVsabNoEu2bYyeE2Oudq1QiYUqYPXEFtZzKqsbbzhTTpt1HM67wY3a0kDOkiw27EsvIKG3Az8DvTCAQCAQCgeDvjhCBBX8KI7t50D/AhaNJBUQnF3AmrZhfo5JZfSyFt2b1YVSYBzX1jTzxYxQJeeX09HGgm6cdk3v5EOJhy+qjyew8n91qXiOp4UyOziZ4GBmobqZqujtsa251k5C8aGQI3bwMb7Fszs41kkl5fUZv7h8ezOH4PE6lFnEus4TolEJWRSWzbNEwbC1abxMFCHC14bOFAw32NWPShpXElRxLKuCV305Qp1Dx0Mhg7h8erNdvZSrH3NgIU2MZRga2fDpYmpJcUKHXVq9Q8traUxxJzMfLwYJP7h6IRzt2EQKBQCAQXA8jAmzp523F8YwqTmRWcSa7mlVnilhztog3xvkwItCWGoWKpzemkFhYRw8PC0JdzZkUak+wixm/nS1md2JZq3mv+1rCwPiOriWaH0o/2N+Vbi6ti6MBWJnKdHO8MsabeyNdOJxayemsKi7k1nAys5rVZ4v4dk4gtmaGL+kDHE355AoP5Csx/hNsm4KczPGxNyWxULsjy81ae51UXKNsFdvsF+zUlAHd/P8rxXoAh6Yif7VN2c5OlnJSSuoNx1po22ob28+MFggEAoFAIPi7IkRgwU2ntkFJckEFbrbmjA7zYHSY1qPvbHoxT/wYxS9HkhgV5sGa46nE55bzrzt6ML2vn94cJVWGM2mul6yS6lZt6cVVAPg4WrXqA3C3096smcilRPo76/VV1Co4lVqEi43WBiK/vJaskhr6+jsxf1AA8wcF0KhU8/mui6yLTmNPTA6z+xm+ObM2M241/7VwLKmAF1dGo9ZoeHV6LyZFtC4QI5FICPGw5Wx6MWU1DdhdJkxrNBryymtws225SVUoVfxrZTQnU4sI87Lnwzv7tSlmCwQCgUBwvdQqVKSU1ONmZczIQFtGBtoCcC6nmqc3pfDrmUJGBNqy7nwxCYV1PD/Ck6lhDnpzlNa2b7F0rTRn9V5OZpm2zcvO8HdjswhqIpPQx1v/eqOyXsnprGqcrbQCZ0GVgqzyBvp4WTEvwol5EU40qtR8eSSXDRdK+ONSOTN7OLY6BmhtHq6c/2ayaE0SSrWG7+cFteqra1Rj0iQ4+9iZYi6XklBQ2youvqkt1MWi6f/a64+00vpWsTkV2mzp5tcz1NWc4xlVpJXW42NvajDW1arzNlqCG8+p+HQefv8nFk0dxsPTh9/q5Vw1uUXl3PHCZ7qfR/UJ4YPH5+j6vtqwjzOJGVRU1xHg6cxd4wcwNrJ1semzlzJZuukAiRn5qDUaegZ68/D04YT4ut2QdSpVau75z3cEerrw1kPTWvUXlVXxxbo/OHYxmcqaOtwdbZk+rBcLxg/U7XpY8OZS4tNbdioIi40/n8zYU6x6+xEGzXyIwbMfvtXLuWoqCnP55skpup+DIkcy/dkPdH2H1nxNVtwZ6msqcPQKoO+kuwgZMKbVPClnozi26XsK0hKQSKS4B4QxeM7DeHbt2e7x133wDIq6Gu58Y+k1n8P78zq2cbzcSmP583Mozk41GLd4yTasHVx0P8ce3s7Jbb9SnJOGkdwYr5AIhs57HCcvfwCSTx9m/YfP6OJv1/eB4K+FEIEFN53UwkoeXnaYGX39eOGOHrr2rm62GBtJddsuy2u1N0z+LtZ64y9mlnI2oxjQFm+5kRyMzyOrpFpXAE6hVPHrkWSMjaQMCzF8ERYZ4Iy5sRG/HUtlci8fPU/e7/bFs/5EGi9N6YmPoxU/HbrEplPpLFs0jG6e2qxhuZGUYHdboGXL6c0ip7SGV9ecRK3R8H/z++k8gA0xoac3p9OK+eFAIs9OainMt+VMBmU1Cj1h/rOdMZxMLaK3nyMf3dUf03ZsLQQCgUAguF7SSut5dF0y08IdeG64p649yMkMY5lUV0+gok6bWXqlRUNMXg3ncrR2TqobfC1xOLWC7PIGnWewQqVm1ZlCjGUShnZpbTsFEOlthZlcytrzxUwKtdfz5F1+PJ8NF0t4YYQn3nam/HyqkN9jSvh2diChrlpBVC6T0tXZHCjhGmq23TScLeUcTKngSGoFgy87972XysirVDClm7YOg5FMwvAAG3YmlHGpqJYgJ+15VTWo2BZXiq+9CSEu2gfq3d0t8LAxZldCGfMinHWCb32jmvUXijGTS+nvq712HBNkx08nC1h5upCBfta6gnnppfVEpVXSw92ilc2HQHAtRAR5M2N4b9wctO/zorIq7n1nOfWKRuaNjsTR1oq9J2L591frKCip4O4JLbv7jsek8OT/fsXDyY777xiCRqNh7b6T3PfOcr59cSE9Aq+vMLRKreaN7zaRmJFPoKdLq/6qmnrue2c5hWWVzBzRhy4eThw+l8Rna/aSWVDKq/fdAcDiGSOoqK5j+ZbDpOcVX9eaBP9sPIMj6DlqOtaO2vvrqtIifn79Phob6uk9fh6Wdo4kHNvD5s/+TVVJAZGTF+jGJp7Yx6ZPXsTawYXBsx8BNJze8Rur/vMIc19Zgndob4PHPLDyC1LOHMYrpNd1rX3yY/8x2J52/hixR3YQFDlC16ZsVFCal4Fv9/6EDZnYaoyZZcv34sntK9m34n84evkz/M4naait4tT2Vfzy+v3c885POHj44uLXlcmP/YeSnDSObfrhus5DIGhGKDeCm06Ylz39/J3ZcDKN6oZGevo4oFCq2XEuk/pGFfMHaouvDQl2Y210Km+tP82Mvn5YmsqJzyljx/ksZFIpSpWKmobW2wavl0XfHWJWvy5YmBix/VwWSfkVPDupOw5Whv3irM2MeWZiOO/9fpYFS/Yxtbcv9pYmnEwpYn9cLhG+Dkzoqb14mzvAnz0Xs3n+l2NM6+uLm605OaW1rD+RirO1mS4r+mbxzd44ahuU9PZzpLq+kZ3ns1rFDAtxw8zYiIk9vTgQl8va6FQKKmrpH+hCSkElm06l08XZigWDAwFIK6xk48k0ZFIJQ4LdOBDf2svQw85Cr+CeQCAQCATXQzdXC/p6W7LpYgk1DSp6eFiiUKrZmVBGfaNaV3xtkJ81684X8/buTKaHO2BhLCOhsJZdCWXIpKBUQ00HhdKuhUfWJjGzuyMWJjJ2xpeSVFzP00M92hQcrUyNeGqoB+//kcXClZe4o5s99hZyTmZWcTClgp4eFowP0T48nt3Dkb2XyvjXllSmhTvgamVMToWCjReLcbaUM6opK/qvwKOD3DifW8NbuzKZFu6Ap60J8QW17IgvxdvWhIcGtDxgf6C/K1FplTy7KZU5PZ2wMJaxKaaYslolr4zpoisMJ5VIeGmUF89vTuWRtUnM6O6ImVzKtrhSsssbeGm0FxZNtQ08bU1YNMCNr6PyeGRNEhND7alqULHufDHGMgnPDLu5112Cfw4eTnZMHNiSNPHtpgOUVFTz3b/vpVdXbdHGmSN6c89by/h20wGmD++NpZn2QdH7P2/H2sKMH157AFtL7QOQcf3DmP7il3y+di/LX77vmtdVVFbFa0s3cjI+rc2YjYfOkFdSwVNzRnPPxEEAzB7Zl8c++oWNB89w17gB+Lk7Mqi79tp/08EzQgQWXBe2zh50u0wUjVq3lJryEu58Y6lOpO05egYrXlnIkbXf0mPkNEzMtUlaR9Z8g5HcmLveXIa1ozahqWu/USx7bjYHV33J3W/ri6P1NVXsWvYeCcf23JC1dzMg5laVFvLHio+xd/Nm4uI3de0l2amoVSoCeg02OK4ZtUpJ1LqlWNo5suCt5bpz9QmL5Nc3HiBq/VKmPPkeVvbOdBsykczYU0IEFtwwhAgs+FN4d15ffj2SzB+xORyKz0MmlRDsbstHd/VnYJD2j3mfLk78Z3Zffj58ieUHEpDLpLjamrNoZAi+TlY8/+txopMLdVm0N4LRYR50cbZm9bEUqusbCXC14f35/RjaRhZwM5N7+eBqa84vR5L47VgKDUoVbrbmPDgimDsHBWDc5NXr62TFV/cP5seDl9h+NouymgZszI0Z2c2DB0cE62UR3wxOphYBcDqtmNNphi/eNjwzBjNjIyQSCf+dF8nqoylsOZvBsaRCbC2MmdHXj0WjQnRF7E6lFaPRgEqj4dMdFw3OObGnlxCBBQKBQHBDeWeCLyvPFLE/qZzDqZXIpNDV2Zz37/BjQFMmaG8vK94c78Mvpwv54UQBcpkEFytjHuzvio+dKS9uTeNEZlVTFu2NYWSgLV0cTFlzrojqBjUBjqa8N8mXIW1kATczKdQeVys5K88UseZ8MQqlGjdrY+7v58K8CCddFquPvSlfzAjgp5MF7Igvo6xWiY2ZjOEBttzfz0Uvi/hW425jwrK5gXx3LJ+dCWVUNShxtjRmTk8nFvZ1wdKkpZaBs6UxX88O5JuoPFae0RafDXQy4/nhnvTwsNSbt6eHJd/MCmR5dD6rzxahUmvwdzTV+903c2cvZzxtTFh1ppBvjuZhLJMS4WHJQwNc9QoICgQ3mn7duugEYACZVEqfEF8SMvJIzysmrIsH5dW1uDnaMtLXTScAAzjbWePn7kh8eu41H3/f6XheX7oRtVrD/ZMH8/3WIwbjsgpKABjUI1CvfUiPQI7HpHApMx8/d8MWMwLBjcI3vJ9elq5UKsM7tA8FaQmU5KbjHqAtYl6Wn4WTd6BOAAatqOzo2YWC9ES9OXMuXWD9h89QX1PNoJkPEbX+u5uy9j3fv09dVQXTn/0IE7OWmjiFmckAOHoFtDu+trKchtpqfMP76QRgAM+uPTC1tGl1XgLBjeSvc9Uo+FtjYSJn0agQFo0KaTfucs/gKzn2n2m6f/fyc9L7uZlJET5MivDpdDvAXYMDuWtwoMG+9sb26eJEny5ObY5rJsjNlvfmRXYYdzPY+VLbTyANYSSTsmBIIAuGtP16zO7XpU0f486w8dlx1zxWIBAIBP9czI1lPNjflQf7t21tBOh5Bl/J4SdabKkiPC31fm5mYog9E0NaP8hsqx1gfi9n5vdq28e/rbG9vazo7dWxZ2+gkxnvTPTtMO7PwM3a2ODr1oyLlTGvjm1df8AQXrYmvDvJt1OxAU5m/HeyX8eBwFB/G4b6ty/CCzrmo193smpPND+8ej/dA/QtCpZvPsRXG/bz9b/uITLUj0alilW7j7PnZBzpecUoGpU42FgyMDyAxTNG4GBj2cZRYPJznwKw9eOn9dq/3XiApb8f5NsXF9InxFfXfvjcJX7ecZT4jDxUajUBni7cNbY/4/qHdXhOve99q8OYLR8+hbuTbYdxl9NsoXAlCRl5SCUSXOy1DytsLc356oW7W8VV1zWQWVCKm8PVHfdykrMLiQztwlNzxyCXydoUgX3dtAJvem4x/h4tf7eyCksBcLb783zE/07s/eljTu9YxYL/fI9HUHe9vqMbl3P4t6+Z9+rX+IT1RaVs5NSOVSQc20tpbjrKRgUWtg506TGQIXMewcLWoY2jwNePN9l1fLlFr/3I2m+JWv8d81/7Bu9uLT62KWeOEL1lBQVpCajVKpy8Augz6U5CB3Z8T9YZP9xHPt+MjbN7h3GXM37RKwbbC9K1nr/WDi3f8/buvlQU5aJUNGBkrM2mVzYqqCotxMpO/168NC8TR09/Rix4Gjf/0JsiAmfEnCTp1EHChk7CKyRCr68w4xKAztNXUV+L3MRMt7OlGXMbO0wtbSjNy0Cj0ej6m8VhF5/WvvoCwY1CiMACgUAgEAgEAoFAINBj6tAIVu2JZvvRC61E4G1HL+DmYEPfJnH2xSVrOXQukTsG92T6sF40NCo5djGFjQfPkF9SwZfPLzBwhKtn5a7jfLxqF+H+njw8bTgA+07F8/I360nPK+6w2Nzbi6Z3eAw76+vbKVBd10BmfgmrdkdzKj6dO8f2w8nWsLBaUlHNpawCvtmwn7p6BY/NGnnNx71v0mDkTbsRc4vK24ybPqw3e07E8fGqXZiayOni7sSxmBTW7z9NZKgfPYM69xBHoE/34VM4vWMVsUd2tBKBYw9vx9rRTSfObvrkRZLPHCZ82GR6jJyGslFB2oVjnN+3kcqSfOb8+4sbsqZm31n3wHAGzV4EwKXofWz5/BVKc9I7LDLWlh/u5ZhZ213XGhtqqynNy+T0jlVkxp6iz4T5WNq1ZKKPXvgc6z98lq1LXmfQrEVIJBKOrFtKbUUpIx59U2+u0EHjCB82+brW0xEHVy9BZiRnyNxHW/UVZSQhNzHlyLqlxEftor6mEhMLK7oNnsCw+U9gbKr1updKZYy57wW2ffUme3/4gN7j56FoqGP/z58gkUgYMP3+m3oOgn82QgQWCAQCgUAgEAgEAoEegV4uhPi6sedkHM/fNQGjJouSiynZZOSXsGjqMCQSCZcy8zl4NpH5Y/rx/F3jdePnj+nHPf9ZxrGYFKpq6rGyuD47jrySCj5ds4dhEV35+Mm5uuy5O8f2519L1rBs8yHG9gtr18rgch/fm8Xb329m78k4AMK6eHDf5CFtxs599WvKqmoBrYfw4O5t78briGYBuCPMTY15bNZIXvlmA0/+b6WuPayLBx8+MbdV1qKgczj7BOLaJYSEY3sYvfA5pDKt1JKbFENpbgaDZj6ERCKhMOMSyacP0XvCfEYvfE43vs+Eeax4ZSFp549RX1OFqcX1ZWRXFudz4NfPCOwzjOnPfaT7vfadcCebPnmRoxuWEzJwHA4evm3O0Z6v7Y1ix9J3SDy+FwC3gDD6T9P3xHYP6k6fiXdydMMyEqP/0LUPm/84YUP1BV+Z0c0t/JmdeJ685Bi6j5iKtUProouFmUk0NtRTUZjD2AdeQoOGpBMHOLNrDfmp8dz5xlLdGv0jBhM+7A7O7F7Lmd1rAZBIZUx+9C18wvre1PMQ/LMRIrBAIBAIBAKBQCAQCFoxZXBP3v9lB0cvJjO0p3aL8rao80gkMHmw1hYkyNuVQ1+/hFQq1RtbWlmjK4ZWXd9w3SLw/lPxqFRqxvULo7y6Tq9vXL8w9p9O4MCZBPzcB7c5R7Pg2h42FmZIpdcuhE4e1IMJA8KJT8/jl53HmP/6Nyx7+T68nPUtYTQaDU/PG4uZsZxD5y6xfv9pUnKK+OZf93Ra0L0Wth+9wOvfbcTe2pJn54/Fw8mOhAztWu9/ZzlLXri7zcxlQfuED7uDPT98QOq5YwT01or/MYe3gURCWFOGqrNPEE//cBCpVP93XFNRqvOHVdTVXLcInHhiH2qViuCBY6mrqtDrCxk0jksn95N06gAOHve2OUdtZXmHxzGztEZyxWf/aggfOplug8eTn5rAyW2/8ONLd3LnG8uwc/UEYMNHz5F2/hi+4f0IGzYZiURCwvG9HFz1JTUVpYy659lrPvbVcrZJrI28o7Wli1qlpP/Ue5EZGdF7/Dxde+jAcVj8aM/pnb9x8cAWeo6egVLRwMq3FlGYcYng/mMI6jcSpaKBiwc2s2XJa9RUltJ34p1/2nkJ/lkIEVjwj8TNzsKgp7BAIBAIBAJBZ+jIG1cg+DswfkA4n6zezc5jFxnaM4hGpYrdJ2LpHeyLh1PLNnBjuRG7omM4HpNCVkEpucXllFbW0JxUqlFrrnstGU0FzV7+Zn2bMXnF5e3OMfqJDzs8zrV4Al/OkCaxfHivYLr5ufPMZ6tZ9vsh3npoml6cRCJh8iDt35BRfUOxsTTj113H2XHsIlOG9Lzm43fEF2v3Ymos5/tX7sfT2U631sjQLiz6vx/5ZNVu3ls886Yd/+9M6KDx7PvlU+KidhDQewgqpZKEY7vxDu2NrXNL3RsjuTHxR3eRdiGasvwsKopyqa0opfkDo1Grr3stZXmZAGz53LD/LkBFUfuFCL9YNLrD41yLJ/Dl+PfSPrQJ7DMcN/9Q1n/4LEc3LGPSo2+SfjGatPPH6BIxiNkvfqYbEzpoPDu+fZtT21fi12MAXXoMuObjdxaVspHkM4dx8w/Fwd23Vb9UZkTkZMO2N73Hz+f0zt9Iu3CcnqNnEHtkB4UZl4gYM4uxD7ykiwsbOonf3nuM/T9/il+PATh6dM4HXyC4GoQILBAIBAKBQCAQCASCVlhbmDG8VzAHziZQU9dAdFwqFdV1TB3SUhCpuq6BxR+sID49l15BPoT5ezB1aAShfu78uvM4249duKZjq64QwpqF5FfunawnQF9ORxmshoqyXUl7ReyulqERXbEwMyE+I6/D2AkDuvPrruMkpOfeNBG4rKqWwrIqBvcI1AnAzfTq6oOPqyPHY1JuyrH/CZhaWhPYZxhJpw7SUFdDxsUT1FVV0H34FF1MQ201q995lPy0eLyCI3APCKP78Cm4+YdyctuvxB7ZcU3HVqtVej9rNNrPz7iHXsG2DZHW0q79IudzX1nS4XHbK2J3tQT0HoqxmQX5afEAFKRrC62FD2tdeLHHyOlc2P876ReO/ykicEbsKRR1NYR0oqDelTS/Rop67U6EwvREAMIve1+A9sFQjxHTyLh4goyL0UIEFtwUhAgsEAgEAoFAIBAIBAKDTBkawe4TsRw6d4n9p+OxMDNhRO8QXf/qPdHEpeXy8sJJzBzRR29sSWV1h/PLZFLqGhpbtReXV+n93Jyda2NhRr9uXfT68koqiE/Pxdu0fUHqynE3gpq6Bu5+6zt83Rz531Pz9PoalSoUjUpM5drb7pPxaby17HfmjOrLPRMH6cXW1jcAYGJ883xN5UYyJBJQt5GZrdFoUN6ALNR/Mt2HTyXh2B5STh8m8cQ+jM0sCIpsKfh3eudq8lPjGPfgv+k5Wj/jurqipMP5pTIZjQ11rdqry4r1frZx0gq/ZpbW+Ib30+urLM4nPzUeY7f2iyBeOe5G0FBXw4pX7sHezYeZL/xPr0+lVKJqVCA31trIGMm1nwVDmdHNIrdapWrVdzPIijsDgF8bgnNW/Fl2fvcuoYPGMWjmQ3p9JTmpANi5agtsyuTGwF/jvAT/PIQILLilnEkr4rEfonhgeFceHBnS8YC/GHllNcz4ZI/u5xGh7rw3LxKAjOIqlu9P4FRqMVX1ChwtTRka4sZDI0OwNNW/uEvKr+CbvXHEZpfRqFLTw9uBR8eEEuBq0+qYB+JyWXH4EulFVZjKjRga4sbi0aHYmBvrxf16JIkvd8caXPeikSHcN7zrNZ1zvULJT4cusediNkVV9ThbmzE63IOFQ4IwNW77T8rH285zJDGfjc8afnp6PqOE7w8kEJtdhlQqoaubLYtGhhDurfVP+3xnDKuOJuvil9w3iF5+7T+9FggEAsHfj7PZ1Ty5MYX7Il24v5/rrV7OVZNXqWDOT/G6n4f52/DORN9WcXH5tTy6LolPpvkT4Wk4M1Gp1rBoTRL+Dqa8Msa7w2O/uTODP5LKWbMwBDdr/euGlOI6lh3PJ66glvpGNcEu5tzdx5k+Xq0zK/deKmPd+WJSiuvRoMHP3pRZPZwYF3x9VeKbqVGouPuXRCZ3szf4O1aqNaw7V8TWuFLyKhXYmxsx0M+aB/q5Ym2qfy1yIbea76MLSCquQ62B7m4W3N/Pha7O7Ysf0RmVvLA5jXsve5+tP1/Mp4dydDH/Hu3FxBD7tqb429AvtAsu9tZsizrPmUsZTBzQHTOTlmvZ8iaf3QBP/UJJF5KzOJ2QDrTO6r0cJ1srzidlUVhWibOdNQCVNXUcPp+kFzeidwhL1u3jh21HGNwjCJOm606NRsMHP2/n0LlLLHl+AW4Ora+fbyYWZiaYmRhz5EIS8el5hPi66fp+3nmURqWK4b2DAQjwcKakopq1+04xY0QfnWeySq3mp+1RgNaa4WZhaWZCz0BvTsankZxdSICns67veEwKmQUljOoTetOO/0/ANzwSKwcXYg5vIyv+DN0GT0Ru0uKHXdvkz+voFaA3LufSBZ3QeGVW7+VY2jmRk3ieqtJCrOy1v7/66kpSzh7RiwvqO4JDq7/i+O8/4h8xGKMmYVWj0bDnh/dJPn2YOS9/ibXjn/s9amJmgdzEjNRzUeSnJeDq1/J+P7H1Z1TKRgL7jgDAr8dAJFIZZ/eso2v/UXo+ys3F1Px69P9T1p2fGofcxMygFQSAg4cfFUW5nNu7gV7j5mBmqf07pFYpOfTb1yCREN7kC+0fMZiT237l9M7VuAe+o5tDrVZxbu8GkEjw7X7jBXiBAIQILBDcEHr6ODC1jy+uNmYAFFTU8tB3h1CpNczs64ebnTmx2WWsi07ldFox3z00FLOmC9eUgkoWLz+MhYmc+QMDkMsk/HYslYeXHWbpQ0Pxd7HWHWf7uUze3nCGME87Fo8Opaiqnt+OpXAxs4Rli4bp5gRILqjERC7jpSk9W6030IC43BmUKjVPrTjKxaxSpvT2paubDWfSivnx4CVSCyr5v/n9DFYU3nAyjXXRabjamhmc93BCHv9efQIPOwseGBGMSqXht+MpPPbDEb66fzBhXvaM6+5JkJsNB+JyORjf8ZY6gUAgEAj+yvRwt+CObg64WrXO+supaODV7emo2rFRVak1vLcnk6SiOvwdOi64tTOhlD+Syg32pRTXsXhdMmZyKbN6OGIml7ItrpRnN6Xy1gQfRgTY6mI3XCjmk4M5BDqZ8UA/FyQSCbsTy3hnTya5lQ3cF3l9gkKDUs3L29IpqmmdGdrM27sy2JdcwcgAG+b0dOJSUR2bLpYQn1/Ll7MCMJZpixSdzKzihc2puNkYc3cfF9Bo2HCxhMVrk/lshj/hbhYG5y+vU/Le3iyufPn7elvx6hhvLuRWszm29LrO83ZCKtV61y7fchiglVXB0IiurN4bzWtLNzBrZF8szUyIS8tlW9R5ZDIpSpWa6rr6NuefPKgHZy9l8tiHvzB7ZB/qFY2sP3AaawszSitrdHE+rg48NHUo3246yJ1vfMvkQT2wMDNh36l4TsanMa5/GP3D/G/Ka9ARL90zkUfeX8FjH/7M7FF9cbS15GRcOn+ciqNnkDd3jdVmD9pZW/DU3DF8+OtO7n17GdOH9UKjgV3RMcSl5bJg/AB6BHrp5j0ek0JpZQ39unW5YTYVL949kQff+4EH3/uBWSP74O5oS3JWARsPnsHe2pKn5465Icf5pyKRSgkbOpljG5cDED5c38ogsPdQTu9czdYlrxMxZhYm5pbkp8QRc3gbUpkMtUpJQ23bGfRhQyeRnXCW3957nF5jZtGoqOfc3g2YWlhrfYWbsHf3YeDMB4lat5QfXrqL8GGTMTaz4FL0PjJiTxIycBx+3f8cAfVKxt7/EqvfeYTf3n2MXmNnY2nrSEbsSRKj/8Cza09dUTR7N28GzXyQI2u/5edX76Pb4AkgkZB06gCZsacIGTgO/4i2C0G2R+zh7QB0GzKxU/GleRlYObi0WQTP3NqW4fOf4I8VH7PilYX0HD0DqVRGXNQu8lPjGDjzIdz8uwHgE9aX8OFTuHhgMzXlJdrCcI0NxB3ZSUFaAv2n3ofTFQ8JBIIbhRCBBYIbgLudOeN7tFywLdkdS019I0sfGkY3T21WzPS+fgS52fDJ9ousP5HGgsGBAHyxMwaVWsM3Dw7BzVablTIi1J27luzji50xfLpwIAC1DUo+3xlDVzcbltw/GOOmqsHB7ra88ttJ1hxPZeHQIN0akvMr8HW01FvX9bL6WAoXMkt5ekI4cwf4687L3MSIzaczuJhVSnfvlm14DY0qvtoTy5rjqW3OWadQ8t/fz+Fqa87Sh4bqMppHhrkz7/M/+G5fPJ8tHERXd1u6utuSXVItRGCBQCAQ3Pa4WRsbzJyNSqvkv3szqahvOxOsuLqRt/dkcia74632oM0+/vRgDsYyCQoDyvKSI7mo1Bq+mBGAt502W2xiiD0LfklkyZFchvvbIJFIqGpQseRILoFOZiydE4iRVPvgd2YPRx5dm8SKk4XcEeqAo+W1bWfPKK3nzV0ZJBe3LRjuTypnX3IFs3s68uSQlkJLzpZyvjuez8HkCsZ01b6u/zuYjZWpEd/MCsTGTHvbMyrIjjt/TuCbqDyWzDJ8k/3+H1lUN7R+/b3tTPC2M0Gl0fyjRGCAKUMi+H7rYbxdHOgeoH9tGRnqx3uLZ/HjtiMs3XQAYyMjXB1tWDxzJH5ujjz96SqOxaQQ4mvYl3Tq0Aiq6xpYv/8UH6/ahYu9DTOG98LL2Z5/LVmrF7to2nC6eDizek803289jEajwcvZnufvGs/skX1v2vl3RLi/Jz++9gDfbjrAb3tPUK9oxMPJjsUzRnDPhIEYy1tuu+eN6Yeboy0/bYtiyfp9SJAQ6OXCu4/MYHz/cL15v99ymNOJGXz74sIbJgIHernwy1uLWLrxIL8fOktFTR0O1hZMGtSDh6cNx8mufV9lQcd0H34HxzZ9j72bNx5B3fX6fML6MuXJ94j+/Uei1i1FJjfGxtGVIXMW4+jhx7oPnibtwnFcuxjeJdt9xFQa6qo5t2cDf6z4H9YOLvQYNR07Vy82ffKiXuzgWYtw9OzC6Z2/cWzj92g0GuxcvRi18Hl6jZ11086/I9wDw7j77R85svZbTu/6DWVDPTbOHgyZ8wiRd9yDkbxlp8qgmQ/h4OHHqe0rOfTbEtRqNQ7uvoy+9wV6jZ19zWvYuuR1oPMicG1lGc4+Qe3G9Jk4H2tHV05s+4Uja79FIpHg5B3IHU+8Q+ig8XqxEx5+DfeAbpzbu4H9P3+KRNoU+/g7hA4e38YRBILrR4jAAsENRqPRcDKliCA3W50A3Mz4Hl58sv0iZ9OLWTA4kNLqeqJTChnfw1MnAAO42VkwspsH289lUlxVj6OVKUcv5VNRq+DRMaE6ARhgZDcP3Gxj2X4uUycCK1VqMoqrGR3uwY3k91PpeDtYMrufvp/anYMCsLc00VtXVkk1T/wYRUFFHdP6+BJ1Kd/gnIfi8yiraeCp8WF6lhbudhY8OT6MRqXwJRMIBALBP4O3dmWw91I5PnYm9PW2Yu+l8lYxh1IqeGdPJiq1hrv7OPPzqcJ251SpNby9OwN3GxN87UzYc8WcKrUGEyMpIwNtdQIwgLmxjFBXcw6mVFBep8TOXM6F3GoUKg2TQux1AjCAkVTCqCA74gtzuZhfo5c53Fk2XCjmi8O5mMqlzO3pxG/nigzGbY4twdJYykP99TOO7+hmT71Sjb259vamok6Jm5UxQf7mOgEYwMlSjo+9CYlFtQbn33SxmGPplTzY35Vvjxm+dvkn4ulsx6kf3mizf2xkN8ZGdjPYd/rHlnF9Qnz1fgZtMaQF4wewYHxrr80rYwFG9w1ldN+/nmVBoJcLHz0xt1OxwyK6MiyiY2u2pf++l3mvfYOJ/Opu292dbA2+ds14Odvz9sPTr2pOQeexdfHkxVUn2+wPGTCGkAGGM65fXH1K92/vbn30fgbt5yVy0gIiJy1od2wzwf1HE9x/dGeX/qfh5B3A9Oc+7FTstZ6DodejmaeW7ePLxZ0XW59bEdWpuKDIEQRFjugwTiKR0HP0zFa+0ALBzUaIwIJO88n2C6w5nsrSB4fqfFqb+fFgIt/+Ec8X9w6iTxcnGpVqfjuewr6YHDKKq1GoVDhYmtI/wJlFo0Kwt2x72+L0/+0CaOUdu2xfPMsPJLbygo1KzOfXqCQScytQaTT4O1szb6A/Y8I9OzynAa9v6jBmwzNjcLMzvF3QEBKJhO8fHkajgUyb8hoFgO7GKTa7DIBQz9Z+ciEetmw7m0l8ThlDgt10sd3aiN0Xm0t1fSOWpnLSi6poVKnp4qy1kmhoVCGTSjCSGd6+0hkKK+rILq1hdr8uSJvWX6dQYmwkw8fRiodH6V+MF1XWYW1mzMtTI4gMcNb9Xq/kdFoxEgn0D9T6yKnUGhRKFWbGRszqd+OLdwgEAoHgz+PzQzmsPV/M17MCCLti6/2KkwV8dzyfT6d1obeXFY0qNWvPF7M/qZyMsgYaVRrszY3o52PFg/1dsTdvO7t09o9xAKy9V/+76PvofH44UcDn0/V9dY+mVbLqbCGJhXWoNRq6OJgxp6cjo4M69rQd8sX5DmMMee52hvTSeh7s78q8CCd+PW1Y3E0pqaO3pyWLB7kjl0k6FIF/PqU9z2XzglhpYE6ZVMJ/J7euQK5UaUgpqcNcLsXKRHvL0MfLihV3dcXerPUtRHmdUjufAVuozpBUVMeYrrYsGuBGVlmDQRFYpdZwIbeGvt5WmMm1D57rG9UYySTYmctZNKDFi9XGzIj/TWttDVCjUJFd3oCrVevfT2ZZPUuO5HF3Hxe6uXb+2k8guFkcOZ9EYWkl/pd59woEgutDo9EQveVnPIN73uqlCAR/OkIEFnSaO3r5sOZ4KjsvZLUSgXecz8LV1ozefo4AvLLmBEcS85nY05spfXxRKFVEJxXy++kM8ivq+PSegTdkTauPJvPZzhjCPO14cITWVH5/XC6vrz1FRlFVh8Xm3pjZu8Nj2FqYdBhzJW2JxiubCps1i9iFldrKri7Wrb1ynay0bXnltfqxNgZim8bnl9cS4GpDUr624EByfgXzv/iDjOIqpBIJPbwdeGJ8GMHutld9TulFVU3nZs666FRWRiWTV16LiVzG6DAPnp4QrlfwLtzLgRWPdvwUNL2oCnNjI2oaGnnv97McvZSPUqXBx9GSx8Z2Y0iwW4dzCAQCgeCvycRQe9aeL2Z3YlkrEXhXQhmuVnJ6NYmzr+/IICqtkgkhdtzRzQGFSk10RhVbYkspqGrk46k35sHgmnNFfHE4l26u5rqCXwdTynlrVyaZZQ0dFpt7tRMF2GzNZB3GGGLpnEDkHTywXdDbWReTV6loNzY2v4YfT+bz+GB3/Ow79g0GqKxXklZSz8+nCskuV/DkEHeMZFph18RIanCe6gYVW+NKkUslhLtfm3j67HAP3XlllTUYjMmtVKBQaXCzNmZ/Ujnfn8gnvbQBI6mEAb5WPD3MA2dLw+J7aW0jycX1LD+eT12jWk8wBq3o/dauTPzsTVgY6cLF3BqD8wgEADlFZWw/egE3BxsiuvrctOOUVdXw9b/uwdz06h8qXS9RF5KoqK6jpFJ8FgTXR3lhDrGHt2Pt6IZXSMStXg4AxmbmTHni3Vu9jHapKi0kM/YUJTlpt3opgr8RQgQWdJoAVxuC3W3ZF5PDMxPCdVmlsVmlZBZX88DwrkgkEpLyKzickM+c/l14ZmKLB9Kc/v488O1BopMLqapTYGV2fRcz+eW1fLk7lqHBrnoFyeYO8Ofl307ww8FERod74uvUtq/VjfTL7Yg9F7PZfDodV1szpvTWXixW12uzZkyNW98smjZluNQptJ50Nc2x8o5jkwsqATiXUcL8gQG425mTlF/Br1HJPLzsEF/dP6SVVUVHVNZrC7T8fiqd0poG7hkShKe9BceTCvj9dAapBZV88+AQnSWE3KhzWcdVTfM+vOwwIR62vDGjN1X1jfx8OIkXV0Xzzpy+jOx2Y20tBAKBQPDnEOBoRldnM/YllfPkUI+WnTD5NWSWN3BfpLawWHJRHUfSKpndw5Enh7b8zZ/Vw4lFa5I4kVlFVYMKK5NrE1ebKahS8FVULoP9rHlvkq/u2mFOT0de257BTycLGBVoi087gqkhH98bRUcCcGdjAGoVKt7ZnUmEhyUzuzt2eg3P/55KfKH2wfNgP2smhrbegXQ5SrWGd/ZkUl6nZF6EE3YGsoQ7Q2fOq6rJJ/lkZhVbYkuYH+FMYH8zYvJrWHOuiOTiepbNDcTatPUaFq68pMtWnhrmQH9f/evD747nkVnWwPJ5QXpWFwKBIc5eyuTspUxG9Qm5qSLwHYN73rS5O+LrDfuJTxd1OATXT3bCWbITzhIUOfIvIQJLJBIGTLvvVi+jQwrSEnXexQLBjUKIwIKrYlKENx9vu8Dx5EIGd9Vmymw/n4VEAhMjtJkxga427H1lUqvtgKXVDVg2XZTXNCivWwQ+EKctYjI63JOKWv1MmDHhnhyMz+NQfF67InB5jeFMk8uxNjPW2R9cK7suZPH2hjOYGMl4d24kZsba10Gj0VpGSDAwv0Tvf2iaalRLDGyzbG5p7urTxQm5TMqc/l101htDgt3oH+jCou8O8emOi3z30NCrOgelSuvNm11aw/cPDyPIzRaA4aHuWJjKWRmVzPZzWUzr43tV8zaq1NQ0KIkMcOa9uZG69iHBbsz9fC+f7rjI8BD36/4dCAQCgeDWMDHEnk8O5hCdUcUgP61N0a6EMiTA+CZBNcDJjF0PhyG94juurLYRS2OtOFiruH4R+GBKBSo1jAqybVV4bXSQLYdSKzicWtGuCNwsJLaHtams1bn82Xx2KIeKehWfz/A2eO3QFvN7OSOTSjiXU82Gi8U8vCaJr2YFGBRWFSo1b+3MJCqtknA3cxYNaD+L+npRqrXXIhllDfzfZD/d+2movw2uVsZ8cjCHVWeKeHigfpavRqPhsUFumMqlRKVV8ntMCWkl9Xw23R8jmYSz2dWsPlvE08M89HyRBYIr6chr9+/EL28uutVLENzm2Di7t+vLK2ifgN5DxOsnuOEIEVhwVYzt7skXu2LYfSGLwV1dUarU/BGTQy9fR9wvs0AwlsnYE5PNibA++rwAAQAASURBVORCsktryC2rpaymQSdSNouf10NWiXZr0utr2/7D2Gyl0BYT3t/R4XGu1hP4Sn46dIlv/4jDTG7Eh3f1J9SjJYPIvMljr76x9Q1lQ6P25rTZYqFZOK5vVGJhou+LWH9F7IBAFwY0eexeTqiHHWGe9lzIKqGmobHVPO3RnG3c3dteJwA3M6OvHyujkjmRUnjVInCzp9+sSH0/QkcrU4YGu7HzfBbpxVU6f2OBQCAQ3F6MCbJlyZFc9iSWMcjPGqVKw76kciI8LXG3aRHc5DIJf1wq50RWFTnlCvIqFZTVKXUPOtXXf+lAVrn24e9buzLbjMmvamx3jjuWxXZ4nGv1BL5RHEwuZ3t8Gc8O80Auk+iE6+Z6BVUNSszqpNgayNodEWgLaIVVdxtjPjuUy7rzxa1sMsrrlLy8LY2LebWEu5nz4ZQunc5SvlZM5dr5Xa3kOgG4mTu62fP5oRxOZVXxMPoisEQiYXyINqN5eIAtNqZG/HauiN2XyhjSxYZ39mQS7mbBiABb3WtV3bS7ql6pprxOibmxFOObfH4CgUAgEAgENxMhAguuCmszY4YGu3EoIZ+ahkZOphRRUatgcq+WbVA19Y088WMUCXnl9PRxoJunHZN7+RDiYcvqo8nsPJ99TcdWXXH3p24Skl+a0hM3O3ODY5ys2ve/+2xhx97E7RWxaw+VWsOHW8/z+6l07CxM+HhBf0I89LeQutlq111cVd9qfLMHsJO19vjudi2xV4q3hZV1SCRa4bQj7K1M0Gi01hFXIwI7N3kROxh4PRyajlvb0HF2VKt5rc1ILqg0+Do7Xse8AoFAIPhrYGVqxOAuNhxJq6BWoeJUVjUV9SomhrTYDNQoVDy9MYXEwjp6eFgQ6mrOpFB7gl3M+O2s1lP4Wrjy2qH5GfQLIzxxb0OkdbBs/7vxk054E9ub39pL7CNpWluo/x3M4X8Hc1r1P7A6CYDDT/Rod56xXe347FAuiU32EM3kVjTw3OZUsssVDPS15q3xPjqB9mbS7PdrqEigXCbF2tSIWoW6w3nGdLXlt3NFJBbW4mZlTGF1I4XVjQYF/lVnilh1poh/j/bSe88KBAKBQCAQ3G4IEVhw1Uzu5cPemByOJOZzMC4PCxMjhoe0ZFysOZ5KfG45/7qjB9P76md3llR1bL8gk0qpV7QW/Yqr9YXSZlHU2kxOpL9+xdz88loScssxc2j/LX7luBuFRqPh3U1n2HEuC28HSz65Z4BepnQzIR52SCQQn1PO9L76fXHZ2hvebp72uljQxvo46ltcNLc1ZwIvXn6YmoZGflo8otUW0PTCKixMjLC7yoJ3/s7WmMplpBZWterLKdVmZbu3Ica3R6inHUeTCkgrrGxl3dE8b7NYLhAIBILbk0mh9uxLKicqrZKDKRVYGEsZ5m+j6193vpiEwjqeH+HJ1DAHvbGlte1n5gLIpBLqGluLfyW1+tcTzdm51qYy+njrf+cUVClILKzDqwMx88pxf0Xu7O3MuK6tvYtXni3kZGY1r431xr4pCzizrJ5/bUmjn48Vzwzz1ItvFlRNjFquJfIrFTyxIYXC6kamhjnwzDAPZH+SZZOtmRFu1sZklTegVGv0vHtrFSrK65QEOWkfWp/Jrua/ezOZ3t2RO3vpX+/VNjafl5QAR1ODwn5ycR1LovIY19WO8cF2+DpcW1KAQCAQCAQCwV8FsadJcNX07eL0/+zdd3iT9RbA8W/SNt177xYoo+y9FFBQGcqUDS5AEEFRVAQRB+h1oOIEBSdDpgyRvYfsTaFAS/feu00z7h+BQmkLZZm2nM/z+NzLu3KSJnnznvf8zg93e0s2nozh30tJdGvkjYXqWrI1M9+Q6K3tXnqY3pnodE5EpQKGSUQq4mprQUZeUUklLEB2gZr9F5JKbde5gRdKBfyx91JJ6wQwJGC/+Oc0U5ceJjo1986f6F1YvD+MjSdjqOVmx7zRD5ebAAZDpWuLABe2h8SRnHXt+SZk5LHzXDztg9xxsjEkazvUdcfa3JQVBy+X9OcF2BESR0JmPr2aXZut3NHGnEuJ2Ww7W7r6Z8PJaCJScuje1Pe2L9jMzUzo2siby8nZ7DoXX2rdon2GiqKudzCB2+NNfDBRKli47xJqzbW/Y2RKDnsvJNDM37mk0lgIIUT11MrXBjcbMzaFZnAwKptHgxxKVY5mXRmCX+uGRNvZhDxOxhluCN5Y1Xs9F2szMgs0pOReSxjnFGr490pF7FUP17JHqYBFx5Ip0lw7l+r1er7aHcc7GyKJzrj1DeuqLtDJglZ+tmX+c75SQdvY07okme1lZ05BsY4toRkk5ZSeY2HhUcNvr05XEvbFWh3TNkSSnFvM8BauvPGIz3+WAL6qR31Hcoq0rDyVWmr5kuPJ6LnWziLQ2YL0fA1rzqSRp772+0Kr07PkWDJgeD/YWpiW+1rVczPcgPayV9HKzxYX68qPnhJCCCGEqIqkEljcNqVSQY9mvvy2+yIAvZqXnhH34fqerDh0mQ9WHaN/60BsLMw4H5fBxlMxmCiVaLRa8m4yvL9HM19ORqUx6fd/6d8mkMJiLWuORmJnaUbGdRO5+bnY8EKX+izYGcqzc3fSs5kf1ham7AyJ51hEKo819qFNnftT6XszWflqftkVCsAjwZ4cCksus42TtXlJbBO7N2Lsgr2MXbCHQe1ro9frWXbgMiZKBROeaFiyj7W5GS8/3pDP/j7Fy7/uo0czPxIy8ll6IIwgD3sGtL1Wdf1St2BORKQy86/jnIlOJ9DNlnNxGWw4EU1tdzvGdQsu2TY9t5DD4SmlYqrI+McaciIylRkrjtK3VQABbrb8eyGR/ReT6NXcj5a1XG/79fJ1tmFct2C+3xLC6J/28FQLP7ILill+MByVqQmTn2xy28cUQghRtSgVCno0cOT3I4Zz4o3D6jsG2rHyVCozt0TTr7Ez1ioTQpPz2RyagYkSNDpKJfJu1L2+I6fi83h9bTj9GrtQqNGx7mwadhYmZFw3kZufoznPtXbnl8NJjFp6ke71HbFWmbArPIvjsbl0q+tA62pQ6XsvmZoomNzFhxkbIxm/Moy+jZ2xvDKB2tGYXB6pY0/XK4nVf86lcymlAGdrUwKcLdgcWrZNR2NPq5Jez3vCsygo1tGptl3JHAB3a1hLN/6NzOaHffFcTiugoYc1p+Pz2HIhg5Y+NvRoYKiAdrQ05aWOnny9J56xyy/xVENn9MD2ixmEJhcwpLkrjT3vfM4HIYQQQojqRpLA4o482dyf3/dcxNfZhsZ+pS/kWtVy5cOBrVm49yI/7wrFzESJh4MVLz7agABXW95YfJBDYcnU93Io99hPtfAnr1DD6qMRfL3pDO52lvRpFYCPkzXTlh0pte2oR+oT6GbLioOX+X3PRfTo8XGy5rWejel/QyuK/8qZmHQKrlyo/rzrQrnbNA9wLkm41vN04IcXHuLHbeeYv+M8ZiZKGvk68dJjwWUmQ+vXOhArc1MW77vEVxtOY2+loldzP158tEHJxHFgSKz+MrYzP+04z7azceQUqnG1tWRIhzo837leSdsIMFTcfrDqWKmYKuJkY86CFzuzYGcoO8/Fk31MjaeDFa90b8SQ9rXv6PUCGPFQED5O1izeH8b3W8+hMlXSIsCFcd2Cy7SIEEIIUT31bODEH0eS8XUwp9ENybeWvra8392fRceS+fVwEmYmCtxtVYxu54G/owVT1kdwODqnpDrzRr2CnchV61h7NpVv98bjZmNG70bOeNureHdjVKltn2/rQYCzBatOpfLHUUNS2ttexaudvOjb2OX+PPkqrlNte77uV5vfjySx8GgyWp0ef0dzXuvsTd/GziWtpY7GGEZYpeVp+GhrTLnHmtrNtyQJ/O3eOBJzilnu1eCeJYHNTZV83a82C48ms/1SJtsuZOJiY8azrd15prUbyuvaYD3d1BUPWxWLjycz/2ACChTUdrHgvSf86Fa3bLsMIYQQQoiaTKHX6+/BXMuiqlMoFKQtfcvYYdQ4CRl59P9qKz2b+fJu/5bGDueO7ToXz5qjkcx55tYT5Rnbgh3n+XnXBb5/viMtAm+/8vhGzkM+Q74GhRCiYgqFgqRvBxo7jBojIVvNoN/P072+I+885nfrHaoxvV5Pz5/OsmhEfZxrUDuFDefT+d+2mLueLM594gqj/wYJ8PMlKubOJm0W4kb+vj5ERpd/g6i68/UPIDY66tYbCmFkPn7+xERFGjsMUUVJJbAQD7j8Ig2rDkfQMvDBrH4SQgghxP2x9mwaDlamOFnJJUdVVVMTdkLca5JUE0LUBPKLTIh7ID4jn02nYvCwt6RZQPVKphZptLSp7cqwjkHGDuWmLsRnEpGSQ1hS9q03FkIIIaq4hGw1m0Mz8LA1o6m3jbHDuS80Oj1f9K5V0k6iuovOKOJ8Uj4hCXnGDkUIIYQQ4rZJEliIe+BkVBono9J4JNir2iWBHa3NGflwXWOHcUubT8fy579hxg5DCCGEuCdOxedxKj6PzrXta2wS+Ommd9+2qSo5Ep3DnD1xxg5DCCGEEOKOSE/gB4T0BBaifNITWAghbk56Agtxf1SFnsBCCCGEeHAojR2AEEIIIYQQQgghhBBCiPtHksBCCCGEEEIIIYQQQghRg0kSWAghhBBCCCGEEEIIIWowSQILIYQQQgghhBBCCCFEDSZJYCGEEEIIIYQQQgghhKjBJAkshBBCCCGEEEIIIYQQNZhCr9frjR2EuP8C/HyIiokzdhhCVDn+vt5ERscaOwwhhKiyAny9iYqNN3YYQtQ4/j5eRMrvcyGEEEL8RyQJLB5Yp0+fZvz48WzcuBFbW1tjh3NfbdiwgXnz5vHXX39hampq7HCEEEKI/8SmTZt4//33mTVrFt26dTN2OOIe0Gq1zJ49mx07djBv3jwCAwONHZIQQgghRLUg7SDEA0mn0/Hhhx8yefLkGp8ABujRowcODg4sXbrU2KEIIYQQ951er2fevHl88skn/PLLL5IArkFMTEyYMmUKo0aNYvjw4Rw8eNDYIQkhhBBCVAuSBBYPpFWrVmFiYkKfPn2MHcp/QqFQ8O677/Ldd9+Rnp5u7HCEEEKI+0atVjNlyhS2bt3KsmXLCA4ONnZI4j4YNGgQX3zxBa+//jorVqwwdjhCCCGEEFWeJIHFAycrK4s5c+bw7rvvolQ+OB+BoKAgevfuzZdffmnsUIQQQoj7Ij09neeee47CwkIWLVqEu7u7sUMS91H79u1ZtGgR8+fP57PPPkOr1Ro7JCGEEEKIKuvByYAJccW3335L165dadSokbFD+c9NnDiRXbt2cfr0aWOHIoQQQtxTYWFhDBo0iNatWzNnzhwsLS2NHZL4D9SqVYtly5Zx5swZJk6cSF5enrFDEkIIIYSokiQJLB4ooaGh/PPPP7z22mvGDsUobG1tef3115k5cyY6nc7Y4QghhBD3xL59+xg5ciQTJkzgtddee6BG+ghwdHTk559/xsHBgeHDh5OYmGjskIQQQgghqhz5hSweGHq9nlmzZjFx4kQcHR2NHY7R9O3bF6VSyV9//WXsUIQQQoi7tmTJEt5++22+/fZb+vbta+xwhJGoVCo++ugjnnzySQYNGsSZM2eMHZIQQgghRJUiSWDxwPjnn3/Izc1l8ODBxg7FqJRKJe+++y5fffUV2dnZxg5HCCGEuCMajYZZs2axcOFClixZQqtWrYwdkjAyhULB6NGjmTFjBmPGjGHTpk3GDkkIIYQQospQ6PV6vbGDEOJ+y8vLo0ePHnz11Ve0bNnS2OFUCTNmzEClUjF9+nRjhyKEEELcltzcXF577TW0Wi1z5szBzs7O2CGJKiYkJITx48czbNgwXnzxRRQKhbFDEkIIIYQwKqkEFg+EuXPn0q5dO0kAX2fSpEn8888/XLhwwdihCCGEEJUWGxvL0KFD8fb25scff5QEsChXw4YNWb58OZs3b+btt99GrVYbOyQhhBBCCKOSJLCo8S5fvsyKFSt44403jB1KleLk5MSECROYNWsWMiBACCFEdXDixAmGDBnCoEGDeO+99zAzMzN2SKIKc3d3Z/HixeTn5/Pcc8+Rnp5u7JCEEEIIIYxGksCiRtPr9Xz00UeMHTsWNzc3Y4dT5QwZMoScnBw2bNhg7FCEEEKIm/r7778ZP348H330ESNHjpTh/aJSLC0t+frrr2nVqhWDBg0iPDzc2CEJIYQQQhiFJIFFjbZ9+3bi4+MZMWKEsUOpkkxMTJg+fTqfffYZeXl5xg5HCCGEKEOv1/Ptt9/y1Vdf8dtvv9G5c2djhySqGaVSyeuvv87LL7/MiBEj2L9/v7FDEkIIIYT4z8nEcKLGKiwspGfPnsycOZOOHTsaO5wq7c0338TDw4PJkycbOxQhhBCiRGFhIdOmTSMuLo7vv/8eFxcXY4ckqrkjR44wadIkXn75ZYYNG2bscIQQQggh/jNSCSxqrAULFtCoUSNJAFfCm2++yfLly4mIiDB2KEIIIQQAKSkpPPPMMygUCv744w9JAIt7onXr1vz5558sXLiQWbNmodFojB2SEEIIIcR/QpLAokaKjY1l4cKFTJkyxdihVAtubm68+OKLfPTRRzJJnBBCCKMLDQ1l8ODBdOrUidmzZ2Nubm7skEQN4ufnx7JlywgPD+ell14iNzfX2CEJIYQQQtx3kgQWNdInn3zCM888g7e3t7FDqTZGjhxJXFwcO3bsMHYoQgghHmC7d+/m+eefZ/LkyUyYMEEmgBP3hZ2dHT/99BOenp4MHTqU2NhYY4ckhBBCCHFfSRJY1Dj79u3j/PnzjB492tihVCsqlYrp06fz8ccfU1RUZOxwhBBCPGD0ej2///4777zzDnPnzqVXr17GDknUcGZmZnzwwQc8/fTTDBkyhBMnThg7JCGEEEKI+0aSwKJGUavVzJo1i2nTpsnQ0TvQsWNHgoODWbBggbFDEUII8QApLi7m/fffZ+XKlSxbtoxmzZoZOyTxgFAoFDz77LPMmjWLl156ifXr1xs7JCGEEEKI+0KSwKJGWbhwIb6+vjz66KPGDqXamjJlCn/88QdxcXHGDkUIIcQDIDs7mxdffJGEhAT+/PNPaeUkjKJLly789ttvfPnll3z77bcyR4IQQgghahxJAosaIykpiZ9++olp06ZJ/8C74OPjw8iRI/n000+NHYoQQogaLjo6msGDBxMUFMTcuXOxsbExdkjiAVa/fn2WL1/O3r17mTx5MoWFhcYOSQghhBDinpEksKgxPv/8cwYNGkRgYKCxQ6n2Ro8eTUhICPv37zd2KEIIIWqoI0eOMHToUJ555hmmTZuGiYmJsUMSAhcXF/744w8Ann32WVJTU40ckRBCCCHEvSFJYFEjHD16lMOHDzNu3Dhjh1IjWFhYMHXqVGbNmoVarTZ2OEIIIWqY1atX88orr/DZZ58xdOhQY4cjRCkWFhZ88cUXPPTQQwwaNIgLFy4YOyQhhBBCiLsmSWBR7Wm1WmbOnMmUKVOwtrY2djg1RteuXfH29mbRokXGDkUIIUQNodPp+PLLL/nhhx9YtGgRHTt2NHZIQpRLoVAwceJEXnvtNZ577jl2795t7JCEEEIIIe6KQi+zHohqbvHixWzatIk//vhDegHfYxEREQwZMoS///4bNzc3Y4cjhBCiGsvPz2fKlCmkp6fz7bff4uTkZOyQhKiUEydOMHHiRMaMGcMzzzwjvzeFEEIIUS1JJbCo1tLT0/nuu++YPn26/CC/DwIDAxk4cCCzZ882dihCCCGqsaSkJEaMGIG1tTW//vqrJIBFtdK8eXOWLl3KypUref/99ykuLjZ2SEIIIYQQt02SwKJa++qrr+jVqxf16tUzdig11ksvvcTBgwc5duyYsUMRQghRDYWEhDBo0CC6d+/O//73P1QqlbFDEuK2+fj48OeffxIfH8/YsWPJzs42dkhCCCGEELdFksCi2jpz5gw7duzglVdeMXYoNZq1tTVvvfUWM2fORKvVGjscIYQQ1ci2bdsYNWoU06ZN48UXX5RRO6Jas7GxYe7cudSqVYvBgwcTHR1t7JCEEEIIISpNksCiWtLpdMycOZPXXnsNOzs7Y4dT4/Xq1QsbGxuWLVtm7FCEEEJUA3q9nvnz5/Phhx8yf/58nnjiCWOHJMQ9YWpqyvTp0xkxYgRDhw7l6NGjxg5JCCGEEKJSJAksqqU1a9ag1+vp37+/sUN5ICgUCqZPn853331HRkaGscMRQghRhanVaqZNm8aGDRtYvnw5jRs3NnZIQtxzw4cP59NPP2XixImsXr3a2OEIIYQQQtySQq/X640dhBC3Izs7m549e/LDDz/QpEkTY4fzQJk1axZqtZoPP/zQ2KEIIYSogjIyMnjllVewtbXl888/x9ra2tghCXFfhYWFMW7cOHr27MmkSZNQKqXGRgghhBBVk/xKEdXOt99+S5cuXSQBbASvvPIK27dv5+zZs8YORQghRBVz+fJlBg8eTJMmTfjuu+8kASweCHXq1GH58uUcPXqUV199lYKCAmOHJIQQQghRLkkCi2rl4sWL/P3337z++uvGDuWBZGdnx6RJk5g1axY6nc7Y4QghhKgiDhw4wIgRIxgzZgxvvvmmVEOKB4qTkxO//fYblpaWDB8+nKSkJGOHJIQQQghRhvxCF9WGXq9n5syZTJgwAScnJ2OH88AaMGAAGo2GtWvXGjsUIYQQVcDy5ct54403+Oqrrxg4cKCxwxHCKFQqFZ9++imPP/44gwcP5ty5c8YOSQghhBCiFEkCi2pj48aNZGVlMWTIEGOH8kBTKpXMmDGDL774gpycHGOHI4QQwki0Wi2ffPIJP//8M4sXL6Zt27bGDkkIo1IoFIwbN463336bF154gW3bthk7JCGEEEKIEjIxnKgW8vLy6NmzJ7Nnz6Z169bGDkcA77zzDjY2NkydOtXYoQghhPiP5eXlMXnyZPLz8/nmm29wcHAwdkhCVCmnT59mwoQJPPPMM4waNQqFQmHskIQQQgjxgJNKYFEt/Pjjj7Rq1UoSwFXI5MmTWbt2LZcuXTJ2KEIIIf5DCQkJDBs2DBcXF37++WdJAAtRjiZNmrBs2TLWr1/PO++8g1qtNnZIQgghhHjASRJYVHmRkZEsW7aMt956y9ihiOs4OTnx8ssvM2vWLGRAgRBCPBhOnz7NoEGD6NOnDzNnzsTMzMzYIQlRZXl6erJ48WIyMjIYNWoUmZmZxg5JCCGEEA8wSQKLKu/jjz9m9OjRuLu7GzsUcYOhQ4eSnp7Opk2bjB2KEEKI+2zDhg2MHTuW999/nxdeeEGGtwtRCdbW1nz33Xc0atSIwYMHExERYeyQhBBCCPGAkiSwqNJ27txJVFQUzz77rLFDEeUwNTVlxowZfPrpp+Tn5xs7HCGEEPeBXq/nhx9+4PPPP+fXX3+la9euxg5JiGrFxMSEKVOmMHr0aIYPH87BgweNHZIQQgghHkCSBBZVVlFRER999BHTp09HpVIZOxxRgdatW9OyZUt+/PFHY4cihBDiHlOr1bz11lvs2LGDZcuWUb9+fWOHJES1NXDgQL788ktef/11VqxYYexwhBBCCPGAkSSwqLJ++eUX6tWrx8MPP2zsUMQtvPXWWyxdupSoqChjhyKEEOIeSU9P59lnn6WoqIiFCxfi5uZm7JCEqPbatWvH4sWLWbBgAZ9++ilardbYIQkhhBDiASFJYFElxcfH89tvv/H2228bOxRRCe7u7owePZr//e9/xg5FCCHEPRAWFsbAgQNp27Ytc+bMwdLS0tghCVFjBAYGsmzZMkJCQpgwYQJ5eXnGDkkIIYQQDwBJAosq6ZNPPmHEiBH4+voaOxRRSc8++ywRERHs3LnT2KEIIYS4C3v37mXkyJG88sorTJo0CaVSfi4Kca85ODiwYMECnJycGDZsGAkJCcYOSQghhBA1nPyqF1XOgQMHOHv2LGPGjDF2KOI2qFQqpk+fzscff0xRUZGxwxFCCHEHFi9ezNSpU/nuu+/o06ePscMRokZTqVTMmjWLPn36MHjwYE6fPm3skIQQQghRg0kSWFQpxcXFzJw5k6lTp2JhYWHscMRtevjhhwkKCuLXX381dihCCCFug0ajYebMmSxZsoQ///yTli1bGjskIR4ICoWCF154gRkzZvDiiy+yceNGY4ckhBBCiBrK1NgBCHG9xYsX4+npSbdu3YwdirhDU6dO5emnn6ZPnz54enoaOxwhhBC3kJOTw2uvvYZer2fp0qXY2toaOyQhHjjdunXDy8uL8ePHExkZybhx41AoFMYOSwghhBA1iFQCiyojJSWFuXPn8s4778iP3mrM19eXESNG8Omnnxo7FCGEELcQExPD0KFD8fPz48cff5QEsBBGFBwczLJly9i2bRtvvfUWarXa2CEJIYQQogaRJLCoMmbPns2AAQOoVauWsUMRd2nMmDGcPn2aAwcOGDsUIYQQFTh+/DhDhw5lyJAhzJgxA1NTGSAmhLG5u7uzaNEiioqKePbZZ0lPTzd2SEIIIYSoISQJLKqE48ePc+DAAcaPH2/sUMQ9YGFhwdSpU5k1axbFxcXGDkcIIcQN1q1bx8svv8zHH3/MiBEjjB2OEOI6lpaWzJkzhzZt2jBo0CDCwsKMHZIQQgghagBJAguj02q1zJw5kzfeeAMbGxtjhyPukW7duuHu7s7ixYuNHYoQQogrdDodX3/9NXPmzOH333+nU6dOxg5JCFEOpVLJa6+9xoQJExg5ciR79+41dkhCCCGEqOYUer1eb+wgxINt6dKlrFu3jsWLF0sv4BomPDyc4cOHs379elxcXIwdjhBCPNAKCwt5++23SUxM5Pvvv8fZ2dnYIQkhKuHo0aO8+uqrjB8/nuHDhxs7HCGEEEJUU1IJLIwqIyODb775hhkzZkgCuAaqXbs2/fv354svvjB2KEII8UBLSUlh5MiRmJiY8Pvvv0sCWIhqpFWrVvz5558sXryYWbNmodFojB2SEEIIIaohSQILo5ozZw7du3enfv36xg5F3Cfjx49n3759nDhxwtihCCHEAyk0NJRBgwbRpUsXZs+ejbm5ubFDEkLcJj8/P5YuXcrly5d56aWXyM3NNXZIQgghhKhmJAksjCYkJIRt27bxyiuvGDsUcR/Z2Njw5ptvMnPmTLRarbHDEUKIB8rOnTt5/vnnefPNN3n55Zdl1I0Q1ZidnR0//fQT3t7eDBkyhNjYWGOHJIQQQohqRJLAwij0ej0zZ85k0qRJODg4GDsccZ899dRTWFhYsHLlSmOHIoQQDwS9Xs9vv/3GjBkzmDdvHj179jR2SEKIe8DU1JT33nuPwYMHM2TIEI4fP27skIQQQghRTUgSWBjF2rVrKS4uZsCAAcYORfwHFAoFM2bM4OuvvyYzM9PY4QghRI1WXFzMe++9x19//cWyZcto2rSpsUMSQtxDCoWCkSNH8tFHH/Hyyy/z999/GzskIYQQQlQDCr1erzd2EOLBkpubS/fu3fn+++/lwvQB8+GHH6LX63nvvfeMHYoQQtRIWVlZvPrqq1hYWDB79mxsbGyMHZIQ4j66ePEi48aNo2/fvkycOFFavgghhBCiQlIJLP5z33//PQ8//LAkgB9Ar7zyCps3b+bcuXPGDkUIIWqcqKgoBg0aRL169fj+++8lASzEA6Bu3bosX76c/fv38/rrr1NYWGjskIQQQghRRUkSWPynwsLCWL16NZMnTzZ2KMIIHBwcmDRpEjNnzkQGIQghxL1z+PBhhg0bxvPPP8/UqVMxMTExdkhCiP+Ii4sLf/zxB0qlkmeeeYaUlBRjhySEEEKIKkiSwOI/o9frmTVrFuPHj8fFxcXY4QgjGTBgAGq1mnXr1hk7FCGEqBFWrVrFpEmT+PzzzxkyZIixwxFCGIG5uTmzZ8+mU6dODB48mNDQUGOHJIQQQogqRpLA4j+zZcsW0tLSGDZsmLFDEUZkYmLCjBkzmD17Nrm5ucYORwghqi2dTsfs2bP58ccfWbhwIR06dDB2SEIII1IoFEyYMIHJkyfz/PPPs2vXLmOHJIQQQogqRCaGE/+JgoICevbsySeffELbtm2NHY6oAqZOnYqDgwNTpkwxdihCCFHt5Ofn8+abb5KVlcW3336Lo6OjsUMSQlQhJ0+eZMKECYwePZpnn31WJowTQgghhFQCi//GTz/9RLNmzSQBLEpMnjyZ1atXEx4ebuxQhBCiWklKSmL48OHY2dnxyy+/SAJYCFFGs2bNWLZsGatWreK9996juLjY2CEJIYQQwsgkCSzuu+joaJYsWSIVn6IUFxcXXnrpJWbNmiWTxAkhRCWdPXuWQYMG0aNHDz7++GNUKpWxQxJCVFHe3t78+eefJCYm8uKLL5KdnW3skIQQQghhRJIEFvfdxx9/zKhRo/Dw8DB2KKKKGT58OKmpqWzdutXYoQghRJW3detWRo8ezTvvvMOLL74ow7uFELdkY2PD3LlzCQoKYvDgwURFRRk7JCGEEEIYiSSBxX21e/duIiIieO6554wdiqiCTE1NmT59Op988gkFBQXGDkcIIaokvV7PTz/9xKxZs1iwYAGPP/64sUMSQlQjJiYmTJs2jWeeeYZhw4Zx5MgRY4ckhBBCCCOQJLC4b9RqNR999BHTpk2T4aqiQm3btqVp06bMnz/f2KEIIUSVo1armTZtGps2bWL58uU0atTI2CEJIaqpoUOH8vnnn/Pqq6/y119/GTscIYQQQvzHJAks7pvffvuNWrVq0blzZ2OHIqq4KVOmsHjxYmJiYowdihBCVBnp6em88MIL5OTksGjRItzd3Y0dkhCimuvQoQMLFy5k7ty5fPHFF+h0OmOHJIQQQoj/iCSBxX2RmJjIzz//zLRp04wdiqgGPDw8eOGFF/j444+NHYoQQlQJ4eHhDB48mGbNmvHNN99gZWVl7JCEEDVE7dq1WbZsGcePH+fVV18lPz/f2CEJIYQQ4j8gSWBxX3z66acMGzYMPz8/Y4ciqonnn3+e8PBwdu/ebexQhBDCqA4cOMDIkSMZN24cb7zxBkql/FwTQtxbTk5O/Prrr1hbWzNixAiSkpKMHZIQQggh7jO5qhD33KFDhzh58iQvvviisUMR1YhKpeKdd97h448/Rq1WGzscIYQwiqVLl/LGG28wZ84cBgwYYOxwhBA1mEql4n//+x/du3dn0KBBhISEGDskIYQQQtxHkgQW95RGo2HWrFm8/fbbWFpaGjscUc107tyZwMBAfvvtN2OHIoQQ94Veryc8PLzMcq1Wy8cff8xvv/3GkiVLaNOmjRGiE0I8aBQKBS+++CLvvPMOo0aNYtu2bWW20Wq1REREGCE6IYQQQtxLkgQW99SSJUtwcXHh8ccfN3YoopqaNm0aP//8M4mJicYORQgh7rk9e/YwY8aMUstyc3MZP348Fy9eZNmyZfj7+xspOiHEg+rxxx9nwYIFfPjhh8yfPx+9Xl+yLicnhyFDhpCbm2vECIUQQghxtyQJLO6Z1NRUfvjhB6ZPn45CoTB2OKKa8vPzY+jQoXz22WfGDkUIIe65P/74g4EDB5b8Oz4+nqFDh+Lu7s78+fOxt7c3YnRCiAdZo0aNWL58ORs2bGDatGkl7bkcHBxo164da9asMW6AQgghhLgrkgQW98wXX3xBv379qF27trFDEdXc2LFjOXHiBIcPHzZ2KEIIcc+Eh4cTGhpKz549ATh16hSDBw9mwIABfPDBB5iZmRk5QiHEg87Dw4NFixaRnZ3NqFGjyMjIAGDkyJEsXLgQnU5n5AiFEEIIcackCSzuiVOnTrF3715efvllY4ciagBLS0umTJnCzJkz0Wg0xg5HCCHuicWLFzNo0CBUKhUbNmxg3LhxfPjhhzz33HMygkYIUWVYW1vz7bff0qRJEwYPHszly5dp2bIlVlZW7Nu3z9jhCSGEEOIOKfTXN3wS4g7odDoGDhzIyJEj6du3r7HDETWEXq/n+eefp2vXrowcOdLY4QghxF3Jycmha9eurFu3jpUrV7Jq1Srmzp1L/fr1jR2aEEJUaOXKlXz55Zd88cUXJCQksHHjRubPn2/ssIQQQghxB6QSWNy1VatWoVKp6NOnj7FDETWIQqFg+vTpfP/996SlpRk7HCGEuCurVq2iQ4cOfP755+zevZvly5dTv359tFot586dk/Y3Qogqobi4mG3bthEXFwfA008/zVdffcUbb7xBQUEBISEhREREGDlKIYQQQtwJqQQWdyUrK4uePXsyf/58goODjR2OqIE++eQTsrOz+fjjj40dihBC3BGtVku3bt2wsbEhMDCQ4cOHc+rUKY4ePcrx48dxd3end+/ejB071tihCiEecHl5ebz77rscOnQIlUpFq1ataN26NZ6ensycORN7e3saN27MjBkzjB2qEEIIIW6TJIHFXZk5cyZarZb333/f2KGIGio3N5fu3bvzww8/0KRJE2OHI4QQt23p0qW89957eHt7k5GRQUBAAK1ataJNmza0bNkSJycnY4cohBCl6PV6IiMjOXr0KEeOHOHo0aMUFBRQXFxMfn4++/btk+8uIYQQopoxNXYAovoKDQ1l48aN/PPPP8YORdRgNjY2vPHGG3zwwQesWLECpVK62Aghqhe1Wk3Xrl0ZPHgwLVq0wNbW1tghCSHETSkUCgIDAwkMDGTgwIEAxMXFcejQIZYtW0ZmZqYkgYUQQohqRiqBRaWEh4ezZs0aJk+eDBiqA0aMGMFTTz3FkCFDjBydqOl0Oh3Dhg1jwIABJRci+fn5vPvuu3zxxRdGjk4IIYQQQgghhBCiapOSOlEply5dKjUJxPr16ykoKChJyAlxPymVSmbMmMGcOXPIysoCDInhHTt2GDkyIYQQQgghhBBCiKpP2kGISsnKysLBwQEw9Gj9/PPP+frrrzExMTFuYOKBERwcTLdu3fjmm2949913sba2pri4GLVajUqlMnZ4ooYI8PUmKjbe2GEIcc/4+3gRGRNn7DCEEPeBnLNETSPnLCGEuL8kCSwqJSsrC3t7ewDmzp1Lhw4daN68uZGjEg+aSZMm0atXLwYOHEj9+vWxt7cnKysLV1dXY4cmaoio2HhSfhlr7DCEuGdcX/jR2CEIIe6TqNh4Er/qY+wwhLhnPF5ba+wQhBCiRpN2EKJSMjMzsbe3Jzw8nFWrVpX0Bo6MjGTMmDEcPXrUyBGKmuqTTz5hzpw5qNVqHB0deeWVV5g5cyZ6vb4kCSyEEEIIIYQQQgghKiZJYFEpV5PAH3/8MePGjcPR0ZGffvqJwYMH07FjR6kKFvfNCy+8QHh4OH369OHYsWMMHDiQgoIC1q9fj729PZmZmcYOUQghhBBCCCGEEKJKk3YQolKysrKIjo4mMTGRFi1aMHDgQBwdHVm5ciW+vr7GDk/UYG5ubnz77bds3ryZSZMm8dhjjzF58mSmTp1KvXr1JAkshBBCCCGEEEIIcQtSCSwqJSMjg9WrV1O3bl3GjRvHM888w88//ywJYPGfeeKJJ1i/fj2FhYW88847BAYGkpCQIO0ghBBCCCGEEEIIIW5BKoFFpURGRpKdnY1er2fdunW4uLgYOyTxALrakuTAgQNMmzaNhIQELl++bOywhBBCCCGEEEIIIao0SQKLSvH09OT111/n6aefNnYoQtC+fXs2bNjAa6+9hoWFhbHDEUIIIYQQQgghhKjSJAksKmXVqlXGDkGIUiwtLZk3b56xwxBCCCGEEEIIIYSo8iQJLIQQQlQhP++N4Jf9kWWWm5kocLRS0cTHnpHt/anjZlNmm3Un4/l00wVsLUxZO6ED5qYmJetmrT/PxrOJ9GvuxRtP1Kvw8aeuOsOeS6m82rUOg1rfXd/3sORcxvx+jK4N3Jj+ZIO7OtZ/4c9D0Xy3M5z9bz9S6X3iMwv4cfdljkdnUqDWUs/DhtEPB9Lcz/E+RiqEEFXDL0dS+O1oapnlZkoFDpYmNPG0YkQLZ2o7lx259fe5DD7fnYituZK/ngnC3PTadDUf74hn04Us+jZ04PVOnhU+/jubYtgbkcvEju4MbOJ0V88lPK2QsasiebSOHdMe9bqrY/0Xlp5M44cDyex5qfLn12eXXiYio6jcdStH1sHNxuxehSeEEKIKuu9J4AA/H6Ji4u73wwjxn/H39SYyOtbYYZQrwNebqNh4Y4chxE35+3gRKeeFW+rTzIumPvYl/y7W6onJyGfV8Tj2haUyd3gL6nrYltrnnzMJWKpMyCnUsP18Mj0bX7twfrVbHY5EprPmRDzdGrjTzM+hzGNuP5/MnkuptPBzYGArn7uKv0ij5YO/z6HW6u7qOP+VfZdSmbf79nqMp+QU8fLiExSotTzdygcHKzNWH4/j1T9P8cXgJrQOuLuEhBBCVBdPBTvQ1NOq5N/FWj0xWWpWn81gf2QO3/UNoK5r6UTwhtAsLM2U5BTp2BGWTY/6DiXrJnZ052hsHmtDMnm0jj3NvKy40Y6wbPZG5NLcy4qnG9/djbcijY4Pt8Wj1urv6jj/lf2ROfx0KPm29lFrdURnFdHa15on6tqXWW9nblLOXkIIIWqS+54EjoqJI2PNzPv9MEL8Zxz7vmvsECoUFRtP8ryRxg5DiJtyG7fQ2CFUCw297HiikUeZ5fU9bJm+JoT5eyP4fGCTkuVRaXmcjctmZHs/VhyNZc2J+FJJYFsLM958oh5TVp3hk42h/D6qdalK4ayCYr7aehFrcxPe6dUAhUJxV/H/sDOc2PSCuzrGjbILi7GzuLdVSlqdniWHopm/JwKt/vYu/n/dH0lKThE/PdOSYC87AJ5o6M7In4/w5ZZLLBnT5q5fRyGEqA4auVvyeDmJxfquFszYEsfPR1L4tOe10SVRGUWEJBUworkzK8+ks+5cZqkksK25CZM7eTB1Yyyf7Urg10GBpSqFswo1fL0vEWuVkqmPet31d+28g8nEZanv6hg3yinSYnuPE6tanZ4/T6bx8+EUbjdfHZmuRquDDv425f6thBBC1HzKW28ihBBCiKqiSz1XrFQmnIrNLLX8n9OJAHSo7UybACdC4rMJS84ttc1DQS483tCdmIwCft4bWWrdnG2XyMgv5tWuQXjY392Eiwcvp7HqWBxjOgXe1XGuOhObxQd/n2P4/MP35HhXZRcW88wvh5m3+zId6jhT74bK6pvR6vRsOZdEYx/7kgQwGJLtvZt6Ep2eT0h89j2NVwghqpvOtWyxMlNyOiG/1PINoVkAtPe3obWvNSFJBYSnFZbapmOALY8F2RGbpeaXI6VbTnyzL4mMAi0TO7rjYXt3NwcPRefy15kMRrVxvavjXHUmMZ+Z2+IY+Wf4PTneVTlFWp5ffpmfDqXQ3t+Geq63d66++voGOpnf07iEEEJUH9ITWAghhKhGFAoFSoUCje5amwWtTs+mkERszE0J9rTjkfpu7LmUypoTcWX6/77WLYijkRksPRzDY8HuBLnbcCA8jS0hSTxUx4VeTSruvVgZGflqPv4nlCcaefBIPVe+33lnF8H5ag2bQ5JYczyOsJQ8rFQm9Gx8rTJ6wA8HSMwuvMkR4NuhzWjhX/EQ4bxCDcUaPR/0DqZbsDsTFp+odHwRqXkUqLU0vC4BfFV9T8Oy8wk5NPKWaishxIPLcM4Cje5a2apWp2fLxSxsVEoauFnSpZYdeyNyWRuSUab/76sPeXAsNo/lp9J4LMiOOi4WHIzKZeulbDoG2NDzuurhO5FZoOF/O+N5vK49XWrZMvfA7bVYuCq/WMfWi1msCckgPK0IKzMlPepf+/4ftCiMxJzimx7j695+NPe2rnB9bpGWYq2e97p50TXInlfWRt1WjOFphl7Ata4kgfOLdViaKmTEihBCPEAkCSyEEEJUI+cTsskt0tDM99rF5cHLaaTlqunRyANTEyUPBTljbqpkc0gS4x+pjZXq2uneztKMyY/X5Z3VZ/ly60W+GtyUL7ZcxMHSjLd7VDxhXGV9suECZiZKXn8siOyCm1/wludySi6rT8Sz6Wwi+Wot9TxsmdK9Ht2C3Uo9j1e71aFArb3psQJcKr6YBnC1M2fp2LYo7+ACODnHcDHtZlu2osrVxrAsIevetsMQQojqJjS5gFy1rlS/4EPRuaTla+hezx5TEwUdA20xN1Ww5WI249q7Y2V2bbCqnYUJr3Xy4N3NcczZm8jsJ/34cm8i9hYmvNX57m5aAny6KwGViZJJD7uTXXjzc0p5LqcVsiYkky0Xs8gv1lHP1YI3O3vQNci+1POY2NGdguKb98j3d7x5ha6rjRmLh9W+o3MWQFhaIRamCn45ksr2sCxyinTYqJQ8Xteese3csDSTQcJCCFHTSRJYCCGEqIIKirVk5l/rT1ik0RGakMMPuwyVtc+09y9Zt/50AgDdgt0AsFKZ0qG2MzsvpLDtXDK9m5We5bxLPVe6NnBj+/lkJi45QUJWIR/1a4SjtequYl59PI5/w1P5dmhzrM1NbysJnJBZwMz15zkVm4WVyoTHgt3p28yrzOR3V3Wqe/fDdk2Vd37Bm1ekAcBCVbbfo8WVC+nCW1zwCyFETZFfrCOzQFPy7yKtngvJBcw9aKisHdnCuWTdP1daQXStYxg1YWWmpL2fDbsu57D9UhZPBZcewdG5lh2P1slhR1g2r66LIjGnmJlPeONodXeXsmvOZnAgKpc5vf2xVpncVhI4IVvNRzviOZ1QgJWZkm5BdvQOdiwz+d1VDwdWvt1QRUyVd1exG55WRKFGT0KO2lBxrdezJyKHv85mcCGlkG/6+GNmIlXBQghRk0kSWAghhKiCvtp6ia+2Xiqz3MPOgvd7B9O2luGCOjNfzb9haThYmtEq4NqFc7dgd3ZeSGHNibgySWCA1x8L4lhUBucScuje0J0u9e4uqRqVlsd3O8MY2saPZn4Ot71/QlYhp2KzsFSZMKlbEE80dMfUpOIkbXZhMTrdzWfFsTE3vekx7sbVOeTKu1yWkbVCiAfN1/uS+HpfUpnlHrZmzOjmRRs/G8DQfuFAVA72Fia09Lk2WqNrkB27LuewNiSzTBIYYNJD7hyPzeN8ciFP1LWnc62yrXhuR1RGET8cSGJwU2eaeVndeocbJOYUczqhAEszJa8+5M5jQYaq5orkFGnR3uqcpTK56THuhkanZ3hzZ8xMFAxo7FSyvGuQPU77Ell1JoONFzLpXc5rL4QQouaQJLAQQghRBQ1r60ubQMOFmgIwM1HiamuOl4Nlqe02hySh0elp4e9ASnZRyfJAFytUpkouJOVyPiGbBp6lL5gdrFS0r+XMxrOJd90HWKPV8cG6c7jamDOwlU9JBXNOoaEqrFirIzNfjYWZCRZm5c+U3tDLjjeeqMvq43F8vCGUH3dfplcTT3o39cTzhucM8PwvR++6J/DdsLxSAVxete/VZTbm8jNLCPFgGNrMida+hkSv4ZylwMXaFC+70iNMtlzMRqODFt5WpOReGy0S4GiOykTBxdRCzicX0MCt9Pe+g6Up7fxt2HQhi571767XukarZ+a2eFyszRjYxLGkgjmnyPDdXazVk1mgwcJUWTKy40bB7pZM7uTB6rMZ/G9nAj8eSqZXfQeebOCAp13ZUTWjVkTcdU/gu2GqVDCkmXO5655u7MSqMxkcicmTJLAQQtRwcnUihBBCVEEBzta0DnC65XYbrrSC2BGawo7QlHK3WXMivkwS+F5KySniQlIuAH2//7fM+m3nk9l2PpkXOgYw6uHAco9hbmZCv+be9GvuzcnoTP46EceSQ9EsOhhFm0An+jTzomMdF0yuDId9r3cDijQ3b7dQx93mLp9ZxbzsDUN+U3OLyqy72i/YtZx+wUIIURMFOJrTyufWCcyNFzIB2Bmew87wnHK3WRuSUSYJfC+l5BVzMdVwE7H/H2Fl1m8Py2Z7WDbPtXLhhdblj5IxN1XSp6EjfRo6cjI+nzUhGfx5Mo3FJ9Jo7WNN74aOdPC3KTlnvdvViyLtzSuB67iU30rifnO60lYjX1oYCSFEjSdJ4NvU9q2FtKjlztxxj/+n+96u/KJift1+hq2nIknPKcTb2YZBHevTr13dSu//+86zbD8dRXJmPu4O1vRqVYsRnRuWGVp7OTGTuZtPcDw8CT1Qz8uJEZ0b0rGBd5njZuUVMX/rKXaFxJCdV4SXsw192gQxsEO9+zZkV1QPD315iGY+tnw3KPg/3fd25au1/H4ojh0X0knLU+PtYMHTzT3o08StUvvnFmn49WAce8IySMlRY60yobmvHWM6+uDvVPqC563VF/g3IrPc43w/OJim3tf6yx2IyOSPQ3FcTM5HqYBgDxtGdfChiffd96ATVVdoYg5hKXn4OFry8iO1y6zPLCjm040X2HY+iYmP1sHG4v6c9p1sVMwZ0rTM8vQ8NR/+fZ42gY4Ma+tXpoq5Is38HGjm50BabhFrT8az7lQCU/86i6+TJUtfbAdAEx+He/kUbpu/sxVWKhNCE8omMc7HZwOG6mYhhBAGF1IKCE8rwtvejPHt3cuszyrQ8NnuRHaEZTOhgzs25uWPHLlbTlamfPmUX5nl6fkaZm2Pp7WvNUObOeNlZ1ap4zXzsqKZlxVp+e78fS6Dv89l8s6mWHztVSweZjg3N/a8/ZYT99Kp+Hw+351AtyA7nmtVOrEdmWG4celdTgWzEEKImkWSwLfp/SEdcbK5s7u0d7Pv7dDp9Ez5YxdHwhLp2yaIet5O7A6J4ZO/DpGaXcCYx8teqF9Pq9Pxxm87OXE5mSdb1aaBrzNno1OZt/kkZ6JS+OL5R0u2PR+Txvgft6DW6ujfri5+LrbsPR/L67/u4LXerRjyUIOSbbPzixjzwyYSMnJ5un09fF3s2BUSzZy/jxKfnsPkPm3u22siqr53u9fGybpyP7bv5b63Q6fXM23dRY5FZ9O7iRt13azZG5bO59siSM1VM6qDz0331+j0TP7rAucScuke7EJDTxsSs4tYfSqZw1GZzBvSkFou1y4SwlLzqetmxeAWZYfq+zle+y7ZfSmd6X9fws1WZYhBr2fliSReWXGerwbUp7mvJKJqqn+uVAH3a+5V4URpW84mcSImk00hiTzd8ubv0TtlbmpSbtVyQmYBAM7W5pWqar6Rs405LzwUyDMd/Nl7MZUt58r2mzQWUxMlj9RzZePZRC4m5pRMYJdTWMz60wkEuFjRwFNuwgghxFX/nDdMCNe3oWOFE6VtuZTNyfh8Nl/MKtW79l4yN1WWW7WckG1oZeRsZVqpquYbOVuZ8lwrV0a0cGFfRA5bL2Xfdaz3ir+jisScYtady6R/IyfsLAwJdo1Oz4LDKSiAHnfZZkMIIUTVJ0ng29SjRS2j7Hs7tp6K5PClRCb2asGIzg0B6Ns2iDd+28lvO8/yZOvaeDpWPER226kojoUnMeaxpox+rAkA/dvVxcbCjGX7QjlyKYHWQYak1Cd/HSRfrWHOqEdpX89Q+ft0h3pMX7yX7/45Tod63vi5GhJQ8zafJColm8+e7ULnhr4A9GsXxCsLtrN8/wVGdmmEm71x75IL43ki2MUo+96O7RfSOBqdzfhOfgxrZfgM9G7sypS1F1l4OJ5ejVzxsKt4+Pf6s8mEJOTycic/hra6ltjtUteZsX+GMHdvDJ/3qwdAdqGG5Bw1j9R1uuXzW/BvLCpTJd8PDi55/C51nRjx22nm7Yvhx6EN7/apiypIrdGx9VwSKlMlPRtX3NN3UGsfTsRksvZk/H1LAt9vpkolj9R345H6lau4vx8OR6STkaemdaATTtaGaqnRDweyPyyNSctOMbi1D9bmpqw+EUdGvprpTzZFITPECSEEAGqtju1hWahMFPSo51DhdgObOHEyPp915zLvWxL4fjNVKuhS244utY13E/5ITC4ZBVpa+VjjZGWKg6UpY9u58e3+JMauiqB3sCMmSth2KZvQlEKea+lyX1twCCGEqBpk/H0NtPH4ZcxMlDzdvl7JMoVCwfBODdFodWw5GXnT/XMK1NTxdKRv2zqllrepY0gyhMalA5CUmUdoXDqt63iUJICvPtazjzaiWKtj/dFwAAqLNWw8HkGbII+SBPDVbV98vCmjujWhqFhzV89biPtt07lUzEwU9G96LRGlUCgY2tITjU7P1tC0m+5/JNJQAXNj64j67tYEOFlyMvZaxUh4Sj4AtZxvfWMkNrOQQGfLUgloL3sLApwtuZScd+snJqqlPZdSyCnU0LW+G3aWFVfCPxTkgpeDBZdT8jgVk/nfBVjD/PFvFB+uP09k6rXPlJudBfNGtqCprz2LD0Xz057LOFqqmDOkGS3v04R0QghRHe2NyCWnSMejdexKqlDL0zHABi87MyLSizidkP8fRlizLDyexqzt8URlXOtbP7CJE7Oe8MbJypRfjqaw4HAKJkoFM7p58UKb8kcTCSGEqFmkEviKM1EpzN96ipDoVADa1fNi6MMNGPXdJkZ3a1LSQuHGvr4vzdtCZl4hHwx9iO82HOdMZAooFDQPdGNCzxbU8nAoeYzK9ASev+UUC7advmmsvVrWYsbgjhWuD4lOpbaHAxaq0n/eYF/DjLDnYlJvevynO9Tj6Q71yiwPjTMkuDwcDMOjkjINF8JBnmUvdP1cDHe+z8ca9gmNTSe/qLhUsji/qBhLlSmN/V1p7C8/PGqqs/E5/HIgjnOJhkmj2gbYM6iFJ2P/DOH5dt4lLRRu7Os7Yfk5sgo0zOhRm7l7YzgTn4NCAc287Rj3sG+ptgmV6Qn887+x/How7qax9gh24Z3uZfuqXnUuIZdaLlZYmJW+eGngYfhMnL/yHCvyRrdAnm3njZWq7MVPVkFxyeQhAGFXk8AuhqqMgmIt5qZKlOVUFvo5WpKYXUSRRoe5qeHenlqjIyVXjYuN9HerbkY9HFjh5GnX69bAnW4NyvZUvJFSoWDFuPblrpv+ZAOmP9mg3HX3gqeDJfvffuS+Hf9e+25489te5+tkxf/6N75fIQkhRJX2QmvXCidPu17XOnZ0rXPrylilQsHS4XXKXTftUS+mPep12zFWlqedij0v3b9z4r32TR//217XqZYdnWpJmzAhhHhQSRIYOBaeyKSft2Nrac6wTsFYqkz552g4r/+ys1L7p+UU8tK8LXRq6MsrT7YkLDGTvw5c5GJCBmve7ndbE551aeyHj8vNewj6OFe8vlCtIbtATXOHsn2sLFSm2FqqSEivfGWgWqMlPj2XnWei+WX7GRr6OtOlkaGS19LcUHmWV1RcZr/MPMNd59RsQyIrMtlQAenuYM3P206z8t8LpOcWYmNhxlOt6zC+R3NUpvdn8gdhPMdjsnnjr1BsLUwZ0tITCzMlG0NSeGv1hUrtn55XzMQV53motiMTOvsTnpLPmtNJXErJY8Xo5pgqKz/UunOQEz4ON+/J7e1QcSuHwmItOUVa3MpJqlqYmWBjbkJidlE5e17jaGWGo1XZis2toamk5hXzcO1rN1SuJoE3n0/l7bUXScsrxsJUSecgw2tx/XFefcSfKWsuMHNjOC+090ahgF8OxJGRr2F897ITnwghhBBCCCGEEOLBIklg4PM1hzFVKvl1Yg/cryRP+7evy+jvNpGVf/OkDkBWflGp/rsAxRotaw+HcSw8kbZ1K3/HOsjTsdzK2srKLTRMaGCpKv9Pa2FmQoG68m0X1h0O4/M1hwFwtDbnzb5tMbuSrA10s8fOSsW+83HkFqqxsbiWHNt+OgqAomItYGgxAfDTlpMUqDWM6tYERxsLtp6K5M+954lNzWH289WnWkxUzpc7IjFRKpg/rCFutoYEa78m7oxdGkJW4a3fh1mFmlL9dwGKtTr+PpvCiZhsWvtXfgKLOq5W1HG9857TuUWG97KlWfk3dSxMlRQU6277uJFpBXy1IwpTpYLn21+rlA9PNSSBQxPzGPewL5amJhyOyuLvM4a+wj8Na4SdheFz3sjThsEtPPntUBy7LqWXHGPcQ770CJYqe3Fnioq15BZV7nyhVCpwtJKqcyGEEMZRpNGRq67c7zATBThYymWwEEKIB88Df/YLT8wgIimLAe3rliSAASzMTBnZpSEz/txXqeN0b1566G59H2fWHg4jLafwtuIpVGsovEVvXJWpCVbm5fd/1Ouv/J+KCiQVCm5nnpqGvi589mwXkjPzWLT7HKN/2MTHIzrRuaEvpiZKXni0MXPWH+PVBduZ0LMFbvZW7A+NY8G201hbmJVUQRdrDQm0lOwClk3ujeuVCeC6NvFn6sLd7DgTzcGL8bS7jYS5qNoup+YTmVZAv6ZuJQlgAHMzJcNaefLhxvBKHefx+s6l/l3Pw5q/z6aQlqe+rXgKi7UUam5+caAyUZbbquF6FX1+FIqKP3YVCUvJZ/JfoWQXanjtUX/qul37DnqysSud6jgyvLVXSZuILnWd8HOy4Lvd0fx5NIGxDxmq8t9ee5HDUVm09rejR7ArCgXsuJDOvH0xZOQXM7FLxcMFhajItvPJfLwhtFLbethZsGp8+S0nhBBCiPttR1g2/9uZUKltPWzNWD6i/JYTQgghRE32wCeBo1IMEzH5u5btjRToXvkqQyeb0rOpXm1toCvJylbOwl0hd9UT2NLc8CctVGvLXV+k1uBmV/mZXxv4OtMAQxKuU0Nfhn75N1+tO1IyudvQTsEUabT8sv0M4+ZtAcDVzpIPhnTkm3+OYWdlSP5Zmhni6tLQtyQBfNWA9nXZcSaawxcTJAlcg0SnG26A+DmWfb8FOlf+PehkXfqGh+rKjQXd7X20WHwk4a56Al+tAC6soNq3sFh3W/13D0VmMmN9GHlqLS938mNAM49S6/s2Kb/Xa/+m7szdE83hqCzGPuTLkagsDkdl0T7Qgc/7Xevl/Vh9Fz7ZcpllxxNpE2BP2wCHSscmBEDbWk7MGdK0Utte7UUthBBCGENrX2u+fKpyLbDMTW73tr0QQghRMzzwSWCt1pBJMrvLfrTK2+hNejM9W9aiaaDbTbdxuUkS18ZChZ2VqqQX7/Wu9gt2tS/bL7gy3B2saRHozt7zsWTlFWFvbUjwPvdoYwZ1rE9YQgbmZqbU8XRAq9MzbdEeGvkZhqK7XamydrYtG7uzrSEpnF9Ob2FRfWn1Vz9bd/fZKG8itDvRPdiFJt4377ftYlN+hT2AtbkpdhampOaVfZ/erF9wef45m8Jn2yIAmPp4LXo1qnzLBpWpElsLU/Kv3Oi5dKV3cM+GLmW2faqxG+vPpnA4KkuSwOK2udiY42JTcZ9sIYQQoqpwsTbDxbri33FCCCGEkCQwvlcmYYu6MnHZ9aKvVAn/l7ydbfG+ycRvlRHs48LJiCSKNdpSye2QmFQAGvmVTRZdb8ofuzkXk8qqKX3LTNaWV1SMUqHA7ErV1/bTUahMlTwc7EuTgGvJ60MX4yjW6mhR21DNGOxrqCa+nJRZ5vHi0nIA8HKyuc1nKqqyq5OwXa0Ivl50xu21SbkXvB0s8L7FxHC3Ut/dmtNxORRrdZhdN+HjuUTDZIvBnrd+D68/k8wnWyOwNFMy66mgcpOzsRmFTF13kUZeNkx5rFapdRn5xWQWaGjgYXgs1ZVqlvIqo/VXEvHa2y2bFjVGx0920tzXge+GN/9P971d+WoNv/8bxfbzyaTlqfF2sGRgKx/6NKvc6JB8tYaFB6LZGZpMUk4R7nbm9GjkwfC2fmUmZ83IVzN/TwQHwtPIKigm0MWake396VLv2s2Yf04n3LIVRo9GHkx/svrMIi+EENVVp7nnaeZlxTd9br+91d3se7vyi3X8cSyVHWHZpOdr8LZTMaCJI72Db3++lzy1lmeWXqZXAwdeaF22WCAqo4j5h1I4EZ9HkUaPj72KgU2c6NXA4R48EyGEEDXVA58ErufthJ+LHZtPRvLMI41KKlU1Wh1L9pwzcnR35onmARy8GM9fBy8y+CHDBaper2fJnnOYmSh5vFnATff3crRm19loVl+3P8DJiGRORSbTOsijpCfxyn8vEJaYwV9T+mFraaiCzCss5qctp3C1syx5LE9HG1rUcufAhXhColNpeCURrdXpWLznHCZKBY80rtwQLlE91HWzwtfRgq2hqYxo7VXS1kGj1bH0WOV6tlU1jzVw5nBUFmtOJTOwhaF9g16vZ+mxBMxMFHS7oX/xjc7G5/D59kisVErmDGhQYdLYw96c7EIN20LTGN7KCx/Ha8nreXtjgGuVv20DHDBRRLH6ZBJdgpxK+gcD/HUyybCNv8MdP2dRvc14sgGO1nc2Ydvd7Hs7dHo9U/86y7HIDHo386Kehy17Lqbw2aYLpOYUMerhwJvur9XpmbLyDCdjMunZ2JMGnraExGczf08EIXHZfDawScm2+WoNk/48SXR6AQNaeuPtaMnGM4m8s/os03rWp1cTwySUzfwcmFFBgvenPREkZRfSue7Nb6gKIYS4N6Z39cLR8s5Gbd7NvrdDp9czfVMsx2LzeCrYgbquFuyNyGH27kRS8zTlJnIrUqTRMW1jLCl55c8TE5+tZvzqSNRaPQMaO+FmY8rWi9l8uiuB9HwNI1vK+UkIIUT5HvgksEKh4M1+bZj083ae/fof+revi6XKjM0nIkqqVu/RaPT/TPfmtVh96BJfrz9GbFoOdTwc2Xk2mgMX4hn3RLNSE+DFpeVwOioFH2dbGvsbfpw817Uxe8/H8vX6Y1xOyqK+txOXk7JYc+gi9lbmvNm3Tcn+L3RtzKs/b2fcvC30aVMHvV7P2sNhxKRm89mzXbAwu/YWm9K/LS/+sJkJ87cxsEM9XO2t2HoyklORyYx5rAm+LmX7MovqS6FQ8PqjAbyx+gIvLDpD36buWKmUbDmfRkRawZVtjBzkbXqigQvrTifz3e4o4jILqe1qxe5L6RyMzGJMRx/cr5sALy6zkLPxuXg7mNPIy1Dd/82uKLQ6Pe0CHInJKCSmnIroJ4JdMFUaXrt3119i/LJz9G/mjo25CXvDMzgWnU33YBceqWtIOPs6WvBcO29+PhDH2D9DeLyBCwpgb3gGx2Oy6VbPmfa1HP6Ll0dUQU808rj1Rvdh39ux/XwyRyMzePmR2gxra7gZ2LupJ1NWnuGPA1H0auKJh33FVfzbzydzPDqTUQ8F8MJDhoRx3+beWJubsuJoLEcj02kV4ATAymNxhKXkMatvQx6pbxi98mQTT1784xjf7QjjkfquWKlM8XawxNuhbPui1SfiSMwuZGR7Px6uW/kLeiGEEHfu8bqVn6flXu57O3aEZXM0No+X2rsxtJnhN9pTDRyYujGWRcdT6VnfAQ/bW7eriMoo4oOtcYSlFVW4zbJT6eQU6fjgcW8eqW24furdwJEXVlzm92Op9GvkiI35/U98CyGEqH4e+CQwQJsgT74d0435W07x+46zmJooeaiBD4M61uODZf/edb/g/5pSqeCrFx7lx82n2HE6irWHwvB1seWdp9vRu01QqW1PRCQzc/m/9GpZqyQJbG9lzs8v9+CnrafYfTaav4+E4WRrSc+WtRndrUmpid1aB3kyZ1RXft52mh83n8TUREljP1emD2xPsG/pu9ABbvb89koPftxyinWHL5FfpCHAzZ73BnegZ8vyJ+MS1Vtrf3u+GlCfX/6NZdHheExNFHQIdGBAc3c+2nS5VEuF6kCpUDC7Xz3m/xvLzovprDuTjK+jBW8/FsiTjUv38j4Vl8PHmy/TI9iFRl625Ku1JW0jdlxMZ8fF9HIf44lgw+emc5ATXz/dgD8Ox7HkaDwarR5fR0tee9Sffk1LTxr3fHsfApwtWX48kZ/2xaDT6/FzsuS1R/zp16z8CeaEqCo2nU3EzERB/xbeJcsUCgVD2/qyPzyNreeSGNm+4mG8uUUa6rha0/uG1hGtAxxZcTSWC4k5JUngTWcTcbM1L0kAA5iZKBnYyoeP/gnl37A0ugWX/5lJySni+x3h+DtbMfqhm1cnCyGEeLBsvpCFmVJBv4bXWj8oFAqGNHPi36hctl3KYkSLm1fo/nU2ne/2J2FhqmRQUyeWnyr/t2JslhqAdn7XRpSZmiho42fD8lPpRGWoaehR+UmYhRBCPDge+CSwXq8nPbeQlrU9aPlS6aqnrScjgdKTmR36bGSpbeaOe7zc4z7ZqjZPtiqd2Lxx3/vJxkLF5D6tmdyn9U23Ky9OAHtrQ8Xv9VW/FWkT5EmbIM9KxeXlZMsHQx6q1LaietPr9aTnF9PC144Wg4NLrdt+IQ0A5+sm8Nj3ettS23w3qPQ+V/Vs6ErPhqUr8G7c936yNjdl0iMBTHok4Kbb3RinlcrktuNs7mtHc9/KVcg/Ute5pDpY1Hxn47L4eV8E5+INPdXb1nJicGsfXvzjOC90DChpoXBjX98Ji0+QWVDMjKcaMHdXOGfislEAzXwdeKlLbWq5XhspUpmewD/vjeCX/ZE3jfVWvXND4rOp5WqDhVnpG64NPA3v/XMJN+/P37+Fd6kE8lUXEg2vjfuVKuLcQg3Rafl0rle2gvf6x6ooCTxvVzgFxVomP1a3TJ9hIYQQt+9sYj6/Hk3lfJJhhFgbPxsGNXFi3F+RPNfKpaSFwo19fV9ZG0VWgZbp3byYdzCZs4kFKICmXlaMa+dGoNO1kVmV6Qn8y5EUfjuaetNYu9ezZ9qjFfepP5dcQC1ncyzMSp8f6rsariPPJ996Poyw1CIeC7JnTFtXYjLVFSaB/RxUHInJIzqziHqu165T464kh12sH/hLfCGEEBWQMwTQ75PVNPJz4YexpRO6m05cBqDxLSZSE0KUb9DPp2joacM3A0sngDafvzJJYSUmUhNClHY8KoPJy09ja2HKkDa+WJop2XAmkTdXnKnU/ul5aiYuOclDQS5MfLQOYcm5rDkRx6XkXFa+1A5TZeUTnJ3rueLjePNqI++brC8s1pJTqCnVSuUqCzMTbM1NScyq/ESSao2OhKwCdl1I4bd/owj2tKXzlbYNKblF6AG3ch7L1cawLKGCx4pIzWNzSBLtajnRMuD2J/gRQghR2om4PN78JwYbcyWDmzphYaZkY2gWUzbEVGr/9AINr66NomOALS93cCM8rYi1IRmEpRaybEQdTJWV7znWuZYtPvY374HvZVdxK4fCYh05RTrcbMpeWluYKbFRKUnMUd8yjtce9sDsyoS/MZkVbz+8uTOHo/P4344EXnvYA1cbU7ZfymZ/ZC4969vjXom2E0IIIR5MD3wSWKFQ8GSr2qw6cJE3fttJh3reaHU69pyL5fClBJ7uUA9/t/+ml5QQNYlCoaBnQxdWn0rm7bUXaBfggFanZ9/lDI5EZdO/qTt+TjJUTYjb9cXWi5iYKFjwbEvc7AxVrv2ae/PiwuNkFRTfcv+sguJS/XcBirU6/j6VwPGoTNoEOlU6ljpuNtRxu/ObOblFhklvLFTlt10yN1NSUKyt9PHWn07giy0XAXCwMuP1x+uWtJ3Ju/JYluU81tXKrcIKHmv5kRj0wDM3aUshhBCi8r7am4iJUsFPAwJxszEkLfs2dOSlvyLJKrz1935WobZU/12AYq2e9eczORGXR2vfyp+bajtbUNu54t7zt5KrNsRrYVr+TVQLMyWFxfpbHudqAvhWXKzNGNXGlU93JTBxbVTJ8o4BNrzRqXKjM4UQQjyYHvgkMMDrvVsT4GbP+iPhfLvhOACBbnZMG9COPm2DbrG3EKIirz4SgL+TJRtCUvhhbzQA/k6WTHkskKdu6KErhLi1yym5RKbm07+Fd0kCGMDczIThbf344O9zlTrO4w1Ltzyo72HL36cSSM+7daXS9QqLtRUmTq9SmSqxUpX/c0N/5Zq4ostehUKBosK1ZTXwtOWT/o1IyiliyaFoxi08zqy+DXm4rmvJY5X/QFf/p+xj5RVp2BySREMvO5r6OlQ6FiGEEOW7nFZIZIaavg0dSxLAAOamSoY2c2bm9vhKHeexoNIts+q5WrD+PKTnV/7mIRgqeQs1uptuozJVYmV285EyFU14rLjJujux6HgqPx1KwdvejCFN3XC0NOFUQgGrz6bz2t/RfNrTB6sKbq4KIYR4sEkSGDA1UTKoY30Gdaxv7FCEqFFMlQqebu7B0809br2xEOKWotLyAfBzsiqzLsCl7LKKOFmXHvZ6tVpWq7t1pdL1Fh+MvquewFcvUguLy7/4LizW4mpz8yG612vgaQdXiqA6Bbkw4ufDzNkWxsN1XUsqgIvKeayry6zNy/4sOhCeRpFGR/dG8j0mhBD3QvSVVgd+DmW/3wOcyrbsqYiTVenvbNWVSlrdTe/6lbXkZNpd9QS2LBlNUv7jFmp096xPb55ay+/HUnGxNuWnAYHYmhvObZ1q2VHP1YJZ2+NZeDyNse2k2EIIIURZkgQWQgghqgntlQvbyg4ZrYjyHpUkdW/sQRPfm7dMcrGp+ILe2twUOwtTUnOLyqy72i/Y1fbOhui62VnQzNeB/WFpZBUU43llgrjyHis5x7CsvH7Bey+lYqJU8Ej9shPKCSGEuH3aK7nSKnMuq2dPE8+b30h1sar4stlaZYKduQlp+Zoy6672C3a1vjd9emMy1RRp9PSsb1uSAL6qW5AdX+xO4EhMHmPb3ZOHE0IIUcNIElgIIYSoJnwdDRepVyuCrxeTXvBfh4O3gyXeDnfX27uBpx2nYjMp1upKKpIBzsVnA9DQy/am+0/76yznE7JZNrYdqhv6MeartSgVhkSDlcoUPycrzifklDnG1ccK9rIrs+5ETCZ13W1wtKp8RbIQQoiK+dgbEqLR5Ux+drMJ0e4XLzsVXnZ39x1f382C0wn5FGv1pZLb55MN5+Zg93szD8bVY+sqGLmj59oNYyGEEOJGlZ8CXNxT64+G0/athaw/Gm7sUO6aRqvjmTn/8OGy/eWuT8nK54Nl++n+4Qo6Tl3M05+tYeGukAp/vFy180w0bd9aSHx6brnrN5+IYNR3G+n8zhI6vbOE577ZwMbjl+/6+YiaZ0NICg99eYgNISnGDuWOqDU6FuyPZdDPJ+ky5zB9fjzO7G0RZBeWrTjJV2uZuzeagQtO8ujXhxn5+2nWnk4u97in43J4deV5Hvv2CD1/OMabqy9wNr5sgkxUHXXdbfB1smTruaRS/Xs1Wh1LD0cbMbI791iwO4XFOtacuNYDUq/Xs/RwDGYmCroFu99kb/C0tyA5p4i1J0v3kDwVk8npmCxaBTiW9CR+LNiNuMwC9l66Nuy3WKtj5bFYHCzN6FDbudQxUnOLSMtVU9/j5oloIYQQlVfXxQJfexXbLmWRfl31rEarZ9mpNCNGdue6BdlRqNGzNiSjZJler2fZqXTMlAq6BpW9yXgnAp3M8bA1Y9flHFLzSk8Gu+F8JoUaPW18re/JYwkhhKh5pBJY3BWtTseHy/ZzIT6dOp4OZdbnFKgZ88MmkrPy6deuLrXc7dl3Po7vNhwnJjWbaU+3L/e4IdGpzFzxb4WPu+LfC8xec5h6Xk68+HgzlArYeCKC95fuJy4tl9GPNblXT1EIo3vvnzD2hmfQoZYDw1p5Epaaz9+nkzkTn8tPwxpifqX6UafXM23dRY5FZ9O7iRt13azZG5bO59siSM1VM6qDT8kx91/OYNq6S5ibGvo2O1qZsTU0lQnLz/N+zzp0qetkrKcrbkKhUDD58bpMXn6a5389Qr/m3liqTNgSkkREat6VbYwc5G16opE7607F8+32MGIz8qntZsPuCykcvJzOi50Ccb9uAry4zALOxmbh7WhJI29DG4pnO/izPyyVb3eEEZGaRz0PWyJS8lh7Mh57KzMmP163ZP8hbXzZHJLEe2tDGNzaF3c7czaeTeRiUi4znmxQ0jf4qqsV1x72dz5rvBBCiNIUCgWvdfLgzX+iGb0ygj4NHbE0VbLtUhYRGYb2PNXsVMbjde35+1wm3/+bRFy2mtpO5uyOyOFQdB6j27jift0EePHZas4mFuBlZ0Yjj8r38wdDC4w3O3vw9oZYxq6K5KlgB5ysTAlJLGDzxSz8HVWMaOFyr5+eEEKIGkKSwOKOpWTl8/6y/RwNS6xwm7WHL5GQkceEni0Y2aUhAAPa1+OVBdtYeziMYZ2CCXAr3U/y7yNhzF5zuMIZ53MK1Hy7/hj1vJz4ZWIPTK8MHx7YsT5jftjErzvO0KdNHVztb+9HlRBVUWhiLnvDM2gf6MBnfeuVLHezUfHT/lg2nkuhbxNDpeT2C2kcjc5mfCc/hrUyzI7Vu7ErU9ZeZOHheHo1csXDzpxirY7Pt0agVMAPgxtSx9XwWenf1I2JK87z+bYIWvjZYWchp4iqqHWAE3MGN+XnfREsPBiFqVJJhzrOPN3Sh1n/nC/VUqE6UCoUzB7YhPl7IthxIZl1pxLwdbTk7R71eKpp6Ul4TkZn8vGGUHo08ihJAttZmvHjMy1ZsDeCPRdTWH86ASdrFT0ae/B8xwBcr+vza6Uy5YfhzZm76zJrT8aj1ugIcLHik/6NeLhu2Z6/mfmGKiubciaME0IIceda+Vjz5ZN+/HIklcXHUzFVKmjvb0P/xo58vCPhrvsF/9eUCgWf9fJlweEUdoXn8Pe5THzsVbzVxZMnGziU2vZUfD7/25lA93r2t50EBmjta8MP/f3541gqK09nkF+sxdXajIFNnHi2pQs2N/QKFkIIIa6SqxpxR3adjeb9pfvR6fQ892gjfttxttztYlINQ8s71vcutfyhBj4cupjAxfj0Ukng0d9v4kxUCk38XbEyN+PgxdLDewFORiRRpNHyVOvaJQlgAFMTJY83DeBcTBqno1Lo2sT/XjxVIYwqJrMQgHaBpW+WdKzlyE/7Y7mUfK037KZzqZiZKOjf9NqM0AqFgqEtPfn3ciZbQ9MY2caLcwm5pOYV82Qj15IEMBg+Q0NbeTJt3SV2XkynTxOZWbqq0ev1pOepaeHvSAt/x1Lrtp1PAsDJ+lpfw/1vP1Jqm++GNy/3uL2aeNKriWepZTfuez9Zm5sy6bEgJj0WdNPtyosTwN7SUPF7fdVvRZxtzJn+ZINKxdW1gRtdG8jnQAgh7iW9Xk96gZbm3tZ86126dcH2MEOPdufrJmLb81Lp7+xv+pT/G79HfQd61HcotezGfe8na5UJrz7kwasPedx0u/LivFFzb+ubxl7P1ZKPuvveSZhCCCEeYNU6CZxfVMx3G45z8EI8yVn52FiY0ayWO6O6NSHI89rFcbFGy9J9oWw/HUVUchZqrQ5nWwva1fVi7BPNcLY1NOo/Fp7I+B+38tHwhwlLzOSfo+Fk5RVRy8OBV3q1JNjXmR83n2TzyUgKiooJ8nLi1Sdb0tDPMOQmPj2Xfp+sZnyP5qhMlCzbH0p6TiE+LrY83b4e/dvf+uL0dGQyv+44y5moFIqKNfi52tO3bRBPt6+L4roxviHRqczbfJJLCRnkFarxcLDhkcZ+vNC1MRaqiv+sV5/jzXg6WrNmav+bbhOWkEnrIA8m9myJqYmywiRwgKuh/1VkSha1PBxKll9NDt9YrRufnsubfdvQv11dZlXQDqJ1kCd/Tn4KZ5uyEyxk5BmGkJkqq1f1QHVwtdfsocgsUnLVWKtMaOZjy3PtfEolEou1OlYcT2THxXSi0gso1upxsjajbYA9Yzr44nRlduTjMdm8suI8H/Sqw+XUfDaGpJJZUEwtFyte7uRHAw9r5v8by7bQNPLVWoLcrJnQyY9gTxsAErKKGPjzScY+5IvKRMGKE4mk5xXj42BB/2bu9G168z6iAGfic/jjUDxn4nNQa3T4OlrSu4kr/Zu6l/q8nUvIZf6/MYSl5JNXpMXdzpwuQU4829YLC7OKqy2uPseb8bBTsXJ0+Yk5AH8nw/s8Or2w1PLYK8lhV5trCb9zCbnUcrEqE1MDD8MF1vlEQ3/tpBxDL9nr/25X+Toahr2HJuXSB0l+VUUD5x2koZcd3w4r/b7ZctaQBL5aISuEEEJUVUMWhxHsZsnXNyR0t17MAqDhPZpITQghhBDXVOsk8LRFezgensTAjvXwd7UjKTOfZftDOXQxgeVv9C5JME5dtId952Pp1bI2fdrUQa3RcvBiAmsPh5GUmc/Xo7uWOu43/xzD2tyMkV0akp2v5o9dZ3nz950EeTqi1et5/tFGZOYVsXB3CJN/28nKt/pgY3EtEbP64EUycgsZ1LE+znaWbDoewaerD5GQkcvLPVtU+Hy2n47i3SV78XWx45kuDbFQmfJvaByz1xzmfGwaMwZ1ACA6JZuJC7bhamfFM10aYmVuxtGwRH7feZbo1Gw+Gdm5wscIcLPn/SEdb/q6Wt0kiXzVs480xMzUkGiqaOI2gD5tg9h2Ooo5fx/FwsyUQHd7Dl6IZ/XBi7Su40GzgNJJprVT+5UctyIWZqbUcncoszy3QM26w5cwM1HSJECSV/fau+svcSImm6ebe+DnaEFSjpoVJxI5HBXCkuea4nIlGfnu+kvsD8+kR0NXnmrshlqj43BUFn+fSSEpW82XA+qXOu73e6KxVpkwrLUnOYUaFh1JYOq6i9RxsUKr1/NMGy8yCzQsOZrAlLUXWPp8U6yvG5q97nQyGfnFDGjujrO1ii3nU5m9PZKE7CJeetivwuez82Ia728Ix9fBnBGtvbAwU3IwIpOvdkRxISmPaU/UBiAmo5DXVoXiamPG8NZeWJmZcCwmm4WH44nJKGDWUxXf3AlwsuTd7rVv+rpaqm4+dL+umzX9m7qz7kwygc6WtAt0IDKtgK93ReFibcZTjQ1D2AuLteQUaXGzKTu7tYWZCTbmJiRmG26SWF3pe5qnLttyJavAMEFLWm5xmXXC+BQKBb2aePLX8TimrDxDu9pOaHV69l1K5UhkBv1beOPvLK1whBBCVF0KhYIe9RxYE5LB1I0xtPOzQaPTsz8yl6OxefRr5Iifo/mtDySEEEKI21Jtk8AZuYUcuBDPgPZ1mdirZcnyul6O/LDpJBfi0nG1t+JifDp7z8Uy+KH6vN67dcl2gx9qwAvfbuDgxXhyCtTYWl5LnGi0eha83ANrC0PFYl6hmiV7z5Ov1vDbxJ4or1SZqjVa/tgVwvmYNFoHXRuempiZx/zx3Wnsb0jODGhXl9Hfb2LxnnP0blMHX5eys8MWqIv55K+D1PN24qeXnihJhA7qWJ8v1x1h2b5QHm8WQLu6XuwOiSGvsJjvxnQg2NdQhdy3bRAmSgWxaTmoNVpUFSRSnW0t6dGi1h295te7VaL2KitzM17q3pwZf+7jtV92lCxv6OfCJ890LlVteTvHvZFGq+P9pfvJyCtieKdgHG1kEp97KSO/mEORWfRr6sb4TtcSq0FuVvy0L4YLyXm42Ki4lJLHvvBMBjb34NVHrlV2DGzhwZglZzkclUVOoQbb63rNanV6fhzasCQxmavWsuxYIvnFWhYMb4TyyntErdWx+EgC55PyaOV3rdIxMbuIuUOCaeRlC0C/pm6MW3qOpUcTeKqRGz6OZd8LBcVaPt8WSV1XK34YElzSQ/Xp5h58vTOKFScS6VbPmTYBDuwNSydPrWVO9/o08DBUIfdu4oaJAuKyilBrdKhMy0/kOlmb8UTw3U/OMbilB5dS8pi9PbJkmYOlKd8MbIDzlaH/uUWGhK6lWfmxWJgqKSjWARDsYYOJUsGOC+mMaO2FyXWV8zsvpgNQpNHdddzi/ni1Wx38nKzYcCaBH3aGAxDgbMWUHvXofUMPXSGEEKIqeuUhd/wdVWwIzWTugWQA/B1VvNXZgyeDHW+xtxBCCCHuRLVNAltbmGFtYcb201EEeTrSqaEvzraWdG7kR+dG15JUdb2c2PHhkFJJDoD03AKsr1Tv5hUWl0oCd6jvVZIABkp61j7a2K8kAQzg62JIOqVkX+vJCYb+t1cTwGBIbI7s0pB3Fu9l77lYhnUKLvN8Dl9MIDtfzaNd/MkrKoaia1V4jzUNYNm+UHadiaZdXS/crlQ4f7fhOM892phmgW6oTE34YOhDt3zdNFoduYXqm26jVCiws7o3d983Hb/M+8v242RjyaQnW+LlbMuFuHSW7DnHmO838e2YbrjY3V3Vmlqj5d0le9l7PpYmAa681L3ZPYldXGOtMsFaZcLOi+nUcbXmodoOOFur6FTHiU51nEq2C3K1ZsuEVtzYjSMjv7hkkoo8tbZUErhdgENJAhgM1bMAXYKcShLAAL4OhmRuSm7p92/7Wg4lCWAAMxMlw1p58t4/Yey7nMGQlmX7hx6JyiK7UMPw1p5XqmGvVcR2refEihOJ7A7LoE2AA662hu+GuXtjGNnGi6betqhMlczoWeeWr5tGqyO3nGrb6ykViptOwBaRls/4peco0ugY1sqTRl42pOSoWXoskfHLzvFpn7o09bl2Y0lRQScUheLaTNtO1mb0a+rGyhNJTF13kefbeWNtbsLW82lsPm+YnOXG70xRdZgqlQxs5cPAVj7GDkUIIYS4I6ZKBQMaOzGgsdOtNxZCCCHEPVFtk8AqUxPeebo9H608wCd/HeKTvw5R28OB9vW8eLJVHQLd7a/bVsnWU5EcuphATFoO8em5ZOQWliRLdHp9qWNf7RF8lcmVKsEblyuVyiv7l46ttkfZu9f+VxLJsWk55T6f6Cs9cr/bcJzvNhwvd5uEjDwAujbx5+DFeDYcu8yx8CTMzUxoFuhGp2BferWqhaXKrNz9AU5FJt+TnsCV9d3GE1iYmTJ//BN4OxsSdZ0b+tK6jgcv/biFOX8fY9bwh+/4+Jl5hbz5+y5OR6bQJMCVr1549I6riUXFVKZK3n48kE+2RPD5tgg+3wa1rrQm6NnQlQDna58NMxMF2y6kcSQyi9jMQhKyi8jI15QkIG/4uJX0CL7qavLR2bp0W4OrN2Bu3L+2S9mbCFf76MZlFpZZB4YWD2BI7M7dG1PuNglZhtYJj9R15lBkFpvOpXI8JhtzUyVNvW15qLYjPRq6YHmTnsCn43PvuifwwkPx5BRp+aBXHbrWcy5Z3rW+M8/8foYPN4az7IWmJRXAhcXlV/AWFutKWnYATOjsjwIFf51K4t/LmYChH/Dn/eoxcfn5myamhRBCCCGEEEIIUb1U66v8rk38aV/Pi39D4zh4MZ5j4Uks2n2OP/eeZ+awh+naxJ/cQjUTftpGaFwazQPdaejrwlOtahPs68KSvefYdDyizHFNleUPp76xdUFFTE3K7q/V6ipcB9cS0WOfaEojP9dyt7larWxqouS9wR0Z1a0Je0JiOBKWyMmIZA5dTGDJnnP8MrEHDtblt0MI8nTk2zHdbhq/+U2SWrcjM6+QlKx8Otb3LkkAX9W8ljv+rnYcuhR/x8ePS8vh1Z+3E5Oaw0MNvPloeKebToon7s4jdZ1pG+DAgYhMDkdmcTw2myVHE1h2LIH3etXh0brO5BVpeHVlKBeS8mjqY0uwpw29GrnRwMOaZccS2Hw+rcxxK5rEr5Ift3L31165M1PRZ/nqjZvRHXxoeGWiuRtdrVY2VSqY3r02z7fzZm94BseiszkVl83hqCyWHkvgp2ENcbAs/8ZLHVcrvrqhB/KNzCtoJXHVpZR8LM2UPFq3dKWMg6UZneo4suZ0MlHphdR2tcLOwpTUvLK9fMvrF2yqVPDqI/48386biPQCbM1NCHS2JCG7CI1Oj4+DtFQRQgghhBBCCCFqimqbMcsvKiYsIQNPRxu6NQ2gW9MAAE5cTuLln7byx86zdG3iz/L9oZyPTWNK/7b0b1d6Aqf0nIL7Elt0SnaZZVFXlvm7lu0HDODlaEhEmZua0iao9PD1rPwijoYl4n6lDURiRh4xqdm0DvJkWKdghnUKplij5ev1x1jx7wW2noxkYMfyE092VuZljn+/mJooUSjKVlpfpdNdS9bdroSMXMbN20JyVj7929Xljb6tMakg4SfuXr5aS3hqPp525nSt51xSkXoyNptXV4ay+EgCj9Z1ZsWJJEKT8nijWwB9m7iXOkZaOcnJeyE6o2y1b1S64bPt51R+ItPL3tDuxNxUSWt/+1Lrsgs0HI3Jwv1KG4jE7CJiMwtp5WfPkJaeDGnpSbFWx3e7o1l1MontoWkMaO5R7uPYWZiWOf7tMjMxJLl1ejC5Id999eOjvfIZq+9uzem4HIq1upI+xwDnEg2jCIKvJLw1Wh3bLqThZmtOC187mnpfu0lzKNIwK3dz39I3bsSD7Z/TCXy8IZRpPevTq8l/cw65lzRaHSuOxrL+TAIJmYU4Wql4tIErz3cMKHcy1PWnEvjreCwRafnYWZjS0t+RsZ1r4W5X+jvldGwmP++NJCQ+GzMTBQ297XiuQwCNvMt+7v85ncDyo7FEp+Vjb2VGu1pOPN8xoMwxhRBC3J2NoZn8b2cCUx/xpEd9B2OHc9uKNDr+OJbK1kvZpOdrcLcx47G6dgxt5lymeCCzQMMvR1I5GJ1Ler4GXwcVfRs60jvYoUwBU2GxjoXHU9kelk1qngZXG1O61rFjRHMXLK6MKPvlSAq/HU29aXzPtXLhhdblFy0JIYSo2qpt1iw8MZMxP2zm1x1nSi2v5+2EytSkpIVDZp5hSHcdD4dS252JSuH4ZcMkBFrdvZ0AaXdINDGp1xLBao2WRbtDMDc1oct1/Yqv17auJ1YqU5buO092flGpdT9tOcW0RXs4EWGI97cdZ5gwfxsh0ddO0GamJtT3MSTmqkoy1MZCRdMAN46GJRKemFFq3aGL8cSkZdOu7u1PYlSs0TLl990kZ+XzTJeGTOnftso855rqcmo+Ly09x2+H4kotr+tmjcpEUZKczCrQAGVbNJyNz+FkrKHlyZ0m/iuyNyyd2OsSwWqNjj+PJqAyUdCpTvkTi7T2t8fSTMmK44lkF2pKrVvwbywz1oeVxLvwcDyTVoZyLiG3ZBszEyX13K0B7nvv3A6BjhQU6/jnbEqp5Wl5avaEpeNsbUatK6/3Yw2cKdToWHMquWQ7vV7P0mMJmJko6Fbf8B1haqLk53/jmL0tgmLtte+/1Fw1S44kUNfNiha+5d+wEqI6+t/GC3y3M5xAZ2smdq3Dw0EuLD8Sy8QlJ1HfMAni9zvD+N/GUGwtzXilax0eC3ZnR2gyLy8+QXbBtZtZ+8NSmbDkJOcSsnm6lTfPdwwgK7+YlxefYGdocqljzt0VzscbQikq1jL64QD6NffiQHgaY34/RnR66XkFhBBCPLg0Wj2T10ez8HgaXnZmvNTejfYBNiw6nsZrf0eXmrg3v1jHK2uj2BCaSadAWyZ2dMfLzowv9iTy9b6kco+76HgaLX2smdjRnXqulvxxLI0Pt8Whv1JQ0LmWLdO7epX5b+ojntiolKhMFLT3L38UnRBCiKqv2lYCN/Z3pW1dT1YduEhugZpmtdxRa7RsOHaZwmINw69MvtYp2Ifl+0N5b+l+BrSvi42FinMxqWw8fhkTpQKNFnIL722FogIFo7/fxMAO9bC2ULHhWDgX4zN4o0/rMn2Fr7KzMuf1Pq35aOUBhn21nr5t6uBka8nhSwnsPBNNi1ru9GxZC4AhDzdgy6lIXv91B/3a1cXL0ZrYtFxWHriAm70V3Zr639Pnczfe7NuGsXM3M3buFvq3q4uXkw1hCRmsPXwJJxtLJvZqcdvH/PtIOBfi03GxtSTQ3YGNxy+X2aaJv2uZFhTizjXysqWNvz1rTiWTV6SlqY8tao2eTedSKCzWMaSVoTLwodoOrDyRyIcbw+nX1B0blQnnk3LZfC7V8HnT6W85UdqdGLc0hP7N3LExN2FjSCqXUvJ57RH/Mn2Fr7KzMGXSIwF8suUyz/5xmqcau+FkbcbRqCx2XcqgmY8t3YNdABjUwoNtoWm8teYCfZq44WlnTlxWEX+dTMLNRsWj1/XpvR+GtfZk3+UMZm+P4GxCLo29bEjJVbP2dDI5RVr+1zuopCXGEw1cWHc6me92RxGXaWgRsftSOgcjsxjT0Qd322sTPj7f3puPN1/mtZWhdKvvTJ5ay+pTSeQWafmod1Cl298IUdWdic1i09lEejX2YFqvBiXLPR0s+GZ7GJtDEnmqqeGG5Ln4bP48FEPnuq7M6tewZHLKIHcbPvz7PKtPxPFshwCKtTo+23QBpQLmjmhBHTfDBXH/Ft5MWHKCzzdfpGWAI3YWZoQn57L4YDR+TlYseLYl1uaGn149G3syYsFhPt90gW+HVdwXXAghxINj7bkMTicU0K2OHe928yr5PdbS25opG2L482Qaz7UyVOGuOp1OZIaaaY960r2eAwB9GjoybWMMq89m8HQTJ3zsDb+FV5xO50xiARM7ujOwiVPJtlZmStafz+RsUgGNPayo7WxBbeeyI1TmHUwmV63jjc4eNHAr/3pWCCFE1Vdtk8AA/xvRmUW7Q9h+OordITGYmCip7+3EF889SscG3gC0quPJrGEP88fOEBZsPY2ZqRJPBxvGPtGMADd7Jv+6k0MX42ngc+8SOd2a+lPLw4Gle8+TU6AmyMuJz57tQueGvjfd76nWdfB0tGbhrnMs3ReKuliLp5M1Yx5ryvDODVBdmfAswM2eeeMe55ftZ/jnaDgZuYU4WJvTtbE/ox9rgp2V+U0f579Ux9OR31/txfytp1h3JIzs/CKcbC3p0aIWYx5riqt92Um9buVwWAIAqTkFfLBsf7nbvDuogySB77FZTwWx5EgCOy6msTcsAxOlgnru1nzWtx7tazkA0NLPnvd71WHxkXh+PRCLmYkSDzsVozv6EuBkwVtrLnIkKpP6V6po74Wu9ZwJdLFi+bEEcou01HG14n+9g3i4zs1nm+7VyBUPOxWLjySw/Hgiao0OT3tzRrX3ZkgrT1RXhtv5O1ny3aAG/H4ojo0hqWQUFGNvYcojdZ14ob33fZ9AzUplwg+Dg/n9UBw7L6az5XwqlmZKmnjb8mxb75IWDwBKhYLZ/eox/99Ydl5MZ92ZZHwdLXj7sUCebOxW6rg9G7pibqrkz6MJfL8nGiuVCc197HihvTd+TvLjXtQcqblF1PewpV8L71LLWwcYRgpcSMzhqaaGZetOGfrUv9qtTkkCGKBrfTciUvPwdzZ8d4XEZ5Oaq+bJJp4lCWAwVNkPa+vH1L/OsjM0hT7NvNhzKRU98FwH/5IEMICrrTk9Gnuw4mgs0en5+Dnd/vlQCCFEzbI3wjASbVx7t1I35Nv721DH2Zx15zJLksBqrZ66LhZ0q1O6BVFLH2v2ReZyMaWwJAm87nwGvvYqBjQuPUpuSFMnnKxMUN3Yc+w6F1IKWHoyjZbeVvQOLn+UnRBCiOpBoddX0LD1Xj2AQkHGmpn38yGqjPj0XPp9sppeLWsxY3BHY4cj7hPHvu9ynz82d0yhUJA8b6Sxw/hPJGQVMfDnk/QIduGd7rWNHY64DW7jFlbpz1DKL2Pvy7Hz1Rrm7rrMwctppOQUYW1uSjNfB57vGFAqkVis1bH8aCw7zicTlZ5PsUaHs42KtoFOjOlUC6crFe7HozKY+OdJPuzTkPCUXDaeSSSzoJhaLtZMeLQ2DTztmL8ngq3nksgv1hLkZsPER+sQ7GVo9ZGQWcDT8w4yrnMtzEyUrDgaS3q+Gh9HS/q38KZf82tJ04p6Ap+JzeL3A5Gcjc2mSKPDz8mS3s286N/Cu9TF67n4bH7ac5mw5FzyirS425vTpZ4rz3UIwOImk5FefY4342Fnwarx7W/rbwGw4UwCH/0TyktdajGinWEEzaB5B7EwU/LHqDaAob2MQkGpHtsAW0KS+ODvc0zqFsTAVj6l1kWk5jFiwWF6N/VkSo/6fLoxlHWnEvjjhdbUdis9hPav43F8seUi7z0VzOMNS/dRv1dcX/ixyn7ehBB3R6FQkPhVn/ty7PxiHfMOJHM4JpeUXA3WKiVNvax4rpVLqUrRYq2elafT2RmeTVSmmmKtDicrU9r62jCqjStOVoabXyfi8nh1XTTvP+bN5fRCNoZmkVWopZaTOeM7uFHf1ZIFR1LYdimLgmI9Qc7mjO/gTrC74cZ0QraawYvDebGtKyoTBSvPZJCer8HbXkX/Ro70aXgtOVlRT+AzifksPJbK2cQC1Fo9vg4qngp2oF9Dx9LnrKQCFhxOITytkDy1DncbMzrXsuWZltd655bn6nO8GQ9bM5aPqFPh+mFLwsku1LL+hbpl1r23JZad4Tn89UwdXKzLn5QY4H874tl4IYt5/QMIdrckObeYpxeGMaCxI68+ZJjHoqBYZ2jpVom2ZhPXRHEuqYBfBwXi53h/i408Xlsr5ywhhLiPqnUlsBBCCFFdvLsmhOPRmTzd0hs/JyuSs4tYcTSWwxHpLBnTFtcr7Tqmrw5hf1gqPRt70LuZF2qNjkMRaaw7lUBSdhFfDm5a6rjf7wzDSmXKsLZ+5BQWs+hgNG+vOksdN2t0enimgz9Z+cUsPhTNWytPs2xsu1IVqWtPxpORr+bplj44W6vYHJLE7M0XScwq5KUuFd9g2RmazHvrzuHraMmI9n6Ym5pw8HIaX269RGhiDu9cab0Qk57PpGUncbUxZ0Q7P6xUphyLymDhgWhi0gv4qF+jCh8jwMWaGU82qHA9gKWq4iTyjTQ6HYlZhRy6nM683ZfxcrCg95VWEGqNjvjMAjrWceFYVAZzd4VzPiEHpQKa+znw2mN1CXSxLvWY+WpNmcfIyje0mErNVZfaNq+cbTPz1Ve2LSqzTgghjOm9zbGciM9nQGNHfB3MSc4tZuXpdI7E5LFoaK2SJOSMLbH8G5lLj/r2PBnsgFqj53BMLn+fzyQpt5jZT5aeD+WHA0lYm5kwrLkzOUVaFp9IY9rGWGo7W6DT6xnZwoWsQi1/nkxj6sYYlgyrjfV13/PrzmWSWaChf2MnnK1M2Xoxiy/2JJKQU8y4dqVHPV1vV3g2H2yLw8dexfAWzliYKjkYncucvUlcSC5k6qOGc0FMpprJ66NxsTZlWHNnrMyUHI/LZ9GJNGKy1Mx8wqfCx/B3NGd615vPd2J5kySyYb2C5FwdWp2+TII2s8DQUi01T1MmCVxYrCMuW83G0Cw2Xsji4UCbkgR6VIbhHONpa8ZfZ9NZejKdxJxizE0VPFrbjokdDS3VynMoOpdTCfn0behw3xPAQggh7j9JAgshhBD3WUa+moOX0+nfwpuXH7lWARTkbsOPuy9zMSkHV1tzLiXlsi8slYGtfJjULahku4GtfBjz+1EORaSTU1iMrcW1iz+NVs+PI1uUJHbzirQsPRJDgVrLgudalbQ1UGt1LDoYzfmEbFoFXGuXkphVyLyRLWjkbRhO2q+FN2MXHufPQzE81dQTH8eybQoK1Fo+23SBuu42zB3RoqRSdmArH+Zsu8SKo7F0C3anbaATey6lklek5eshDWjgaahC7t3MC6VSQVxGAWqNrqT9yo2crFU80cjjjl7z8hyOyODNFacBsDBT8vpjdbGzNLyWeUUa9EBkWh5vrDhN32ZejGznT0RqHgsPRvHSwuMseK4VPo6WNPSyw0SpYMf5ZEa08y91ob7jyqRwRRrDxXoTHweWHYll67lkmvg4lGyn0+vZfTH1yrb3doJaIYS4G5kFGg7F5NG3oSMvtb82SiHIxYKfDiVzMaUQF2szwlIL2R+Zy9ONHXnloWvf1U83cWLsqggOx+SRU6TF9roEo0anZ25/f6yuJHZz1TqWn0qnoFjHT08HXHfO0rPkRBqhyYW09LnWRiwpp5jv+/nTyMNwburb0JHxqyNZdjKNJxs4lLQ/uF5BsY7ZuxMJcrHg+74BmF1pfTCgsRPf7Etk5ZkMugbZ0cbXhn0ROeSpdXz5lFdJ79ungh1RKiA+uxi1VofKpIJzlpUpj9e1L3ddZTXxtOJSahF7InJ4pPa1SXqTcos5l1wAQJGmbKXsr0dT+PNkOgDe9maMv+7vllNkOMesO5dJRoGGEc1d8LI343B0Hn+fzyQivYjv+vmX+7yWnUrHRAlDm93fOTCEEEL8NyQJLIQQQtxn1ipTrM1N2HE+mTpuNjxUxxlnG3M61XWlU13Xku2C3G3Y8trDZap/MvLUpZK81yeB29d2LlXZ6+9iuDDuXM+1VF9bH0fDxWxKjrrUsTvUdi5JAIOh9cHwtn7MWBvCvkupDGlTuooL4EhkOtmFGkbUcyWvqHSFa7cGbqw4GsvuCym0DXTC7UqF8w87w3mmgz9NfRxQmSp576ngW75uGq2O3KKyFbTXUyoV2FlUPCz2et4OFnzcrxHZhcUsOxLLmytO8/rjdenfwptireEiOTajgNcfC2JAS0O1V+d6rtR1t+HNlWdYsPcy7/duiJO1iv7NvVlxLJapq87w/EMBWJubsjUkiS0hSZgqFZgqDRfTDwe5UM/dhjUn4rBSmdCjkQdqjY4/DkSRnF0IUDK5oxBCVAVWKiXWKiU7w7Op42JOxwBbnK1MeTjQlocDr825UcfFgk2j6pY61wBk5GtKqnfz1bpSSeB2fjYlCWCAAAdD0rZzLdvS56wrydyUvNITeLf3tylJAAOYmSgY2syZ97fGsT8yh8FNyyYrj8bmkV2kZVgtw0S41+tax56VZzLYczmHNr42uNoYzqfzDiQzsqULTTwtUZkoebebd5nj3kijvfUEyCZKRanX40aDmzqz6UIWn+9KIK9ISwsfaxKyi/l2fxLmJkqKNNpyzxnt/Gxo7GFFdKaaJSfSGLUigi+f8iPY3RKNzpA0jstW89OAQIJcDO08Oteyw1qlZOmpdDZdyCrT7zcmU83R2Dy61rHD0678yZaFEEJUL5IEvoe8nGw49NmD0Y9VCGPztDdn3+ttjR2GEJWiMlUytUd9/rcxlM82XeAzoJarNe1qOdGrsScBLtaltt12LpnDkenEZhSQkFlARn4xVy/5buyVd7VH8FUmVy6iXWxKD9u8enGtu2H/Wm5lJ2r0dzZcYMdlFpb7fGLS8wH4Yddlfth1udxtErMM+z5S35WDlz3YdDaR49GZmJsqaeprz8NBrvRo5HHTdg6nY7PuaU9gf2frksndHq3vxjM/H+GHXeE80dAdiytxmCgV9GlWejhvhzouuNuZcyQio2TZhK61QWHo67s/PA0AXydLPh/YhAlLTmBnaVpyvNmDmjJz/TkWHYxm0UFDv8g2gY681b0e09eElFQjCyFEVaAyUTKliyef7kpg9u5EZu9OJNDJnHZ+1vSo70DAdW0BzEyUbA/L4khMHnFZahJyisko0Jacs2485zhblb78vHrT09n6huVXDnBje9haTmVbEvg7Gs6DcVnFZdaBIZkJMO9gMvMOJpe7TWKOYd8ute04FJ3H5otZnIiPxtxUQRNPKx4KsKF7PYebtnM4k5h/1z2BPWzN+OJJPz7aHs9nuxMBMFVC72BH7CxM+O1oKnYWZWNo7n3tXN7Kx5qxqyKYdzCZb/r4Y2FqeDEbe1iVJICv6tvIkaWn0jkak1cmCbz7cjbAXVc3CyGEqDokCSyEEEL8Bx6p70bbWk4cCE/ncEQ6x6MyWHIohmWHY3m/TzCP1ncjr0jDK3+e5EJiDs18HWjoaceTTTxp4GHL0iMxbA5JKnPcu60ivVqxej3tlaqhiiaMubKaMQ8H0tDbrtxtrlYrmyqVvPtkA57vGMC+SykcjcrgVEwWhyMy+PNwNPOfaYmDVfkVRnXcbZgzpGm5664yr6CVxK1Ym5vyUJAzK4/FEZNRQD13GyxVJliamWBazpBYZ2sVYcl5Jf82VSqZ1C2IFzoGEJGah62FKYEu1iRkFaLR6fF2sCzZ1slaxVeDm5GQWUBSThEedhZ42Fvwz+kEAHyu21YIIaqCLrXtaONnw8GoXA7H5HIiLp8/T6az/FQ6Mx7z5pHaduSptby2LpoLKYU09bKigbslPRs4UN/VkuWn09hyMbvMcSs6r1T2TFbeOe9qpWtFp4OriehRbVxp6F7+962tubLk+O909eK5Vi7si8zhWGwepxMKOBKTx7JT6czrH4CDZfmX0HVcLPjyqbKjZ65nbnLrZxrsbsmiobW4nF5EvlqHv6M5dhYmfLQ9HhMluNve/MZhXVcL/B3NuZBiuBnramPY3smqbNxXk/L5xWXbEu2PzMXewoTWPmVvFgshhKieamQS+Fh4IuN/3Mrobk0Y8/jNLx6rovj0XPp9srrk34809uOTkZ3LbBcSncqYHzbx7ZhutKxdtmfiyYhkFmw9xYX4dPR6aBrgypjHmlLf5+Y9nSb/uoP8Ig1zxz1eZl1WfhE/bj7J7rMx5BSq8XW2pXebOgzsUB/lPRrOWqAuZsRX/6DV6VgztX+Z9X8fCWPZvlCiUrKwt7agQz0vRnVrgrtD6R8oGq2OZftC+ftoGPHpuTjZWNC1iT+jujXBytzwY+jr9cdYsudcyT4/jH2s3NdSVOx4TDavrDjP8+28GdWh4skyqqqErCIG/nyy5N9dghyZ9ZRhRmaNTs+K44n8czaZ+KwinKzN6FjLkVHtfUqq/K46EJHJH4fiuJicj1IBwR42jOrgQxNvW250OTWfBf/GciImG61eTy1nK55t6037Wg735DklZhcx8vfTvPZoAD0bupZZfy4hl98OxXEmLoeCYh0edioeq+/CyDZeZXqz7gvPYNHheCLSCjAzUdDKz54xHX3wdjBUkuy/nMGUNRdLtq+u74P7LV+tITw5Dw97C7o2cKNrA8PkNSejM3ll6UkWHYzm0fqGNgqhiTm8+URd+jYvPfQ0LU9d3qHv2tWq3utFpRmW+TuV7QcM4Hnl729uqqT1df2FAbILijkalYGbrWGbxKxCYjPyaRXgxJA2fgz5P3v3HR1F2T1w/LvpvfcOoYTQCb1XpQkiKNhF8MWuiF3Qn4IodrF3RQUBpSq9915CIBAC6b33tuX3xyYLSxJCaJOE+znnPefN7DOzdyKb2blzn/t0D6BCo+WLzdH8cySJTZHpTAir+d+Mg5V5tePX12cbz7LxVBq/PtrNsPheleLKabuWZiaoVCraeNlzLCGXnOJynC9KTOt0OpLzSg3nrdZo2RSZjoe9JV0Cneno72QYu/+8vidjlwB9RVVGQRn7z2fTwd+RABcbvC9K+O47n42FmQmhPjUn0oUQQgnFFVrOZZXibW/O4BYODG6h/xt1LLmY6avjWHg0i0HBDvx9IofTGaXM6O/F2LbGVaTZxZdvi3C1EvKqXwvjc/TbApxqXrisqpWBpamKrpckNPNLNRxOKsKjshI5raCCxLxywvxsmdjRlYkdXanQ6PhqTxrLInLYHJ3P+PY1X5fsLU2rHb++ojJKiUwvYXhrR4JdL1TtarQ6DiUW0dbT2tC7d9o/Mai1On66u3m14xRXaA0J5+YulliZqYjJrr4IaVLl79P7ksRymVrL6fQS+je3x+wKEtdCCCEah6srnxE3RadmHvzfpD5M7BtS7bXErAJe+X27oVrrUvujknnyuw2k5hYxeXB7Jg9uz/m0PKZ+tY7w2JqnQQF8teYIuyKTanytoKSc/329jn/2RtEu0J1nR4XRNsCNz1Yf5rU/dlSbony1Pl55kMSsglrjm7N0L6UVav53Wycm9GrFntNJTP5iLfEZxtUG7/69l/n/HaaZhyPPj+5Kv1B//tp1mie/20h55YI9t3cK4v8m9WFgO//rErtovDr62jNreDB3d77wEOCdNdF8tSOe5m42PDcoiB5BTiw/nsaMZacpv2ghp+1ns3l5+RnSC8qZ0tuPyT19Scgp5dmlkRxNMP53eSqlkGmLTnIqpZD7unkztbcf+aVqXlpxhu1ns6/5PHJLKnh5+RlKaqjoAIhKL+KpJaeITC3k7i5ePDcokGA3G37Zl8TLK84YTdv8NyKdV1dGodbqeKyPHxM6e3EoPo8pf0aQmKOvLmnlbsus4cE82P3yq2Hf6s5nFPH4H0f4bU+s0fZWXnZYmJoYKqNyS/TTUYPd7YzGRSTlcSw+FwDNdfpbW2VHVCaJORcSweVqLQv3x2NhZkL/Vm417tO9mQvWFqYsOZRIfqnx9NsfdsYwa8VJjifo412wN47n/jrOqeQLnwVzUxNae+kfkNRWFXa9+Dpbk1tSwaIDxlN0E7KL2XomgwAXG4Iq21+MaO+FVge/7o41GvtveAq5xRUMDtEn781MTfhxZwwfro8y9BIGyCws48/98bTytKNLoBOgv3F/b+1p/tgbZ3TMYwm5bI/K4M5OPpdtiSGEEDdbTFYZTy2PY8HhTKPtrdytsDBRGVo15JXoe7Y3dzVOvkakFnMsWT9zQnN9L1nsjCkg8aJEcLlGy1/Hs7AwVdGvefUH7wDd/W2xNjdhaXg2BWXGyemfDmbw1oYkjqfor4O/H8lk+up4TqWVGMaYm6po7a5PyN7oHu7nskr5eEcqW88Zf39cdCyLrGK1Uc9jd1tzzmaWsSvG+L5p09k8UgsqDL8PSzMTBgU7EJNdZmjxcOG4+u+eg1oYP4yMyixFo4PW7jJTRQghmpImWQncVPi62DGiS/Unu7tOJfLOkj3kFVd/mlvlwxUHcLC24KenRuBoq/9iNqxTEHd/sIIv1xzl+ydvNxpfUFLO+8v2sel4XE2HA+DnzSeITc/noUFteWpEF8P2Vj4ufLjiAGuPnGdkWHB9T9PI1hPx/HvonGGl+YtFp+Tw+/aTBLo78PMzI7Cz0j/VHxUWzKSPV/H+sn18PU1fvRwem86aw+cZ3TWYWff0NhzDx8WOz1YfYt2R84zp3pIQP1dC/FxJzCxgW0TCNcUuGjcfR0tuD72Q8NoSlcWWqGzu6eLFswMDDds97C34YXci285mc1sb/fgf9yRiYWbCVxND8XLQf94GtnLhgV/D+XZXAt/d2xbQVxO+v+E8ZiYqvpnUFm9H/dgRoe7c+8txvtmZwICWV1/1GJFcwNtrzpGSX/vfho83x2JmouKH+9oZYh3X0ZMvtsWx+Egq285mM7iVK2VqLfO3xRHoYsXXE0MNFcJ9mjsx+Y8IftmXyKwRLXC3t+D2UDeOJOTz+4Hkq469qWvn60j3Zs4sP5pMYZmGTv6OlKu1rItIpbRCw73d9Q+i+rVw4+9Dibyz+hTjuvhiZ2lGZEo+6yLSMDVRodbqKCy7ztVVKpi24Ajjw3yxtTRj7YlUzqYXMn1YS1ztaq6qcrAyZ/rQlry35jQP/XSQMR29cbG14GBsDtvOZNDZ34nh7fQPVCZ282NTZBovLQ1nbGcfvB2tSMotYdmRJDzsLQ1V0TfKnZ192BSZxuKDiWQVltM5wImUvFJWHNX/e31jVAiqyn7JI9p5se1MBn8fTiItv4yezV04l1HEyqPJNHOz5f4eF6b5Tu4TxNw1p3n+r2MMDfWkuEzNsiPJFJWpmTuuneGYXo5WjGjnxX8nUtEB7f0cScopYemhRJq72fJo36Abev5CCFFfbb2s6eZvy4qTuRSWa+nobUO5Rsf6M3mUqnVM7KRPRPYJsuefEznM2ZTMne2csbMw4XR6Keuj8gzXrEsXYrsenlgWy13tnLG1MGV9VC5nM8t4rq9ntX7DVewtTXmuryfztqbwyOLzjG7jhIuNGYcSi9h+voBOPjbc3lrf9/buDi5sjs7nlTUJjG3rhJe9Bcl55Sw/mYO7rZmhKvpGGRjswKJjWXy6M434nHJ8HMw5nlLC+qg8RrR2NFqY78neHoSnFPPOpiTGtnXGz9GCyPQS1p3JI8DJgse6X5gN9nhPD44lF/POxmTGtC0myNmSvXGF7IkrZERrR7r4GlcwV/VRrqv1hBBCiMZFksCNzKyFO9lwLJYgD0d6tPJmw7HYamPyisrwdrYjpL2LIQEM4OFoQ5CHI6eTsozGn4jLYMYvWyksLWfq0A78uCm8xvfeHhGPjYUZU4Z2MNo+rmdLvt9wnGX7zl5TEjgjr5j3/tnH+F6t2R2ZWP39Tyag08HkIe0NCWAAd0cbRnUNZvGu08Rn5BPg7kBmQQlt/FwZ36uV0TG6t9QnJU4nZTPmqiMVt4JV4enYWZryWB/jaepj2ntQWqHF1fbCl+LE3FKC3WwMSVUAH0crglytOXtRD9ETyYWczyrhsT5+hgQwgL2VGc8ODCQ5r4xytbZaS4Yr8e3OeP48mIKbnTljO3iwMrx6xX9hmZrI1EL6t3A2ihXg9lA3Fh9J5VhCAYNbuZKaX0YrD1sGtnQxiqelhy2OVmacSSu69PCiDnPubMfC/fFsOZ3BzrMZmJqoaO1pzwd3d6B3sP6GOizImbfHtuWPfXH8vCsWc1MVXo5WPNa/GUGuNrz09wkOxGQT4lVztdPVGBLiQXN3WxYfTKCwTE0LDzvev6sd/VpVbyVysVEdvPFysOLP/fEsOZhImUaLt6MVU/oGcW/3AMO/m0BXW768rzO/7Ylj7YlUcorLcbQ2Z1BrD6b0DcLB6sbeYJqbmvD5pE78tieOTafS2HomA3srM3oFuzC5TzPDIngAKpWKuePasfhgAv+Gp7LvfBZONhaM6+LDY/2aG1XsjurgjaW5CYv2J/DV1nPYWJjSJcCp2jEBXh7eGj9na9adTGVzZDpudpZMCPPjwV4Bht7JQgjRkMy+zZdFx7LZei6fXTEFmJqoaOVmxfsj/ekVqJ+tEuZny1vDfPnzaCa/HszA3NQET3szpnR3J8jZglfWJHIwoei6VpMObuFAMxdLloZnU1imIdjVineH+xklR2syMsQJTztzFh3LYml4NuUaHV725jzazY2JHV0NLRYCnS2ZPzaQBYczWXs6j9wSDY7Wpgxsbs/kbu7YW97YmRvW5iZ8ckcAPx3IYH1UHgVlGvwcLXihnxdj2joZjfVxsOCHCc348UAG68/kUVCuwcPWnHs6uPBQmBt2F8XqbGPGd+OD+PlgJtvPF7C6NBcve3Oe7u3B3R2qFyDkluiT9/YWMnFYCCGaEsWTwJ+sOsjiXaf58anhtA80vuH8ZfMJvl1/jK/+N5SuLbypUGv4a9dpNofHEZeeR7lGi6u9FT1b+TDt9k642tf+BePO95YBVOsx+8OG4/y4KbxaL9hdkYn8uf0UpxOz0Oh0tPByYlK/NtzWqVmd59Tj5d/rHLP81XH4uNjVOe5SMWl5TLu9I/f3b8uCrRE1jnG0teSLx4ZW215YWk5CVgHezsZPeuMz8gn2cuLZUWG08XetNQmclldMsJcTVuaXruprgq+rHacTs9DpdIbqp/rQ6XS8s2Q3znaWPDOqS41J4LRcfdKppbdztdcC3PRP5SMTswhwd2Bw+0AGtw+sNu50on7Kk5fTrbnAwedb41h6NJVvJ4XSzsf4y/Jv+5P4YXcin08IISzAkQqNlqVHUtkSlU1cdgkVGh0utub0CHLksd7+uNjWnriY8ONRAP6e2tlo+097EvllXxLz725DF/8LlRR7zuew8FAKZ9KK0OqguZs1E7t4MzTk8v2rAfp+sr/OMUundDJKutZFo9URnlRAt0BHrM31X6BLKzSYmZrgbGPOtL7G7UMCnK1JzS+jTK01LFJVrtaSUViOm92FBxZHKltD9AxyAvT/7ksqtNhYmBqqiq9WdEYx93TxYnIvX3ZE59SYBLY2N2XR5I6Y1PAZzS3WT+mvmpof6GLNl/eEVhuXkldGXqmaEK9b8zN0LWwtzXisf3Me6199hsfFLu4ZfKndrw4y/P8ugc5GP1cZ1cGbUR28r3g7wH09ArivR+2L2dS2b1iQM2FB1f8mX6qVpz3vjmtX57gbxcrclGkDmjNtwOV/96Bv9XB/z0Du71n9GnKpoW08GdrGs85xFmYmPNIniEf6BF1JuEIIoTgbC1OmdHdnSvfLPxC8uGfwpXY80cbw/zv72hr9XGVEiBMjQpyueDvAvZ1cubdT7d8Ra9s3zM+WsCvo2dvSzYrZtyu3voGbrTmvDLqyNlue9ua8MeTKxjpZm/FCfy9e6F/3+if3dXblvs51fw8XQgjRuCieBL6jWwsW7zrN2iPnqyWB1x45j7ezrSE5+9ofO9gVmciosGDGdm9BuVrDvqgUVh6IJi23mM+nDrkuMS3aGclnqw/RLsDNsLDc1hPxzFq4i7j0/DoXm/u/SX3qfA/nWqbY1uWXZ0Zgbla/J9BZBSWcTcnh+w3HKSlT88Rw48TcbZ2CGNW17gpeGwszii7p/Vglt6iMCo2WvOIynGytahxzOQt3RHLkfDo/Pjm8WpLZ8P6Vi7kVlVWPIbdIP/09I7/6AkdqjZaUnEL2RaXw9dqj+LrYMbZHy3rH2BSMaufO0qOprI/MrJYEXn8qEy8HC0Nydta/Z9l9LpcRbd25o70H5WotB+LyWH0ig7T8cj4ZX71X9dVYciSF+dviaettZ1hQbPvZbP5vTTRx2SV1LjI2a3jd/3adapkeWJvkvDLKNTq8HS3ZEpXFz3uTiM0qwcxERa9mTkwfHIjHRQtMPTcokFdWnGH22nM82ssXlQp+3ptETrGaJ4dfSKzFZev7y1lbmPDuunNsjcqmVK3Fw96CyT19uaP91U+Lf29sqxrbqFzM1ERlWNDtUn8dTgWgi3/NlTTpBWVEphbx3a4ELExVTO7pW+M4IYQQQgghhBBCNDyKJ4FbejsT4uvC5vA4XhjTDbPKJEZEfAZxGflMHdoBlUpFVHI2O08lMrFvCC+M6WbYf2LfNjz6xRr2RSVTUFKOvbVFbW91RVJzivjiv8P0b+vHBw8NNFS1Turbhtf+2M7Pm08wrFMQQR6OtR6jpj6+10t9E8AA93+ympzKJOldPVvRO8Q4eXOlx+wY5MGu04mcjM+kbcCFqsVTCZmk5BQCUFZR/75fUcnZfLv+KFOGtqeNf+1PnDsGebBoZyTrj8bQMehCskyr1bEtIr7W998flcILv2wB9NVgL97ZHUebq0vCN3Yt3G1o7WnL1qhsnhsUZFjc4mRKIfE5pUzu6YtKpeJsRhG7zuVyd2f9wmFV7u7ixWMLIzgQl0dBqRp7q2v7E5KaX8ZXOxLoF+zM3DEtDZ+3e7p4MXP1WX7bn8TQEFcCXWqv8r+4j+/1UlCqX+jkYGXS+76u3rTobcPJlEIWH0klenExP93fDgdr/fm387ZjYhdvft2fxLaLFnd7vK8/I0Ldqx33tZVROFqb89LQZmh0OpYeSWXexhgKyzTc27XmSs261JUAvpw/DiRzIC6PNl629A2uXtVZWqHhrh+OGX5+rI8fod71n8kghBBCCCGEEEIIZSieBAYY3a0FH604wL4zyfQN1Vf9rTl8HpUKRnXVJ1Rb+biw5Z1J1VYRzy4swbayP2xRacU1J4G3RsSj0eoY1jGo2sJrt3VqxraIBLafTLhsEji3qLTO93GwtsTkBq8uC/rp5s+ODsPKwoxdpxJZti+Kc6m5fD1tmCHhfqUmD2nPnjNJvLJgG9PHdKONnytnU3L4ZOVB7K0tyC8ur/cxyyo0vLloF619XXl40OWnCvdv60drXxeW7zuLraU5I8OCKVdr+HXLCdLy9K0ianp/X1c75j00gPziMhbtjOSFX7bw4tjuTOjdul6xNhUj27rx6ZY49sfm0qe5PuG37lQGKmBEW31CtaW7LRue7sql/0RziisM/cWKyjXXnATeEZ2NRqtjSGtX8ioTpFWGhriyIzqHHdE5PNi99iRwbknN1ekXc7Ayq7EFQm0qtPqlrOOyS5l3ZyvD72lASxc8HSz4dEsciw6nGNpCvLoyigNxeXQLdGBEqDsqFWw5k823uxLIKa7gmcqF5aqOa2thyhd3tzH8PRvcyoUHfgvnpz2JjG7nfs2/1/r482Ay3+5KwNnGjLdHtaixnYtGq+PNEcGYqFRsPJ3JD7sTOZdRzDujb82KeiGEEEIIIYQQorFpEEng2zsFMf/fQ6w7GkPfUD/UGi2bjsfRpbknPi4XpiZbmJmw8Xgs+6NSSMgqIDm7kJzCUqpyFlqd7ppjScjU9+yctXBXrWNSsgsvfz5vL63zfa62J3B9qVQqw2Jtg9sH4mhjycKdkaw7GsPoK2gBcbG2AW588PBA3v9nH6//sQPQV9Y+NKgdablFrDwQXe8k/Bf/HSYpq4BvHr+NgpJyw/aq/5a5RaWYmZhgZ22BqYkJnz46mLcX72bBtpMs2HYSgB6tvHn1rp68/scOHGp4/yAPR0PSfkiHIO7/dDVfrTnCiC7Nsb0FF+QZFuLGV9vj2RCZRZ/mzqg1WracyaazvwM+jhdaBZibqth0JouDsXkk5paSkl9GTrGaqhThdfi4EZ+jf2Dyf2uiax2Tml9W62sAo785Uuf71LcnsLW5/mGCl4OFIQFcZUx7D+ZvjeNgXB7T+vpzMC6PA3F59GrmxIfjLjxYGBbixvsbzrP4SCrdgxzpEeRkOO6dHT2NHmhZmZsyvI07v+5P4kRyAb2b191j9VpptDo+2xrL8uPpuNqa8+n4EKP//heztTQz9CweGuLKG6ui2BKVzZ0J+Ua9nUXj4u1kXWNPYSGEEKKh8XawqLGnsBBCCCGuXINIAjvYWNK/rT87TiVQVFrBwegU8orLuKNbC8OYwtJynv5+E6eTsujczJO2/m7c0TWYUH83Fu48xbojMVf13hqt1uhnbWWl3mvje9aapHVzuPwKtzUtynapyy1idyPd3rkZC3dGcjoxq95JYIC+bfxY8dpdnE3JQa3REuzlhI2lOY9/sx5PJxss6tmuYldkIuVqLVO+XFdzvG8vpUtzT755/DZA/3ubP3UoydmFpOUV4e1kh5ezLasP6pOI/m6XXxnY1sqcfqF+LNl9hvjMfNr43XoLHjhYmdGvhTO7zuVQXK7hYFweeaVqRrW90LagqEzNc3+f5kxaER397An1tmNUOw/aeNmy+HAK6yOzruq9NZdkjqt+fHlos1qTtG52l0/Uf3oFvYkvt4hdTTzs9Q8TXGyq72duaoKDtRnF5frWI2cz9H2oR7at3pbijvYe/BuRwYG4PHoEOeFuV/txXSvPs7hcW+216620QsNb/0Wz+3wufk5WfHxX61p7BdfktjZubI/O4UxakSSBhRBCCCGEEEKIRqBBJIFBv0DcpuNx7IpMZGtEPLZW5gxqd2FBpSW7TxOZmMUrd/Xgrp6tjPbNLiip8/imJiaUlKurbc+8ZN+qxK+DjQXdWxr35kzNKeJ0UhY2FpdPely63812KDqVOUv3ML5Xax4c2NboteIy/e/A0rz+vYWPxqQRm5bHuJ6tjJKnBSXlRMRnMrRjUL2P+fa9fWvs4/vWol2G16uqizPyitkblUynIA8C3B2MkvT7ziRjaWZq6FX88cqDbDgWwx/Pj8bd0cbo2EXX8DtoKka1dWfzmWx2ncthR3Q2thamDGh5ofp06dE0TqcV8eLQIO7s4Gm0b1ZR3e0XTFUqStXVk5lZhcb7ejvoE78OVmZ0CzRusZKaX8aZtCKszS+fnLx0v+vBydocb0dLEnJKUWt1ht7JAMXlGnKL1bTy0K8ubWGqf01bQ2W0rjLLral8MdTbjuXH04nJKqFnMyejsUm5+qro+lQsX41ytZZXV0ZxKD6fdt52vH9nK5ysqyelN0Rm8s3OeJ4eEMiQ1sYPS6oS4JZmV9+HWAghhBBCCCGEEDdPg0kCd2/hjaeTDWsOn+doTBrDOzfDyuJCeLmVC5u18HIy2u9EXAZHzqcD1at6L+bmYE14bAbpecV4VCYF84vL2BWZZDRuYLsAvll3jN+2RNAnxM+QKNTpdHy04gA7IxOZP3UIXs6213zON0qwlxNZBSX8s/cM43q2xK6yZ7JGq2XBtggABrT1r/dxD0en8sPGcII8HOnc/EJi8Ks1R1BrtdzXv/5TtC5e4O1iVb/3ixPqaq2Wd5fuZXTXYGbd09uw/WhMGlsj4rm7d2usLfTJLH83e3KLyvhzxymev6OrYWx8Rj5bw+MIdHeg2WX6Ojd1XQMd8bC3YN2pTI4nFXBbG1esLkqK55XoE+XBbsYJ9IjkAo4lFgAXEps1cbOz4ERyARkF5bhXVtXml6rZE5NjNK5/S2e+353A7weT6dXcyZBU1Ol0fLollt3nc/l0fAheDjd/Ib8RoW78vDeJv4+mMinswr/DPw8mowMGt9InRnsEOWGqimP5sTQGtnQxavOw7FiafkygEwD9g5353NKUf46lMqqdOw6VvX9ziitYczITbwdL2njd2L8tX2yP41C8vo3DB3e2MvrvfrHmbjZkFlbw1+EUBrR0MSTCyyq0LD6SipmJit7NnW5orLeKI3E5PLPoGI/2CWJKv2ZKh1NvKbklTPh2n+Hnga3deXdcO8Nr3++M4Vh8LvmlFTR3t2NSN3+GtKn+t/9sWiHf7zjPyeR8KjRaOvg58sTAYFp4VJ+Vs/1MBr/viyM2sxgrcxP6t3Jn2oDmOF7yQEOt1fLXgQT+DU8hLb8MdztL7ujozb09/DEzufqHGKUVGhbsjWPjqTQyC8rxcLBkSBsPHuoVWO0z9V94CksOJRKfVYyjjTk9m7swuU8Qng7GD7jKKjT8ukd/zKzCcjwdLbkt1JP7ewQYrolfbolm0YEEwz5f3NuJLoE3vn2MEEJUOZpUxHOr4nmkqxuPdnOve4cGJiW/nIl/njP8PKC5PbNv169JE5dTxg/7MziaXESZWoefowV3d3BhVBunyx4zs6iCyUtiaOZiyfyxgUav6XQ61p7J458T2STmlmNjYUrvIDumdnPH2cb4VrywTMOvhzPZeb6AjKIKbC1M6exjw5Tu7gQ6X/134RMpxfx8MINTaSWYm5oQ6mnFQ2FutPOyqTZ2V0wBfx7NIjanDDMTFd38bJnawx0fB+OWe9vO5fPmhqRq+wOMCnHklUE+fLUnjcXHLyyY/PmYADr7Ntx7aCGEuBU0mCSwiYm+d+0vm08AGLWCAOgf6seS3ad566/djO/VCjsrC04lZLL2yHlMTVSoNVBYWnuF4qiw5hyLSefZHzcxvldrSsvVrNh/FkcbC3IKLyzkFuDuwJSh7flhYzgPff4vI8OCsbU0Z2tEPIeiU7mtUxA9WvncmF/CdeJsZ8Uzo8L4eOVBpny5jju7t0AHbDgWS2RiFvf1D6VDLcnXyxnXsxXL9p3ltd+3c0/fEJxtrdgVmciuyCSmDu1AKx8Xw9isghIOnE3Bxc7quv2+vJ3tGBXWnH8PnUMHdAx0JzG7gMW7ThPs5cTUYR0vxNqjJRuOxbJoZySZ+SV0CfYkJaeQZXujAJh1T+8aF8C6VZioVIwIdeO3/ckARq0gAPoGO/H30VTeWXuOcR09sbMwJTKtkPWnMvWfN62OwvLqFdxVRoS6cTypgOnLTjOuowdlFVpWhqfjYGVGTvGFivwAZ2se6enLz3uTePSPEwwPdcfWwpTtZ7M5nJDP0NauN6TS90rc39WHPedz+Wp7POcziwn1tuNEUgHrI7MIC3AwLKLn72zFIz19+WlvEtMWneS2Nm6ogJ3ncjhSeQ69KpOltpZmvDS0GW+viWbqnxHc2dETnU7H8uNpFJdreHtUC6MF7NafygTg9tDqrSauRkxWMSuOp2Oqgr7Bzmw/m1NtjK+TJe187GnhbsPEMC/+OpzKU4tPcVuIKyUVWv47mUFCTinPDwpUJDkvGq6Ofo6M7eSDZ2Vv6YyCMqb9foRStYYJYX642VmyJTKdN1eeJD2/lHt7XJjtcz6jkKcWHsHWwoxJ3f0xN1Gx5FAij/9xhO8f7EJz9wuJ4LUnUpnzXyRtfRyYNqA5mYVlLDmUSHhiHj88FIa1xYUk7Adrz/DfiVSGtvFgUjd/jiXk8e3288RlFTNz9NX1llRrtExffJwTiXmM6eRDK087jsTn8tueOM5nFPHeXe0M15dvtp3jj33x+DtbM7VfEGqtjmVHktgTncWX93cmwMXG6JjHE/MIC3RiUnd/UnJL+X1vPAdisvn83k5YmpkyLNSTlh52bIvKYEdU5lXFL4QQAjp4WzMm1BnPynZcyfnlPLk8lnKNjvHtXfCwM2NjVD7ztqWQXazmwbCav4vpdDrmbkkhr7Tm78U/H8zkt8OZdPWz5Y5ezmQUVfD3iRwOJhTxw4QgHCsLAtRaHS/9l8CptBJub+1IW09rUgsqWHEyhwMJRXxzVxDNXOr/vWtPbAFvrE/E0tSE8e1dcLYxY9PZPJ5dGcebQ30ZGHxhhut/kbnM25ZCiLsVU7u7U1iu5Z/wbPbFF/L9hGb4OV5IBJ/L0hdozejvZVjzoopvZcJ4aEsHWrpZsf18ATtjCuoduxBCiOuvwSSBAe7oGsyvW04Q4OZA+0DjpFTXFt7Mua8fC7ae5MeN4ZibmeDtZMe02zsR5OHIjF+2sj8qudYer3d0a0FhaQXL9kXx2epDeDracGePlvi52fPa7zuMxk4d1pFmnk4s2X2aXzefQAf4udrzwpiujO/VusbjNzT39AnBy8mW37ed5Jv1x1ABLbydeefevtze+eoqzVztrfn28dv4dv0xlu4+Q2mFhmaejsy5rx/DOgUZjY1Nz+P//tpNl+ae1zVp/ur4nvi52bP2cAybjsfi7mDDPX1CeHhQO6NF6czNTPnysaH8suUEG47FsuVEHA7WFvQO8WXq0A4E3sJVwFVGtnVnwf5k/J2taOdj3Es5LMCR/xvVgj8PJvPL3kTMTU3wcrBgah9/glyseHlFFAfjcgnxrPlp/qh27hSWa1hxPJ0vtsXjYW/BmPYe+DlZMfPfs0ZjH+3lRzNXa/4+msaC/fqKAl8nK54bGMi4Tp41Hf6msDQ3Yf7dbfj9QDKbTmex8XQWbnYWPNLDl4d6+Bglayf38iPI1ZolR1L5flcCWp2OABdrpg+qfg5DWrviYmPOr/uT+GVvIiYqFaHetvzfyBbV/jvMXqevVLleSeAj8fnoAI0O5m+Lq3HMiFA3QxxPDwgkyNWaZcfS+HJHPOamKtp42TF9UCDdg5yuS0yi6fBxsub2dl6Gn3/eFUNWUTlf3deZTgFOAIzt5MNjvx3ip12xjOnkg62l/mvIF1ui0Wh1fPNAF7wqk8gDW7vz4E8H+WJLNJ9O7ARAcbmaL7ZE09rTji/v64xF5eyBEC97Zq44ydLDiTzUS1+FFZGUx38nUrmriy8zbtO3kbqzsy/2VmYsO5LE2E4+tPer/7VgcWXC+bkhLbinm7/huDYWpqw+nkJEUj7t/Rw5l17In/viCXCx4ceHwwznOrK9Nw/8eIAP153hi/s6A7DiWDLHE/MYFurBW3eEGpLIYUHOvLQ0nD/3xfNo32a09rKntZc9iTklkgQWQohr4ONgwW2tLlwDFh/PpqBMy9u3+TKoMjE6po0zjy49z2+HMxnXzhk7y+qzp5aEZxOeUlzje6QXVrDgcCZdfG34eLS/4W97iIc1b6xLZGl4NlO76wtz/ovM5WRaCU/28mBSpwv3swOD7Xl8WSzf7E3jg1EBNb5PbSo0Oj7akYoJKr4aF0iwq/76Oq6tM8+uiuPjHamE+dlib2lKmVrLF7vTCHCy4MtxgViY6q+vvQPtmLI0hl8PZTBziK/h2NFZpThYmjK2be2zUVq7W9Pa3ZrEvHJJAgshRAPRoJLAvq727Jv3YK2vD+0YVGvf2f0fXNgvLNjL6GcAlUrFff1Dua9/6GX3rTKkQyBDOgRW296QPHZbRx67rWOtr/dv60//q2j7UNPvo0qAuwNzH+hf5zHCgr2Y99AAlu87W+fYmqx47a4at1uYmfLokA48OqRDncewsjDjieGdeWJ456uKoanzdbJi5ws9an19SGvXar1gq+y6aL8u/g5GP4P+8zYpzNuojUJN+1YZ1MqVQa0a3iJ9NhamTOvrz7S+dX+O6nMOnf0d6HwFC6qteTKMO787ckXHvNjItu6MbFt9iub4zl6M7+xVwx61G93Og9Ht6j9zQAiAbkHOhgQwgKmJii6BzpxJKyQuq5hQHweyi8o5EJPD7W09DQlgAG8nawaFuLP2RCqZhWW42Vmy51wWeSUVPDGwuSEBDDAoxANvx3OsPZFqSAKvjUgFYGI348/v/T0CWHYkibURqVeVBF51LBl/F2smdPUz2n5v9wBcbC0wr4xrx9lMdMAjvQMNCWAAd3tLRrT3YumhROKziwlwsWFHVAYATw4MNpql0jvYlRYedqw8lsyjfRtfqxAhhGgsEvPKAegZcGHmiZmpiu4Bdiw5nk1cTjltvYwX9j6XVcoP+zOY0t2db/amVztmSn4FoZ7W3NXO2ehve5ifvojiTMaF2agHE4oAGBPqZHSM1u7WBDlbcjy55kTz5ZxKKyGzSM2oEEdDArjqvO7t5Mob6xLZei6fMaHOpBZU0MrdigHN7Q0JYICWblY4Wpkaxao/97KrqkwWQgihrAaVBBZNR3FZBX/vOUNYcP0STkIIPZ1Ox6JDKXTwta97sLhpPtt0lqWHEvnuwS608zVOIP62J5bvd8Qwf1InwoKcqdBoWXIokS2R6cRlF1Oh1uJqZ0GPZi481r85LrYWtbwLjP96LwD/PNnLaPtPO2P4eXdstV6we6IzWbg/gdNpBWi1OoLdbZnYzZ+hoXVX0/d5f2udY/5+vCfeTtZ1jrvYKyNCatx+JrUAExV4VLYTOZmcD0CoT/UHI228HVhzIpXIlAL6tbTkVOXYtjWOtWfL6QwKS9XYWZlxKjkfR2tz/JyN4/ZytMLJxpzIlPx6nQ9Aen4piTklTAjzNcwGKCnXYGFmQqCrDf/r39xoLFBjT+OqNhCnUwoIcLEhPb8MByszPByqL4QZ4GJNdHohGQVluNvLDbcQ4srN35XK3ydy+HpcYLX+rwsOZ/LjgQw+vSOAMD9bKjQ6/g7PZuu5fOJyy6nQaHGxMaOHvx1TurvjYlP7beM9f0QDsOQB43Z+Px/M4NdDmdV6we6NK2DRsWzOZJSi1elo7mLJPR1cGNKy7gdz/b+JrHPM4vuD8Xao/RpbkwAnCw4mFBGfW0Zr9wvXjaTK5LCbrfH5l6m1vLMxmbae1kzs6FJjErijjw3f3BVUbXtUhn5hci+7C33sZ/T34qEwN2wsqlcb55VqjNacuFLplQsyt3Crfm3xr2ztcCa9FEIh0Ll6P2PQ91DOK9UQ4n7hGIVlGlILKgwJc7VGhxadUfJYCCFEwyRJ4AYsKbuQtUfO4+VsS+dmyk2LvxplFRq6t/Tm/gHVK68bktOJWcSk5xGdWr03qri1JOeVsf5UJl4OFnT0q7tK92awsTDhrZEt6h6ooIyCco4k5BObXaJ0KDfF6A7eLD2UyPqTadWSwOsi0vBysKJL5UKAM5efZHd0JiPbezGmkw/lai37Y7JYdVy/SNknE2ufyVEfiw8mMH9zNG19HJjaNwiAbWcyeGvVKeKyiutcbO7NK+iN62RTv5vpSxWVqYnPLmbpoUSOxOdyT1d9j2DQ9w4G8Kwhwelup3/f1LxSo7Ee9tVvaKuOl5pfSgsrO9ILyvCoJWnqbmdJSl5pja9dTmyWvhLL29Gafw4nsuhAAil5pViamTCkjQfPDWmJXWV/x6rexEXl6mrHyS3WJxUyC8sMY9Pyy9BoddVu9HOLKwxjJQkshKiPUW2c+PtEDhui8qslgTdE5eFlb04XX/32Nzcksie2kBEhjowOdaJcreNAQiGrI3NJK6zgo9H1a0VQmyXHs/lyTxptPa15tJu+3dX28wW8vSmZuNzyOhebmzmk7jZzTtb1v8W9v7MrB+KLeG9LCtP7eeFuZ8bms/nsji1kZIgjnvbGC49+szedjKIKPhjlb9QirDZanY60ggrCU0r4dl86jlamRm0fnG3Mqi0UB7DpbB6ZRWr6Nav+QLEuVb16i8qrL56eW9nDOLO4+jUK9Ank0+klfL8/AwtTFY90vfDfpaofcEZRBf/7O4azmaVoddDa3Yr/9fSgq58s/iaEEA2VJIEbsGMx6RyLSWdQ+4BGlwR2trPioUHtlA6jTuuPxbJwxymlwxANwPGkAo4nFTCwpXODSAKrVCoe7O5b90CFRWUUGXoX3wpaeNjR2sueLZHpPDe0BWYm+husk8l5xGcX82ifIFQqFWfTCtkVncndXf14fmhLw/53d/Xjsd8OsT8mm4LSCuytzGt7qyuSmlfKV1vP0a+lm9GCZPd08+eN5RH8uieWoaEeBLrWfkN2cR/fG+X9tafZclrf8iDUx8HQsgGgsEx/A2pVQ/WTpbl+W0mF5pKx1auNrCrHllaOLSpTGypuq481MYyrj4JS/fuvOp5MdlE5D/YKxNfJmv3n9cn98xlFfPNAFyzMTOjg58Tig4lsPJVOBz8nwzG0Oh3bK/v5lqn1N+Yd/ByJSitke1QGg0MutF9Jyy81VEpXjRVCiCsV7GpFa3crtp7L59m+nphVPmQ6lVZCfG45j3R1Q6VSEZ1Zyu7YQia0d+bZvheuCRM6uDDtnxgOJBRRUKbBvoaeuPWRVlDBN/vS6Btkx7vD/QzXrLs7uDBrfRILDmcypIUDgc61P/C6uI/v9eRma86U7u7M25bCMysvrJvQJ8iOF/sbtzfbF1fIsogcZg7xqZYcrk18TjkPLT4PgKkKXhzgja/j5R+wxuaU8dnONMxM4OGwyyfHaxLqaY2pCWw9l8/9nV2NHjJuO1f7taW0QsuE36MNP0/t7k4bzwsPX89l6x+inkgpYVInFx7u6kZCbjl/HcvixX/jmX27H/2ayUw2IYRoiCQJ3AD5uNhdti+vuH6eGx3Gc6PDlA5DKMjb0bLGPsXiyvRp7nzL/f5Gtffik41n2X8+mz4t9FVM6yLSUAEj2utvnlt62rFher9qVZ05ReWG/rBFZZprTgJvj8pAo9UxtI0HeSUVRq8NC/VkR1QmO6IyebBX7UngqqrUy3GwNr+iSqfajGjnxW2hnpxOLeCvgwk8/MtBvr6/C37O1uh0OgBqOnrVW1a9VjkUVQ2jLw2vamxNVCpVjceoi1qjv1lOzCnhx4fDaOWpv8kd2NodW0szFh1IYG1EKmM7+dCvpRutPe1YcTQJGwtTRrTzolytZcHeOEOriKqEzL3dA1gbkcq8tWcoLFPTNdCZ5NwS5m+OxtLMhDK11jBWCCHqY0SII5/tTONAfCG9g/R/s9adyUMFDG+tT6i2cLNi3ZRW1f7O5xSrsa18QFdcrr3mJPD28wVotDCkhQN5pcYP4oa2dGBnTAE7YwoumwTOLam5cvViDlam9b5m/XEkk+/3Z+DraM6kjh44W5tyPKWE5RHZTF8dz7yRfthYmJJboub9rckMCravV0La1sKEt2/zpVytY9WpHOZtSyE+t5wnetW87sK5rFJm/BtPfpmG5/t50sq9+gyYurjYmHFnW2f+OZHDG+sSebirG3YWJmw8m8+GqDzMTKjx2qLR6Zg1xAcTFWw8m8+PBzI4l1XK27fpe+G3crPioTBXbmvpSMBF/60GBjvw8OLzfLYzlT5Bdtf0vUEIIcSNIUlgIYQQoh6GtfXkyy3n2HAyjT4t3FBrtGyOTKdzgBM+F/XNtTAzYdOpdA7EZpOYU0JKbgk5xRUXJTQvk6W8QgnZ+vYEb62qfUZDXW0PRs3fXef7XE1P4Iv1rkyW92vlThtvB1755wS/7o5l5ug22Fjov4qUVtRcjQRgZ2ncYqG0QmO02FrVtkvHltVS7avfv/7JjKpq4w6+joYEcJVxXXxZdCCBgzHZjO3kg6mJio/u6cjsf0/xx754/tgXD0D3Zs68PLw1M1ecxMFa/xDAy9GKzyZ24p1/TzFv7RlAf2M+tpMPjtbm/Lw71jBWCCHqY1hLR77ek87Gs/n0DrJHrdGx9Vw+nXxs8Lmob665qQmbo/M4mFBEUl45KQUV5JRoDNcs7XW4ZiXm6dsIvL0pudYxqQUVtb4GMObXuhedrm9P4KJyDb8dzsTN1ozvxzczJLv7N3egtbsVczYn8/uRLKb19OD9rSlodPrq2EsT0mqtjtwSNRZmJtiYG89YcbczZ1BlD+ChLR14cnksi49nMSbUqVpF8IH4Qt7amERRuZYne3lwVzuXKz6XSz3V2xMVsPxkDnviCgF9P+B5o/x5bmUcDjVcC20tTBlWmeAe0tKRmesS2XqugDuTiujsa0s7L5tq7UUAvOzN6dfMjg1R+cRml9Hctf6JayGEEDeWJIGFEEKIenCwMqdfKzd2ns2kqEzNodgc8koqGN3hwnTRojI1zy46xpnUAjr5O9HW24HRHbxp42XPXwcTWH8y7areW3PJTbi28sdXhrfG26nmm62qXrm1+WxS3b2JXeyurSfwxfq2dMPW0pQzaQUAeDvq467qj3uxqh7AVb1wfRytK8eWV0sCZxSUoeLC+fo4WpFZWHOVc3pBWY09iOtStZhdTb8Pt8qF/orLLySeXWwt+HRiJ1JyS0grKMPLwQovRyv+C08BwO+ixHqojwOLHuvB+YwiisrVBLna4mBtzux/IzE1UeFVw6JxQghRF3tLU/o2s2dXbAHF5RoOJRaRV6phVBsnw5iicg3TV8VzJqOUjj42tPG0ZmQbJ0LcrVkSnsWGqPovpAmg0dZ8zXppgFetSVq3yyxAB/DJHXX3Jr7cInY1Scgtp0ytY2SIfbVq56EtHfh4ewoHE4qY1hNDIvX+ReerHScitYQxv55leGtHXh9ce+9iUxMVg1s4EJleytnMUqMk8JrTuXy4XX+NeHWQNyNDnOp1LpcyM1HxbF8vHunqTmxOGXYWJjRzsSSloAK1ljpbUgAMa+XAjpgCzmSUGi3wVxOXyn7MxTU82BVCCKE8SQILIYQQ9TSqgzebI9PZHZ3F9qgMbC1NGdD6Qr++pYcSOZ1awEu3t+LOzsa9nbOK6m6/YGqiqrFnbdYlSU2fysSvvZUZ3YKMK4VS80o5k1qAtfPlK14v3e96KCpTM/W3wwS42DBvQnuj19QaLeVqLZZm+iqpEG97VEBkSgF3djY+zqnKfrhtffR9utt466tvT6fkE+hqXIUUmVJAoKuNYWG2EG8HVh5LJj2/FI+LEqipeaXkFlcwqHX9+ys2d7fFytyEmMyiaq8l5eoXR6xKxmcUlLH/fDYd/B0JcLExqqTedz4bCzMTQivPKyq1gJMp+Yxs50Wwx4XFfzRaHQdjs2nn44CFmay6LoS4OiNDHNkSrV/kbEdMAbYWJvS/qGfr3ydyOJ1Ryoz+Xoxt62y0b3Zx3f3TTVVQUkNv2axLFh3zdtBXwtpbmlZbPCytoIIzGSVY15GUvBGLjpmb6uudtdqaq511XHgIW1sS+oXV8QS7WvJUb09DInvh0SwWH89i7nB/2noZz6YprlyszfKiv+3/RubywbYUrM1NmH2bL90D6r8Y3MXUGh2bo/PxsDOjs68tHbwvXDcPxOuvY5189Ns2ROXx3b50nuztyZAWxmtjXIhV/3t6Z2MSJ9NK+OWeZthc0s8/Nkf/QNa3HpXYQgghbh65oxBCCCHqqVuQM54OlqyLSGXvuSyGhHgYWgUA5Fb25w12N76Bi0jK41h8LlC9qvdibnYW5BSXGyphAfJLK9gdnWk0rn8rd0xU8Pu+eMrUF27UdTodn2yM4vXlEcRXtoy4mWwtzbA2N2Hv+SzOpBYYvbbwQAIVGh39W+mTsG52lnQOcGLL6XRDr1yAlNwStp3JoGdzF5wrq2x7Bbtia2nK34cTDf15AbaeTiclr5SR7S8saHRbqH5B1b8OJhi9/5/79W0ZRrSv/4J4lmamDA7x4HxGEdvOZBgft7LdQ9XCbhqtjvfWnuaPvXFG444l5LI9KoM7O/kY2ltEZxTy0foow+J5F8eaVVjOvd396x2rEEJU6epni4edGeuj8tgXX8jgYAesLmpXkFfZ1qC5q/EMiYjUYo4l65OFmst0g3C1NSO3RENG4YVWDgVlGvZWVs1W6d/MHhMV/Hk0y2hBMp1Ox6c7U5m5Pon43OqzQm60Zi6WeNmbs+18AZlFxu0o1kTmUqrW0d1fn3zu6mdb4//gQnI7yEX/ewx0tiCnRMOiY1lGx8wtUbM6MhcHS1M6ViZhI1KL+XhHCjbmJnx6R8A1J4ABzExV/HQwg493pFJx0X/AzKIKFh7LoqWbJV189e8f7GpJZpGaJcezUF+UDC9Ta1kano2ZCfQK1D84cLM1I6WggmUROUbvdyy5iP3xRfQMtMO5ntXYQgghbg756yyEEELUk4lKxfB2Xvy2R5/gG9XBeOXwfi3c+PtQIu+sPsW4Lr7YWZoRmZLPuog0TE1UqLU6Cstqr64a0d6L44l5PL/4GHd19qVUrWXl0WQcrM3JKb5wgxrgYsPkPkH8tCuWyb8cYkQ7L2wtzdh2Jp3DcbkMC/Wge7PrX+l7JWbc3opnFx3j+b+OcVcXX9zsLDkcl8PWMxl08HNkYjc/w9hnBrfg8T+O8PgfR7inqz86nY4lhxIxNVHx1KAWhnG2lmY8OTCYD9dH8cyiYwxv50VKXgmLDybS0sOOu7pcOGanACeGtvFg8cFE8oor6BTgxJH4XDacTOOOjt609bmwoE92UTkHY7JxtrWo8/f15MBgjsXn8n+rTjK2kw9Bbrbsjc5i97ksRrb3IixQX0Xn5WjFiHZe/HciFR3Q3s+RpJwSlh5KpLmbLY/2DTIcc1BrD/7cH8/HG6OIyy7Cx8ma4/G5rDuZxsj2XvRrVf+qZSGEqGKiUjG8tSMLDuuTkSNCjBc06xNkzz8ncpizKZk72zljZ2HC6fRS1kflGa5ZReW1X7OGt3YiPKWEGf/Gc2c7Z8oq9Iuf2VuaklNyYT9/J0seDnPjl0OZTFkaw/DWjthamLD9fAFHkooZ2sKBbv7XnvysLxOVipcGePHqmkSm/RPLHaFOuNiYcTK1hPVReQQ6W/BAF7d6H7dPkD0Dmtuz/XwBL/0bT58ge3JL1aw8mUtuqZq3h/kaegd/uTsNjRZ6NLMlMa+cxLzqs4aqFqLLLlZzKLEIZ2vTOn9fk7u68d7WFF5YHc/Qlg4UlWtZcTKHwjINc273Q1W5eFuwqxX3dHRh8fFsnlkRx9CWDpSqtayJzCMhr5zn+nriZa+v5L6/iys7Ywr48UAGCbnlhHpaE5tTxqqTubjZmvFCv/o/ZBVCCHFzSBJYCCGEuAqjOnizYE8c/i42tPM1vqEOC3Lm7bFt+WNfHD/visXcVIWXoxWP9W9GkKsNL/19ggMx2YR42dd47NEdvCkqU7PiaDLzN0fj4WDJ2E4++DpZM3PFSaOxj/ZtRjM3W5YeSuS3vXGgA19na54f2pJxXWrvSXijtfVx5PsHw/hxVwz/HE6iVK3Bx8max/o1474eAUbtDVp52fPV/Z35bvt5ftwZg7mpina+jjw+oDnN3Y2n/t7Z2RcbCzP+3B/PZxvP4mhjzsj2XjzWr5mhsrbKzNFt8HexYe2JVLaczsDTwZInBzZn4iWVtbGZRbzzbySd/Z3qTAI721rw/cNh/Lwrlm1nMsg/loK3kxXPDG5hlNgGeHl4a/ycrVl3MpXNkem42VkyIcyPB3sFYG91YaE3awtTPp/UiR92xLA+Io380gr8nG148bZWjO2s3H9DIUTTMTLEid8PZ+HvZFFtUa8wP1veGubLn0cz+fVgBuamJnjamzGluztBzha8siaRgwlFtHaveYHQUSGOFJVrWHkyly93p+FhZ84doU74Oljw5oYko7GTu7nTzMWSv09k8/uRLNDp8HW04Nk+ntzZzrnG498M3fzt+PquQBYczuTv8ByKKzS425pzdwcXHg5zw+4qFhMF+L9hviwJz+a/yFzm707F2tyE9l42PNzVjTYe+t9ncYWWU+n6mTBbzxWw9VxBjceqSgLH5ZQxZ3MynXxs6kwCjwhxwsLMhMXHsvh6bzo25iZ09rHhka5uBDgbV34/1duTIGdLlkVk8/WedMxNVbTxsOK5fp50v+h9HK3M+OauIH4+mMnu2AI2nM3D2dqM4a0dmdzNDTdbWchUCCEaKpXueixPfrk3UKnIWTH7Rr6FEDeV852zuMEfm6umUqlI//ZBpcMQ4rI8Hv+9QX+GMn6epnQYjUpKbgkTvt3HiHZezBzdRulwrtr2MxmsPJbMJxPrXihPaT/tjOHn3bF8cW8nugRePmni/uh3DfbzJoS4NiqVitRPxyodRqOSkl/OxD/P1bl4W0O343w+q07l8tHouhfKU9rPBzP49VAmn48JqHNhOa/pK+WaJYQQN9ANrwQO9PfF+c5ZN/pthLhpAv196x6kkEA/Hzwe/13pMIS4rEC/xnvTJZqm4nI1y44k0SXQSelQhBBCiMsqrtCyPCKnzoSqEEIIcakbngSOjU+80W8hhKgUm5BU9yAhhLgBknNLWB+RiqejFZ38nZQOp17K1Fq6NXPm3u4Nu6LqTGoBsZlFRGcU1j1YCCFErZLzy9kQlYennblhcbbGolytpau/LZM6uiodymWdySghLqecc1k3f7E/IYQQNZOewEIIIYS4ZscT8ziemMfA1u6NLgnsbGPBAz0DlQ6jThtPpbHoQILSYQghRKMXnlJCeEoJA5rbN7oksJO1Gfd3rv9CdTfbprP5LD6erXQYQgghLnLDewILIYQQjYX0BBZNjfQEFqLpkp7AoqmRnsBCCHFjmdQ9RAghhBBCCCGEEEIIIURjJUlgIYQQQgghhBBCCCGEaMIkCSyEEEIIIYQQQgghhBBNmCSBhRBCCCGEEEIIIYQQogmTJLAQQgghhBBCCCGEEEI0YZIEFkIIIYQQQgghhBBCiCZMpdPpdEoHIYQQQjQEQf6+xCUmKx2GENdNoJ8PsQlJSochhLgB5Jolmhq5ZgkhxI0lSWAhhBDiFnXmzBkeeeQR/vvvP1xcXJQO54b66KOPyMjIYN68eUqHIoQQ4ioUFxczcuRIPvroI7p27ap0ODfUli1b+OCDD1i1ahUWFhZKhyOEEKKJkHYQQgghxC1Ip9MxZ84cnn766SafAAZ44okn2Lt3L4cPH1Y6FCGEEFfhu+++IywsrMkngAEGDRqEv78/CxYsUDoUIYQQTYgkgYUQQohb0Jo1aygoKGDSpElKh3JT2Nra8vLLLzN79mw0Go3S4QghhKiHuLg4/vrrL15++WWlQ7kpVCoVb7zxBj/88ANpaWlKhyOEEKKJkCSwEEIIcYspKirigw8+YNasWZiamiodzk0zatQo7OzsWLJkidKhCCGEqIe5c+fy2GOP4enpqXQoN01QUBATJ07kww8/VDoUIYQQTYQkgYUQQohbzLfffkv37t0JCwtTOpSbSqVSMXPmTObPn09OTo7S4QghhLgCW7duJS4ujoceekjpUG66adOmcfDgQQ4dOqR0KEIIIZoASQILIYQQt5CYmBiWLFnCSy+9pHQoiggJCWHUqFF89tlnSocihBCiDmVlZcydO5eZM2fekguk2dra8sorr/DOO++gVquVDkcIIUQjJ0lgIYQQ4hah0+l49913+d///oeHh4fS4SjmmWeeYdOmTZw8eVLpUIQQQlzGzz//TKtWrejbt6/SoShmxIgRODo6snjxYqVDEUII0chJElgIIYS4RWzZsoWkpCQefPBBpUNRlKOjI88//zyzZ89Gq9UqHY4QQogaJCcn8+uvv/Lqq68qHYqiqloZffnll2RnZysdjhBCiEZMksBCCCHELaCsrIz33nvvlp1Se6nx48ejVqtZtWqV0qEIIYSowbx583jwwQfx9/dXOhTFtW7dmtGjR/Ppp58qHYoQQohGTJLAQgghxC3gxx9/pE2bNvTp00fpUBoEExMT3nzzTT766CMKCgqUDkcIIcRF9u7dy4kTJ5g6darSoTQYzzzzDFu3buXEiRNKhyKEEKKRkiSwEEII0cQlJSWxYMGCW35K7aU6dOjAgAED+Oqrr5QORQghRKWKigrmzJnD66+/jpWVldLhNBgODg5Mnz5dWhkJIYS4apIEFkIIIZq4qim1vr6+SofS4LzwwgusWLGCs2fPKh2KEEII4M8//8TLy4shQ4YoHUqDM27cOACWL1+ucCRCCCEaI0kCCyGEEE3Ynj17OHnypEyprYWrqytPPfUUc+bMQafTKR2OEELc0jIyMvjmm2944403UKlUSofT4JiYmDBz5kw+/fRT8vPzlQ5HCCFEIyNJYCGEEKKJKi8vZ/bs2TKltg733nsv2dnZrF+/XulQhBDilvbxxx8zYcIEmjdvrnQoDVaHDh0YNGgQX3zxhdKhCCGEaGQkCSyEEEI0UX/88Qe+vr4MHjxY6VAaNDMzM2bNmsW8efMoLi5WOhwhhLglHTlyhD179vDEE08oHUqDN336dFavXk1UVJTSoQghhGhEJAkshBBCNEHp6el89913MqX2CnXv3p0uXbrw/fffKx2KEELccjQaDXPmzOGll17Czs5O6XAaPBcXF5555hlmz54trYyEEEJcMUkCCyGEEE3QRx99xN13302zZs2UDqXRePnll1m0aBFxcXFKhyKEELeUpUuXYm1tzejRo5UOpdGYOHEieXl5rF27VulQhBBCNBKSBBZCCCGamMOHD7Nv3z6ZUltPnp6eTJkyhffee0/pUIQQ4paRk5PD/PnzmTVrlsxcqQczMzPefPNN5s2bR1FRkdLhCCGEaAQkCSyEEEI0IRqNhtmzZ/PKK69ga2urdDiNziOPPEJMTAzbtm1TOhQhhLglfP7554wYMYKQkBClQ2l0unbtSrdu3fjuu++UDkUIIUQjIElgIYQQoglZvHgx9vb2jBw5UulQGiULCwveeOMN5s6dS3l5udLhCCFEk3bq1Ck2btzIs88+q3QojdZLL73E4sWLiY2NVToUIYQQDZwkgYUQQogmIicnhy+//FIWg7tG/fv3p0WLFvzyyy9KhyKEEE2WTqfjnXfe4fnnn8fR0VHpcBotT09PHnvsMd59911ZJE4IIcRlSRJYCCGEaCI+/fRTRo4cKVNqr4PXXnuNn3/+mZSUFKVDEUKIJmnlypVUVFQwfvx4pUNp9B566CESEhLYunWr0qEIIYRowCQJLIQQQjQBJ0+eZPPmzTzzzDNKh9Ik+Pv7c//99zNv3jylQxFCiCansLCQjz/+mDfffBMTE7klvVYWFhbMnDmTuXPnUlZWpnQ4QgghGii54gohhBCNnFarZfbs2UyfPl2m1F5Hjz32GOHh4ezbt0/pUIQQokn56quv6NevHx07dlQ6lCajb9++hISE8NNPPykdihBCiAZKksBCCCFEI7dy5Uo0Gg133XWX0qE0KdbW1rz66qvMmTOHiooKpcMRQogm4dy5cyxfvpwZM2YoHUqT8+qrr/Lbb7+RlJSkdChCCCEaIEkCCyGEEI1YQUEBH3/8MbNmzZIptTfAsGHD8PDwYOHChUqHIoQQjZ5Op2POnDk8+eSTuLq6Kh1Ok+Pn58eDDz4orYyEEELUSO4WhRBCiEbsyy+/ZODAgXTo0EHpUJoklUrFG2+8wTfffENmZqbS4QghRKO2YcMGMjMzue+++5QOpcmaOnUqERER7N27V+lQhBBCNDCSBBZCCCEaqbNnz7Jq1SpeeOEFpUNp0oKDgxk3bhwff/yx0qEIIUSjVVJSwvvvv8+sWbMwMzNTOpwmy8rKitdff53Zs2dLKyMhhBBGJAkshBBCNEJVU2qfeuopXFxclA6nyXvqqafYtWsXx44dUzoUIYRolL7//ns6d+5M9+7dlQ6lyRsyZAje3t788ccfSocihBCiAZEksBBCCNEIrVu3juzsbCZNmqR0KLcEOzs7XnzxRd555x00Go3S4QghRKOSkJDAwoULefnll5UO5ZZQ1cro22+/JT09XelwhBBCNBCSBBZCCCEameLiYubNm8ebb74pU2pvojFjxmBlZcU///yjdChCCNGozJ07lylTpuDl5aV0KLeM5s2bM2HCBGllJIQQwkCSwEIIIUQj89133xEWFka3bt2UDuWWolKpmDVrFp999hm5ublKhyOEEI3C9u3bOX/+PI888ojSodxynnjiCfbu3cuRI0eUDkUIIUQDIElgIYQQohGJi4vjr7/+kim1CmnTpg233347n3/+udKhCCFEg1deXs67777L66+/joWFhdLh3HLs7Ox46aWXmD17trQyEkIIIUlgIYQQojF57733mDp1Kp6enkqHcst67rnnWL9+PZGRkUqHIoQQDdqvv/5KcHAwAwYMUDqUW9bo0aOxsbFhyZIlSocihBBCYZIEFkIIIRqJbdu2ERMTw8MPP6x0KLc0JycnnnvuOWbPno1Op1M6HCGEaJBSU1P56aefeP3115UO5ZZW1cpo/vz55OTkKB2OEEIIBUkSWAghhGgEysrKePfdd5k5c6ZMqW0AJkyYQGlpKatXr1Y6FCGEaJA++OAD7rvvPvz9/ZUO5ZYXEhLCyJEj+eyzz5QORQghhIIkCSyEEEI0Ar/88gutWrWiX79+SociAFNTU958800+/PBDCgsLlQ5HCCEalP3793Ps2DH+97//KR2KqPTss8+yadMmTp48qXQoQgghFCJJYCGEEKKBS0lJ4ZdffuHVV19VOhRxkU6dOtG3b1+++uorpUMRQogGQ61WM2fOHF555RWsra2VDkdUcnR05Pnnn2f27NlotVqlwxFCCKEASQILIYQQDdy8efN44IEHZEptAzRjxgyWL1/OuXPnlA5FCCEahIULF+Lm5sZtt92mdCjiEuPHj0etVrNq1SqlQxFCCKEASQILIYQQDdjevXsJDw/nscceUzoUUQM3NzeeeOIJ3n33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hBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6MckBBZCCCGEEEIIIYQQQoh+TEJgIYQQQgghhBBCCCGE6Mf+H0BoCtTWoMpSAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1800x1440 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# визуализация на схеме НА ПРИМЕРЕ МАЛЕНЬКОГО ДЕРЕВА\n",
    "nodes_num = len([i for i in node_counts if i < 20])\n",
    "print('Количество узлов:', nodes_num,\n",
    "      '\\nТочность дерева на тестовой:', \n",
    "      np.around(test_scores[node_counts.index(nodes_num)], 3))\n",
    "\n",
    "fig = plt.figure(figsize=(25,20))\n",
    "_ = plot_tree(clfs[node_counts.index(nodes_num)], \n",
    "              filled=True, \n",
    "              feature_names=list(X.columns))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f46adc5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "|--- PAY_0 <= 1.50\n",
      "|   |--- PAY_2 <= 1.50\n",
      "|   |   |--- BILL_AMT1 <= 704.50\n",
      "|   |   |   |--- BILL_AMT5 <= 957.50\n",
      "|   |   |   |   |--- class: 0\n",
      "|   |   |   |--- BILL_AMT5 >  957.50\n",
      "|   |   |   |   |--- class: 0\n",
      "|   |   |--- BILL_AMT1 >  704.50\n",
      "|   |   |   |--- LIMIT_BAL <= 75000.00\n",
      "|   |   |   |   |--- PAY_4 <= 1.00\n",
      "|   |   |   |   |   |--- class: 0\n",
      "|   |   |   |   |--- PAY_4 >  1.00\n",
      "|   |   |   |   |   |--- class: 0\n",
      "|   |   |   |--- LIMIT_BAL >  75000.00\n",
      "|   |   |   |   |--- PAY_4 <= 1.50\n",
      "|   |   |   |   |   |--- class: 0\n",
      "|   |   |   |   |--- PAY_4 >  1.50\n",
      "|   |   |   |   |   |--- class: 0\n",
      "|   |--- PAY_2 >  1.50\n",
      "|   |   |--- class: 0\n",
      "|--- PAY_0 >  1.50\n",
      "|   |--- PAY_6 <= 1.00\n",
      "|   |   |--- BILL_AMT1 <= 779.00\n",
      "|   |   |   |--- class: 0\n",
      "|   |   |--- BILL_AMT1 >  779.00\n",
      "|   |   |   |--- class: 1\n",
      "|   |--- PAY_6 >  1.00\n",
      "|   |   |--- class: 1\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# визуализируем дерево в виде текстовой схемы\n",
    "viz = export_text(clfs[node_counts.index(nodes_num)], \n",
    "                  feature_names=list(X.columns))\n",
    "print(viz)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "061a0c22",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "# Бэггинг  \n",
    "\n",
    "Модель бэггинга использует бутстреп, чтобы вырастить $B$ деревьев на выборках с повторами из обучающих данных. Построим модель для $B=40$ деревьев.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "c1e2e90c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Обучение модели с бэггингом на 40 деревьях и перекрёстной проверкой заняло 57.58 секунд\n"
     ]
    }
   ],
   "source": [
    "# параметр B: количество деревьев\n",
    "num_trees = 40\n",
    "\n",
    "# разбиения для перекрёстной проверки\n",
    "kfold = KFold(n_splits=5, random_state=my_seed, shuffle=True)\n",
    "\n",
    "# таймер\n",
    "tic = time.perf_counter()\n",
    "# модель с бэггингом\n",
    "tree_bag = BaggingClassifier(base_estimator=cls_one_tree,\n",
    "                             n_estimators=num_trees,\n",
    "                             random_state=my_seed)\n",
    "\n",
    "cv = cross_val_score(tree_bag, X, y, cv=kfold)\n",
    "\n",
    "# таймер\n",
    "toc = time.perf_counter()\n",
    "print(f\"Обучение модели с бэггингом на {num_trees:0.0f} деревьях\", \n",
    "      \" и перекрёстной проверкой \", \n",
    "      f\"заняло {toc - tic:0.2f} секунд\", sep='')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "0221cd4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.811"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# точность\n",
    "np.around(np.mean(cv), 3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba7ecbf3",
   "metadata": {},
   "source": [
    "Итак, мы построили модель, выбрав параметр $B$ случайным образом. Воспользуемся функцией `GridSearchCV()`, чтобы перебрать 8 вариантов значений для параметра $B$.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "0895547b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Сеточный поиск занял 240.27 секунд\n"
     ]
    }
   ],
   "source": [
    "# настроим параметры бэггинга с помощью сеточного поиска\n",
    "param_grid = {'n_estimators' : [5, 10, 15, 25, 30, 40, 50, 60]}\n",
    "\n",
    "# таймер\n",
    "tic = time.perf_counter()\n",
    "clf = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()),\n",
    "                  param_grid, scoring='accuracy', cv=kfold)\n",
    "\n",
    "tree_bag = clf.fit(X, y)\n",
    "# таймер\n",
    "toc = time.perf_counter()\n",
    "print(f\"Сеточный поиск занял {toc - tic:0.2f} секунд\", sep='')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "0d14e6f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.812"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# точность лучшей модели\n",
    "np.around(tree_bag.best_score_, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b6bc0aeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_estimators': 40}"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# количество деревьев у лучшей модели\n",
    "tree_bag.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "81d046b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'base_estimator__ccp_alpha': 0.0,\n",
       " 'base_estimator__class_weight': None,\n",
       " 'base_estimator__criterion': 'gini',\n",
       " 'base_estimator__max_depth': None,\n",
       " 'base_estimator__max_features': None,\n",
       " 'base_estimator__max_leaf_nodes': None,\n",
       " 'base_estimator__min_impurity_decrease': 0.0,\n",
       " 'base_estimator__min_samples_leaf': 1,\n",
       " 'base_estimator__min_samples_split': 2,\n",
       " 'base_estimator__min_weight_fraction_leaf': 0.0,\n",
       " 'base_estimator__random_state': None,\n",
       " 'base_estimator__splitter': 'best',\n",
       " 'base_estimator': DecisionTreeClassifier(),\n",
       " 'bootstrap': True,\n",
       " 'bootstrap_features': False,\n",
       " 'max_features': 1.0,\n",
       " 'max_samples': 1.0,\n",
       " 'n_estimators': 40,\n",
       " 'n_jobs': None,\n",
       " 'oob_score': False,\n",
       " 'random_state': None,\n",
       " 'verbose': 0,\n",
       " 'warm_start': False}"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_bag.best_estimator_.get_params()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f2b835b",
   "metadata": {},
   "source": [
    "Таким образом, перебрав несколько вариантов для $B$, мы немного улучшили первоначальную точность модели бэггинга.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1c66b71f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acc с перекрёстной проверкой \n",
      "для модели bagging_GS : 0.812\n"
     ]
    }
   ],
   "source": [
    "# записываем точность\n",
    "score.append(np.around(tree_bag.best_score_, 3))\n",
    "score_models.append('bagging_GS')\n",
    "\n",
    "print('Acc с перекрёстной проверкой',\n",
    "      '\\nдля модели', score_models[2], ':', score[2])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "041010d5",
   "metadata": {},
   "source": [
    "\n",
    "# Прогноз на отложенные наблюдения по лучшей модели\n",
    "\n",
    "Ещё раз посмотрим на точность построенных моделей.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "13f506f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Модель</th>\n",
       "      <th>Acc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>one_tree</td>\n",
       "      <td>0.731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>pruned_tree</td>\n",
       "      <td>0.731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>bagging_GS</td>\n",
       "      <td>0.812</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Модель    Acc\n",
       "0     one_tree  0.731\n",
       "1  pruned_tree  0.731\n",
       "2   bagging_GS  0.812"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# сводка по точности моделей\n",
    "pd.DataFrame({'Модель' : score_models, 'Acc' : score})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99c7c905",
   "metadata": {},
   "source": [
    "Все модели показывают среднюю точность по показателю $Acc$, при этом самой точной оказывается модель Бэггинга. Сделаем прогноз на отложенные наблюдения.   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "6f19fd52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.94      0.88      3454\n",
      "           1       0.66      0.37      0.48      1046\n",
      "\n",
      "    accuracy                           0.81      4500\n",
      "   macro avg       0.75      0.66      0.68      4500\n",
      "weighted avg       0.79      0.81      0.79      4500\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# данные для прогноза\n",
    "X_pred = DF_predict_num.drop(['Y'], axis=1)\n",
    "# строим прогноз\n",
    "y_hat = tree_bag.best_estimator_.predict(X_pred)\n",
    "# характеристики точности\n",
    "print(classification_report(DF_predict_num['Y'], y_hat))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "envformo",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
