# 
# Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy

# Формат выполнения: самостоятельная работа.

# Перед началом:
# 1. Создайте виртуальное окружение:
#    python -m venv numpy_env
   
# 2. Активируйте виртуальное окружение:
#    - Windows: numpy_env\Scripts\activate
#    - Linux/Mac: source numpy_env/bin/activate
   
# 3. Установите зависимости:
#    pip install numpy matplotlib seaborn pandas pytest

# Структура проекта:

# numpy_lab/
# ├── main.py
# ├── test.py
# ├── data/
# │   └── students_scores.csv
# └── plots/

# В папке data создайте файл students_scores.csv со следующим содержимым:

# math,physics,informatics
# 78,81,90
# 85,89,88
# 92,94,95
# 70,75,72
# 88,84,91
# 95,99,98
# 60,65,70
# 73,70,68
# 84,86,85
# 90,93,92

# (Дополнительно можно использовать публичные датасеты Kaggle:
# Students Performance Dataset:
# https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
# или любой аналогичный табличный CSV)

# Задача: реализовать все функции, чтобы проходили тесты.
# 

import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def func(pos_only, /, standard, *, kw_only):
#     pass
# pos_only: CẤM dùng tên (không được pos_only=1)
# standard: Dùng cách nào cũng được
# kw_only:  BẮT BUỘC dùng tên (không được truyền khơi khơi)

# print(np.array([[1,2,3],[4,5,6],[7,8,9]]))
# print(np.array([(1,2,3),(4,5,6),(7,8,9)]))
# print(np.array([(1,2,3),[4,5,6],(7,8,9)]))

# print(np.array((1,2,3),[4,5,6],(7,8,9))) # syntax bug

def save_figure(name):
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig(f"figures/{name}.png")


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector():
    """
    Создать массив от 0 до 9.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    # Подсказка: используйте np.arange(10)

    # print(np.arange(10))
    # [0 1 2 3 4 5 6 7 8 9]

    # print(np.shape(np.arange(10)))
    # (10,) # => 10 row - 1 column ??
    return np.arange(10)


def create_matrix():
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Изучить:
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    # Подсказка: используйте np.random.rand(5,5)

    # print(np.random.rand(5,5))
    # [[0.72084734 0.65486936 0.49755924 0.13184927 0.61126369]
    #  [0.25784305 0.60355404 0.4664948  0.07718884 0.98685796]
    #  [0.42711358 0.16519391 0.13742476 0.73401582 0.97867133]
    #  [0.39150981 0.34369703 0.06534119 0.17238172 0.9137438 ]
    #  [0.61326675 0.73571581 0.91765901 0.99605199 0.09929395]]

    # print(np.shape(np.random.rand(3,5)))
    # (3, 5)

    # print(np.empty([4, 3], dtype=int))
    #[[ 2171465 13486539 -3237113]
    # [25226685 -9595542 15493566]
    # [15493566 -9595542 25226685]
    # [-3237113 13486539  2171465]]
    # print(np.empty([4, 3], dtype=int).ctypes.data)

    # return np.random.rand(0,1) # False rand -> (0,1) || randint -> (0, num), so in rand() is size of array not data value
    return np.random.rand(5,5)

# def reshape_vector(vec: np._ObjectArrayT, *, size = (2,5)): # _ObjectArrayT private
def reshape_vector(vec: npt.NDArray, *, size = (2,5)):
    """
    Преобразовать (10,) -> (2,5)

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    
    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    # Подсказка: используйте vec.reshape(2,5)
    if size[0] * size[1] > np.size(vec):
        res = np.zeros(size)
        try:
            res[:] = vec
        except ValueError:
            res.flat[:np.size(vec)] = vec
        return res

        # vec.resize(*size)
        # return vec
    return vec.reshape(size)

# print(np.shape(reshape_vector(create_vector(), size=(3,4))))
# print(reshape_vector(create_vector(), size=(3,4)))

def transpose_matrix(mat: npt.NDArray):
    """
    Транспонирование матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    
    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    # Подсказка: используйте mat.T или np.transpose(mat)
    # return np.transpose(mat)
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: npt.NDArray, b: npt.NDArray):
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    # Подсказка: используйте оператор +
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape == b.shape:
            return a + b
        raise ValueError("Matrixes are not same size")

    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        raise ValueError("All args must be matrixes")
    
    return a + b

def scalar_multiply(vec, scalar=1):
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    # Подсказка: используйте оператор *
    if not isinstance(scalar, np.ScalarType):
        raise ValueError('Scalar must be a number')
    
    elif not isinstance(vec, np.ndarray):
        raise ValueError('Vector must be a type of NDArray')

    return vec * scalar

# print(scalar_multiply(np.array([1,2,3])))

def elementwise_multiply(a, b):
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    # Подсказка: используйте оператор *
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')

    return a*b


def dot_product(a, b):
    """
    Скалярное произведение.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    """

    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')
    # Подсказка: используйте np.dot(a, b)
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a, b):
    """
    Умножение матриц.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    
    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    # # Подсказка: используйте a @ b или np.matmul(a, b)
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')
    return a @ b



def matrix_determinant(a):
    """
    Определитель матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    """
    # Подсказка: используйте np.linalg.det(a)
    if not isinstance(a, np.ndarray):
        raise ValueError('Vector must be instances of NDArray')
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    """
    # Подсказка: используйте np.linalg.inv(a)
    if not isinstance(a, np.ndarray):
        raise ValueError('Vector must be instances of NDArray')
    return np.linalg.inv(a)
    


def solve_linear_system(a, b):
    """
    Решить систему Ax = b

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    
    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    """
    # Подсказка: используйте np.linalg.solve(a, b)
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    # Подсказка: используйте pd.read_csv(path).to_numpy()

    # don't count first line (headers)
    # print(pd.read_csv(path).to_numpy())

    return pd.read_csv(path).to_numpy()

def statistical_analysis(data):
    """
    Представьте, что данные — это результаты экзамена по математике. [3,2,4,3,4,5,1,2,3,4,5,2,4,5]
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    https://numpy.org/doc/stable/reference/generated/numpy.median.html
    https://numpy.org/doc/stable/reference/generated/numpy.std.html
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    """
    # Подсказка: используйте np.mean(), np.median(), np.std(), 
    # np.min(), np.max(), np.percentile(data, 25), np.percentile(data, 75)
    if not isinstance(data, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')

    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "percentiles": {
            "25": np.percentile(data, 25),
            "75": np.percentile(data, 75)
        }

    }


def normalize_data(data):
    """
    Min-Max нормализация.
    
    Формула: (x - min) / (max - min)
    
    Args:
        data (numpy.ndarray): Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    # Подсказка: вычислите min и max с помощью np.min() и np.max()
    if not isinstance(data, np.ndarray):
        raise ValueError('All vectors must be instances of NDArray')
    
    max = np.max(data)
    min = np.min(data)

    data = data.astype(dtype=float) #[0. 0.16666667 0.33333333 0.5  0.66666667 0.83333333 1.  0.5  0.66666667 0.83333333 0.5]

    for i in range(len(data)):
        data[i] = float((data[i]-min)/(max-min))
    
    return data

# print(normalize_data(np.array([1,2,3,4,5,6,7,4,5,6,4])))


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data):
    """
    Построить гистограмму распределения оценок по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    # Подсказка: используйте plt.hist(), добавьте заголовок, подписи осей,
    # сохраните в папку plots с помощью plt.savefig()
    if not isinstance(data, np.ndarray):
        raise ValueError('Vector must be instance of NDArray')
    # plt.hist(data, align="mid", label=["band", "frequence"], range=(0,10), rwidth=0.8)
    plt.hist(data, align="mid", label=["band", "frequence"], range=(0,10), edgecolor='black')
    save_figure("plot_histogram")
    plt.show()

# plot_histogram(np.array([1,2,3,4,5,6,7,4,5,6,4]))

def plot_heatmap(matrix):
    """
    Построить тепловую карту корреляции предметов.

    Изучить:
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    # Подсказка: используйте sns.heatmap(), добавьте заголовок, сохраните
    if not isinstance(matrix, np.ndarray):
        raise ValueError('Vector must be instance of NDArray')
    # sns.heatmap(matrix, xticklabels="x", yticklabels="y", cmap="coolwarm")
    ax = sns.heatmap(matrix, cmap="coolwarm")
    plt.xlabel("x", labelpad=5, loc='right')
    # plt.yticks(rotation=0)
    plt.ylabel("y", labelpad=10, loc='top')
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    # plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    # ax.invert_yaxis()
    save_figure("plot_heatmap")
    plt.show()

# plot_heatmap(np.random.rand(10,10))

def plot_line(x, y):
    """
    Построить график зависимости: студент -> оценка по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    
    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    # Подсказка: используйте plt.plot(), добавьте заголовок, подписи осей,
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError('Vector must be instance of NDArray')
    # сохраните график
    # print(list(zip(x,y)))
    # plt.plot(list(zip(x,y))) #==> 1x -> 2y: x=0 => y=2 and y=3
    plt.plot(x,y)
    plt.xlabel("isu number")
    plt.ylabel("band")
    save_figure("plot_line")
    plt.show()


# plot_line(np.array([2,5,6,7,8,9,13]), np.array([3,2,4,6,7,8,4]))


# ============================================================
# ========================== ТЕСТЫ ===========================
# ============================================================

def test_create_vector():
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    assert np.array_equal(
        vector_add(np.array([1,2,3]), np.array([4,5,6])),
        np.array([5,7,9])
    )
    assert np.array_equal(
        vector_add(np.array([0,1]), np.array([1,1])),
        np.array([1,2])
    )


def test_scalar_multiply():
    assert np.array_equal(
        scalar_multiply(np.array([1,2,3]), 2),
        np.array([2,4,6])
    )


def test_elementwise_multiply():
    assert np.array_equal(
        elementwise_multiply(np.array([1,2,3]), np.array([4,5,6])),
        np.array([4,10,18])
    )


def test_dot_product():
    assert dot_product(np.array([1,2,3]), np.array([4,5,6])) == 32
    assert dot_product(np.array([2,0]), np.array([3,5])) == 6


def test_matrix_multiply():
    A = np.array([[1,2],[3,4]])
    B = np.array([[2,0],[1,2]])
    assert np.array_equal(matrix_multiply(A,B), A @ B)


def test_matrix_determinant():
    A = np.array([[1,2],[3,4]])
    assert round(matrix_determinant(A),5) == -2.0


def test_matrix_inverse():
    A = np.array([[1,2],[3,4]])
    invA = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system():
    A = np.array([[2,1],[1,3]])
    b = np.array([1,2])
    x = solve_linear_system(A,b)
    assert np.allclose(A @ x, b)


def test_load_dataset():
    # Для теста создадим временный файл
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78,81,90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([10,20,30])
    result = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    data = np.array([0,5,10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0,0.5,1]))


def test_plot_histogram():
    # Просто проверяем, что функция не падает
    data = np.array([1,2,3,4,5])
    plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1,0.5],[0.5,1]])
    plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    plot_line(x, y)



