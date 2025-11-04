 **Простой вектор:**
```python
np.array([1,2,3])
```
**Нулевой вектор размерности n:**
```python
np.zeros(5)
---
-> array([0., 0., 0., 0., 0.])
```
**Единичная матрица указанной размерности:**
```python
np.eye(5)
---
-> array([[1., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0.],
          [0., 0., 1., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 1.]])
```
**Вектор из последовательных значений с заданным шагом:**
```python
np.arange(0, 9, 1.5)
---
-> array([0. , 1.5, 3. , 4.5, 6. , 7.5])
```

**Прибавление скаляра к вектору:**
```python
np.array([1,2,3]) + 2 
--- 
-> array([3, 4, 5])
```
**Умножение вектора на скаляр:**
```python
np.array([1,2,3]) * 2 
--- 
-> array([2, 4, 6])
```
**Скалярное умножение двух векторов:**
```python
a = np.array([1,2]) 
b = np.array([3,4]) 
np.sum(a * b) 
np.dot(a,b)
a @ b
--- 
-> 11 
-> 11 
-> 11
```
**Скалярное умножение двух матриц:**
```python
k = np.array([
    [1,-4,2],
    [-6,0,10]
])

r = np.array([
    [0,-5],
    [-2,8],
    [1,4]
])

k @ r
---
-> array([
		  [ 10, -29],
	      [ 10,  70]
		])
```
**Склейка двух массивов:**
```python
a = np.array([1,2,3])
b = np.array([4,5,6])

np.stack([a,b],axis=0)
---
-> array([[1, 2, 3],
	      [4, 5, 6]])
---
np.stack([a,b],axis=1)
---
-> array([[1, 4],
	      [2, 5],
	      [3, 6]])
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706632.png)<br>
**Число Пи:**
```python
np.pi 
--- 
-> 3.141592653589793
```
**Перевод из градусов в радианы:**
```python
np.rad2deg(np.pi) 
--- 
-> 180.0
```
**Перевод из радианов в градусы:**
```python
np.deg2rad(180) 
--- 
-> 3.141592653589793
```
**Тригонометрические функции для углов в радианах:**
```python
np.sin(np.deg2rad(30)) 
np.cos(np.deg2rad(30)) 
np.tan(np.deg2rad(30)) 
--- 
-> 0.49999999999999994 
-> 0.8660254037844387 
-> 0.5773502691896257
```
**Обратные тригонометрические функции:**
```python
np.rad2deg(np.arcsin(0.49999999999999994)) np.rad2deg(np.arccos(0.8660254037844387)) 
--- 
-> 29.999999999999996 
-> 29.999999999999996
```
**Вычисление L1 нормы (сумма модулей координат):**
```python
a = np.array([3,-2,8]) 
np.linalg.norm(a,1) 
--- 
-> 13
```
**Вычисление L2 нормы (корень из суммы квадратов координат):**
```python
a = np.array([3,-2,8]) 
np.linalg.norm(a,2) 
np.linalg.norm(a) 
--- 
-> 8.774964387392123 
-> 8.774964387392123
```
**Вычисление максимальной нормы (максимальная по модулю координата):**
```python
a = np.array([3,-2,8]) 
np.linalg.norm(a,np.inf)
--- 
-> 8
```
**Косинус угла между двумя векторами:**
```python
a = np.array([1,2,3]) 
b = np.array([4,5,6]) 
cos_angle = a @ b / np.linalg.norm(a) / np.linalg.norm(b) 
--- 
-> 0.9746318461970762
```
**Вычисление L1 расстояния между векторами:**
```python
a = np.array([22, 4, -6, 3.9, 7]) 
b = np.array([12, -0.5, 6, 42, 1]) 
np.linalg.norm(a - b, 1) 
--- 
-> 70.6
```
**Вычисление L2 расстояния между векторами:**
```python
a = np.array([22, 4, -6, 3.9, 7]) 
b = np.array([12, -0.5, 6, 42, 1]) 
np.linalg.norm(a - b, 2) 
--- 
-> 41.85522667481327
```
**Вычисление максимального расстояния между векторами:**
```python
a = np.array([22, 4, -6, 3.9, 7]) 
b = np.array([12, -0.5, 6, 42, 1]) 
np.linalg.norm(a - b, np.inf) 
--- 
-> 38.1
```
**Матрица (одна строка матрицы - один массив):**
```python
m = np.array([[2, 17, 43], 
			  [36, 8, 4], 
			  [10, -11, 6], 
			  [3, 7, 15] ])
```
**Вектор-столбец:**
```python
k = np.array([[1], 
			  [14], 
			  [6], 
			  [9]]) 
```
**Транспонирование матрицы:**
```python
k = np.array([[1], 
			  [14], 
			  [6], 
			  [9]]) 
k.T
---
-> array([[ 1, 14,  6,  9]])
```
Чтобы вывести целую строку, второй параметр индекса нужно заменить на `:`. Чтобы вывести целый столбец, первый параметр индекса нужно заменить на `:`.

**Главная диагональ матрицы:**
```python
c = np.array([
    [-8,-2,3,1],
    [-1,14,-9,9],
    [6,-4,2,-10],
    [0,-2,4,-14]
])
c.diagonal()
---
-> array([ -8,  14,   2, -14])
```
**Размерность вектора или матрицы:**
```python
a = np.array([
    [5, -7, 2],
    [-2, -1, 9],
    [3, 1, 4]
])
k = np.array([3, 4, 5])
a.shape
k.shape
---
-> (3, 3)
-> (3,)
```
**Превращение вектора в матрицу (строку или столбец):**
```python
k = np.array([3, 4, 5])

k.reshape(1,3)
k.reshape(3,1)
---
-> array([
		  [3, 4, 5]
		])
-> array([
		  [3],
	      [4],
	      [5]
	    ])
```
**Произведение вектора на матрицу:**
```python
a = np.array([
    [5, -7, 2], 
    [-2, -1, 9], 
    [3, 1, 4]
])
k = np.array([3, 4, 5])

k.reshape(1, 3) @ a
---
-> array([
		  [ 22, -20,  62]
		])

```
**Произведение матрицы на вектор:**
```python
a = np.array([
    [5, -7, 2], 
    [-2, -1, 9], 
    [3, 1, 4]
])
k = np.array([3, 4, 5])

a @ k.reshape(3,1)
---
-> array([
		  [-3],
	      [35],
	      [33]
	    ])
```
**Продублировать массив n раз:**
```python
x = np.array([0,1,2])

np.tile(x, 3)
---
-> array([0, 1, 2, 0, 1, 2, 0, 1, 2])
```
**Продублировать каждый элемент массива n раз:**
```python
x = np.array([0,1,2])

np.repeat(x, 3)
---
-> array([0, 0, 0, 1, 1, 1, 2, 2, 2])
```
**Декартово произведение двух массивов:**
```python
x = np.array([0,1,2])
y = np.array([0,1,2])

np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
---
-> array([[0, 0],
       [1, 0],
       [2, 0],
       [0, 1],
       [1, 1],
       [2, 1],
       [0, 2],
       [1, 2],
       [2, 2]])
```
**Нахождение обратной матрицы для квадратной матрицы:**
```python
a = np.array([
    [3,4],
    [2,6]
])

np.linalg.inv(a)
---
-> array([
		  [ 0.6, -0.4],
	      [-0.2,  0.3]
	    ])
```
**Нахождение определителя для квадратной матрицы:**
```python
a = np.array([
    [3,4],
    [2,6]
])

np.linalg.det(a)
---
-> 10
```
**Решение системы линейный уравнений:**
```python
# Коэффициенты
a = np.array([
    [5,-6],
    [4,3]
])

# Свободные члены
b = np.array([5,43]).reshape(2,1)

np.linalg.solve(a,b)
---
-> array([[7.],
	      [5.]])

```
**Нахождение ранга матрицы:**
```python
a = np.array([
    [1,2,3],
    [0,1,4],
    [15,1,1]
])
np.linalg.matrix_rank(a)
---
-> 3
```

**Создание функции для нахождения корня полиномиального уравнения:**
```python
# 2x^2 + 4x + 6
poly_func = np.poly1d([2,4,6])
poly_func(3)
---
-> 36
```
- Вектор начинается с коэффициента при самой старшей степени x и заканчивается коэффициентом при свободном члене;
- Коэффициенты при отсутствующих степенях равны нулю;
- Если в многочлене какая-то степень вычитается, значит, коэффициент при ней отрицательный.

**Подбор полинома указанной степени, описывающего зависимость между наборами данных:**
```python
x = [1,2,3,4,5]
y = [5,4,3,2,1]

coefs = np.polyfit(x, y, deg=3)
coefs.round()
---
-> array([-0.,  0., -1.,  6.])
# Полином 3-ей степени, поднходящий для описания зависимости:
# 0x^3 + 0x^2 - 1x + 6
```
**Принимает:**
- Набор значений переменной `x`,
- Набор соответствующих им значений функции `y`,
- Степень полинома `deg` (1 — прямая, 2 — квадратичная функция и т. д.).<br>
**Возвращает:** вектор коэффициентов полинома, который приближает исходные данные. Полученный вектор можно подать в `np.poly1d(coefs)` и вычислить значение полинома для произвольного аргумента.

**Логарифмирование чисел и массивов:**
```python
line = np.array([386, 711, 712, 655, 978, 380, 868, 844, 338, 152])

# Степени, в которые нужно возвести 2, 10 и e, чтобы получить числа в массиве
np.log2(line)
np.log10(line)
np.log(line)
---
-> array([8.59245704, 9.47370575, 9.47573343, 9.3553511 , 9.93369065,
       8.56985561, 9.76155123, 9.72109919, 8.40087944, 7.24792751])
-> array([2.5865873 , 2.8518696 , 2.85247999, 2.8162413 , 2.99033885,
       2.5797836 , 2.93851973, 2.92634245, 2.5289167 , 2.18184359])
-> array([5.95583737, 6.56667243, 6.56807791, 6.48463524, 6.88550967,
       5.94017125, 6.76619171, 6.73815249, 5.8230459 , 5.02388052])
```
**Реализация логарифмической шкалы в python:**
```python
x = np.concatenate((np.random.randint(1,1000,size=25),
					np.random.randint(100000,10000000,size=25)))
y = np.random.randint(1,1000,size=50)

plt.figure(figsize=(7,5))
sns.scatterplot(x=x,y=y)

# plt.yscale('log')
plt.xscale('log')
plt.show()
---
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706631%201.png)

**Вычисление производной в точке численным методом:**
```python
# Определяем функцию и точку для вычисления производной.
def f(x):
    return 2*x**3 - 3*x**2
    
x = 4

# Определяем длину шага.
dx = 1e-6

# Рассчитываем значение производной в точке.
df = (f(x + dx) - f(x)) / dx

print("Приближенное значение производной:", df)
---
-> Приближенное значение производной: 72.00002100660186
# f(x) = 2x^3-3x^2
# f'(4) = 72
```
**Приближенное вычисление определенного интервала функции:**
```python
from scipy.integrate import quad

def f(x):
    return 2*x - x**2

quad(f, 0, 1)
---
-> (0.6666666666666666, 7.401486830834376e-15)

```
Вычисляет значение определённого интеграла функции при заданных пределах интегрирования. Возвращает два числа `(res, err)`: приближённое значение интеграла `res` и оценку погрешности `err`.

**Реализация градиентного спуска с предрасчитанными частными производными:**
```python
import numpy as np

# f(x,y) = (x-1)^2 * (y+1)^4 + (x+1)^4 * (y-1)^2 + (x-3)^2 * (y-2)^2

# Стартовая точка для спуска: 1.5, 1.5
x = np.array([1.5, 1.5])

# Сама функция
def f(x):
    return (
        (x[0] - 1) ** 2 * (x[1] + 1) ** 4
        + (x[0] + 1) ** 4 * (x[1] - 1) ** 2
        + (x[0] - 3) ** 2 * (x[1] - 2) ** 2
    )

# Производная по Х
def dx(x):
    return (
        2 * (x[0] - 1) * (x[1] + 1) ** 4
        + 4 * (x[0] + 1) ** 3 * (x[1] - 1) ** 2
        + 2 * (x[0] - 3) * (x[1] - 2) ** 2
    )

# Производная по У
def dy(x):
    return (
        4 * (x[0] - 1) ** 2 * (x[1] + 1) ** 3
        + 2 * (x[0] + 1) ** 4 * (x[1] - 1)
        + 2 * (x[0] - 3) ** 2 * (x[1] - 2)
    )

# Значение градиента в точке
def grad(x):
    return np.array([dx(x),dy(x)])

# Градиентный спуск с максимальным количеством шагов n_iter,
# скоростью спуска gamma и точность eps
def gradient_descent(n_iter,gamma,eps,x0):
    i = 0
    xi= x0
    f_old = f(xi)
    while i <= n_iter:
        xi = xi - grad(xi)*gamma
        f_new = f(xi)
        if abs(f_new  - f_old) <= eps:
            return xi,f_new
        f_old = f_new
        i += 1
    return xi,f_new


# Вывод точки
result,f_result = gradient_descent(n_iter=1000000,gamma=0.0001,
								   eps=1e-6,x0=x)
print('Финальное значение точки x:')
print(result)
print('Финальное значение f(x):')
print(f_result)
---
-> Финальное значение точки x:
-> [1.03551325 1.18053937]
-> Финальное значение f(x):
-> 3.1795824425287895
```
**Градиентный спуск для поиска коэффициентов линейной регрессии:**
```python
import numpy as np

# Признаки
x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 0., 1., 2., 3., 4., 5., 6., 7.,
       8., 0., 1., 2., 3., 4., 5., 6., 7., 8., 0., 1., 2., 3., 4., 5., 6.,
       7., 8., 0., 1., 2., 3., 4., 5., 6., 7., 8., 0., 1., 2., 3., 4.])

# Значения
y = np.array([ 9. ,  7.5,  7.5, 10. ,  5.5,  7.5,  8.5,  5. ,  5.5,  5. ,  7. ,
        8.5,  8.5,  5. ,  6.5,  8.5,  3.5,  3. ,  5.5,  6. ,  7. ,  7. ,
        5.5,  6. ,  8.5,  7. ,  6. ,  6. ,  8. , 10. ,  7. ,  6. ,  4.5,
        6.5,  4.5,  6.5,  7. ,  8.5,  6. ,  6.5,  9.5,  8. ,  8. ,  5. ,
        3.5,  6. ,  9. ,  6.5,  5.5,  8. ])

X = np.stack([x,np.ones(len(x))],axis=1)

# MSE
def f(X, y, w):
    return np.linalg.norm(X @ w - y)**2 / len(X)

# Градиент MSE
def grad(X, y, w):
    return 2 * X.T @ (X @ w - y) / len(X)

gamma = 0.01
eps = 1e-5
n_iter = 1000

def gradient_descent(gamma,eps,n_iter,x,y):
    
    w = np.array([1,1])
    i = 0
    f_old = f(x,y,w)
    while i <= n_iter:
        w = w - grad(x,y,w)*gamma
        f_new = f(x,y,w)
        if np.abs(f_new - f_old) < eps:
            return w, f_new
        f_old = f_new
        i += 1
    
    return w, f_new

w, mse = gradient_descent(gamma,eps,n_iter,X,y)
print(w)
print(mse)
---
-> [-0.21865413  7.53541504] 
-> 2.3612573476683845
```
**Градиентный спуск для поиска коэффициентов полиномиальной функции:**
```python
import numpy as np
x = np.array([-9.93841085, -8.2398223 , -9.06505398, -7.35203062, -5.82847285,
        -5.08181713, -3.37174708, -3.6361873 , -0.06175255,  0.09106786,
         1.46721029,  0.41053496,  1.71012239,  1.84871104,  6.68526793,
         6.82543486,  6.64741998,  8.01775519,  8.57773967, 11.8291112 ])
y = np.array([ 0.99747243,  3.28729745,  2.0644648 ,  2.88068415, -0.05454181,
         0.63703982,  0.06238763,  0.25253028,  0.06582577,  0.05755049,
         0.20686123, -0.03885818,  0.40837474,  0.52833438,  0.25072492,
         0.26994154,  0.29157405,  0.52908138, -0.04000158, -0.98596774])

# Полином 3-ей степени
X = np.stack([x**3,x**2,x,np.ones(len(x))],axis=1)

def f(x, y, w):
    return np.mean((x @ w - y)**2)

def grad(x, y, w):
    return 2 / len(x) * x.T @ (x @ w - y)

gamma = 1e-6
max_iter = 1e8
eps = 1e-5
n_iter = 100000

def gradient_descent(gamma,eps,n_iter,X,y):
    
    wgh = np.array([0.01, 0.02, 0.03, 0.04])
    i = 0
    f_old = f(X,y,wgh)
    while i <= n_iter:
        wgh = wgh - grad(X,y,wgh)*gamma
        f_new = f(X,y,wgh)
        if np.abs(f_new - f_old) < eps:
            return wgh, f_new
        f_old = f_new
        i += 1
    
    return wgh, f_new

result, mse = gradient_descent(gamma,eps,n_iter,X,y)
print(result)
print(mse)
---
-> [-0.00177639  0.0129965   0.0297391   0.03998941]
-> 0.47597796477278287
```
**Собственные значения и векторы матрицы:**
```python
a = np.array([
	    [-2, 1, 0],
	    [5, 6, 0],
	    [0, 0, 4]
	])
w, v = np.linalg.eig(a)

print("Собственные значения:")
print(w)
print("Собственные векторы:")
print(v) 
---
-> Собственные значения:
[-2.58257569  6.58257569  4.        ]
-> Собственные векторы:
[[-0.86406369 -0.11573221  0.        ]
 [ 0.5033825  -0.99328045  0.        ]
 [ 0.          0.          1.        ]]
```
**Спектральное разложение матрицы:**
```python
a = np.array([
    [9, 2, -1],
    [2, 0, 3],
    [-1, 3, 4]
])
L, Q = np.linalg.eig(a) # сначала значения, потом векторы
L = np.diag(L)
Q_1 = np.linalg.inv(Q)

print("Проверка:")
print((Q @ L @ Q_1).round()) # проверка с округлением до целых
---
-> Проверка:
[[ 9.  2. -1.]
 [ 2.  0.  3.]
 [-1.  3.  4.]]
```
**Сингулярное разложение матрицы:**
```python
a = np.array([
    [3, -9],
    [1, 9]
])

c = a.T @ a

# Находим собственные числа и значения матрицы A.T @ A
c_vals, c_vecs = np.linalg.eig(c)

# Сортируем собственные значения.
indices = np.argsort(c_vals)  
# "Переворачиваем", чтобы сделать сортировку по убыванию.
indices = indices[::-1]  
c_vals = c_vals[indices]
# Сортируем собственные векторы 
V = c_vecs[:, indices]

# Составляем диагональную матрицу из корней собственных числел матрицы A.T @ A
S = np.diag(np.sqrt(c_vals))

# Вычисляем матрицу левых сингулярных векторов
U = a @ V @ np.linalg.inv(S)  

print("Проверка:")
print((U @ S @ V.T).round())  
---
-> Проверка:
[[ 3. -9.]
 [ 1.  9.]]
```
**Сингулярное разложение матрицы (функция):**
```python
import numpy as np 

a = np.array([[3, -9],
              [1, 9]]) 

U, S, Vt = np.linalg.svd(a) 

print("Левые сингулярные векторы:") 
print(U) 
print("Сингулярные значения:") 
print(S) 
print("Правые сингулярные векторы:") 
print(Vt.T) 
print("Проверка") 
print((U @ np.diag(S) @ Vt).round())
---
-> Левые сингулярные векторы:
[[-0.72498785  0.68876166]
 [ 0.68876166  0.72498785]]
-> Сингулярные значения:
[12.81024968  2.81024968]
-> Правые сингулярные векторы:
[[-0.11601662  0.99324727]
 [ 0.99324727  0.11601662]]
-> Проверка
[[ 3. -9.]
 [ 1.  9.]]
```
Возвращает матрицы $U$,$S$ (в виде вектора диагонали) и $V^T$ для формулы:<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706631.png)

**Пример реализации LSA:**
```python
import numpy as np

# Загружаем матрицу встречаемости слов.
# allow_pickle=True сообщает NumPy, что файл сохранён в специальном формате pickle.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_term_occurence.npy"
filename = np.DataSource().open(url).name
X = np.load(filename, allow_pickle=True)

# Загружаем слова, соответствующие строкам матрицы.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_words.npy"
filename = np.DataSource().open(url).name
words = np.load(filename, allow_pickle=True)

# Загружаем оригинальные описания фильмов.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_summaries.npy"
filename = np.DataSource().open(url).name
summaries = np.load(filename, allow_pickle=True)


U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Количество используемых сингулярных векторов\топиков.
k = 10

# Считаем эмбеддинги слов с помощью матрицы U и s.
word_vectors = (U @ np.diag(s))[:, :k]

# Считаем эмбеддинги документов с помощью матрицы V и s.
doc_vectors = (np.diag(s) @ Vt)[:k].T # Транспонируем, чтобы объединить с пространством слов.

# Новый документ по словам. Поддерживаются только слова из `words`.
new_doc = ["бильбо", "кольцо", "хоббит"]

# Строим вектор нового документа как сумму векторов его слов. 
# Сразу создаём вектор-строку, чтобы потом не транспонировать.
search_vector = np.zeros((1, k))
i = 0
while i < len(new_doc):
		# Находим вектор слова.
    search_index = np.where(words == new_doc[i])[0]
		# Прибавляем вектор слова к вектору нового документа.
    search_vector += word_vectors[search_index]
    i += 1

# Усредняем векторы слов в документе.
search_vector /= len(new_doc) 

# Ищем ближайшие векторы документов и возвращаем соответствующие тексты.
def find_nearest_doc(doc_vector, n=3):
		# L2 расстояние между векторами.
    distances = np.mean((doc_vector - doc_vectors) ** 2, axis=1)
		# Индексы векторов с наименьшим расстоянием.
    return summaries[np.argsort(distances)[1:n]]

print(find_nearest_doc(search_vector, n=10))
```

**Поиск документа, который ближе всего ко слову "вкусный" (индекс=3)**
```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

preprocessed_texts = np.array(['игра крутой большой онлайн',
                                 'огурец отстой не вкусный',
                                 'огурчик супер',
                                 'обычный огурчик понравиться соленый',
                                 'игра ужасный геймплей донат',
                                 'лопата шикарный взять на дача',
                                 'вкусный как на дача',
                                 'дрель мощь просверливать весь стена',
                                 'огурец понравиться свежий',
                                 'вкусный немного горьковатый'])

# Строим матрицу встречаемости слов.
count_model = CountVectorizer()
X = count_model.fit_transform(preprocessed_texts).toarray().astype(float).T
words = count_model.get_feature_names_out()

# Функция возвращает индекс ближайшего к `search_vector` вектора из строк `vectors`.
def find_nearest_vector_id(search_vector, vectors):
		# L2 расстояние между векторами.
    distances = np.mean((search_vector - vectors) ** 2, axis=1)
		# Индексы векторов с наименьшим расстоянием.
    return np.argmin(distances)

U, s, VT = np.linalg.svd(X, full_matrices=False)
word_emb = U @ np.diag(s)
doc_emb = np.diag(s) @ VT

search_index = 3
search_vector = word_emb[search_index]

ans = find_nearest_vector_id(search_vector, doc_emb)
print(preprocessed_texts[ans])
```

**Дискретные распределения в scipy.stats:**<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706630%202.png)<br>
**Вероятность конкретного значения:**
```python
# pmf - probability mass function (функция вероятности)
from scipy.stats import poisson 

X = poisson(85)
p = X.pmf(86)
print(p)
---
-> 0.042726301471754845
```
*Эквивалентно:*<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706630%201.png)
```python
def poisson(mu,k):
    return (mu**k * np.e**(-mu)) / math.factorial(k)

p = poisson(85,86)
print(p)
---
-> 0.04272630147175384
```

**Вероятность такого или меньшего значения:**
```python
# cdf - cumulative distribution function (функция распределения)
from scipy.stats import binom 

X = binom(10,0.4)
p = X.cdf(3)
print(p)
---
-> 0.3822806015999999
```
*Эквивалентно:*<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706630.png)
```python
def binom(p,k,n):
    
    c = len(list(itertools.combinations([i for i in range(n)],k)))
    pfin = p**k
    qfin = (1-p)**(n-k)
    
    return c * pfin * qfin

p = binom(0.4,0,10) + binom(0.4,1,10) + binom(0.4,2,10) + binom(0.4,3,10) 
print(p)
---
-> 0.3822806016
```

**Вероятность большего значения:**
```python
# cdf - cumulative distribution function (функция распределения)
from scipy.stats import binom 

X = binom(10,0.4)
p = 1 - X.cdf(3)
print(p)
---
-> 0.6177193984000001
```
*Эквивалентно:*
```python
def binom(p,k,n):
    
    c = len(list(itertools.combinations([i for i in range(n)],k)))
    pfin = p**k
    qfin = (1-p)**(n-k)
    
    return c * pfin * qfin

p = 1 - (binom(0.4,0,10) + binom(0.4,1,10) + binom(0.4,2,10) + binom(0.4,3,10))
print(p)
---
-> 0.6177193984
```

**Визуализация распределения:**
```python
from scipy.stats import binom 

X = binom(10,0.4)

line = np.arange(0,11)
cdf_line = X.cdf(line)
plt.plot(line,cdf_line)
plt.show()
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706629%201.png)
```python
from scipy.stats import binom 

X = binom(10,0.4)

line = np.arange(0,11)
pmf_line = X.pmf(line)
plt.plot(line,pmf_line)
plt.show()
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706629.png)

**Непрерывные распределения в scipy.stats:**

**Плотность распределения в точке:**<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706628%202.png)
```python
# pdf - probability density function (функция плтности вероятнсти)
from scipy.stats import uniform 

X = uniform(loc=0, scale=100)
X.pdf(50)
---
-> 0.01
```

**Функция распределения:**<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706628%201.png)
```python
# cdf - cumulative distribution function (функция распределения)
from scipy.stats import expon
X = expon(scale=45)

X.cdf(120)
---
-> 0.9305165487771985
```
Функция для экспоненциального распределения принимает на вход не сам параметр $\lambda$, а среднее значение, то есть $1/\lambda$.

**Применительно к нормальному распределению:**
```python
from scipy.stats import norm 

X = norm(loc=10,scale=2)

# вероятность получить значение меньше или равное 8:
print(X.cdf(8))
# вероятность получить значение больше 8:
print(1 - X.cdf(8))
# вероятность получить значение от 6 до 8:
print(X.cdf(8) - X.cdf(6))
# плотность вероятности в точке 8
print(X.pdf(8))
---
-> 0.15865525393145707
-> 0.8413447460685429
-> 0.13590512198327787
-> 0.12098536225957168
```
Вместо $1 - cdf(x)$ можно использовать $sf(x)$:
```python
from scipy.stats import norm 

X = norm(loc=10,scale=2)

# вероятность получить значение больше 8:
print(X.sf(8))
---
-> 0.8413447460685429
```
**Обратная функция распределения (квантиль):**
```python
from scipy.stats import norm

# Параметры стандартного нормального распределения
Z = norm(loc=0, scale=1)

# Находим квантиль уровня 0.95 (percent point function)
z_95 = Z.ppf(0.95)
print(z_95)
---
-> 1.6448536269514722
```
*Обратное действие:*
```python
Z = norm(loc=0, scale=1)
Z.cdf(1.6448536269514722)
---
-> 0.95
```
**Выборка из известного распределения:**
```python
# выборка размера 10 из биномиального распределения
np.random.binomial(n=10, p=0.3, size=10)
# выборка размера 10 из распределения Пуассона
np.random.poisson(lam=15, size=10)
# выборка размера 10 из распределения Бернулли
np.random.binomial(n=1, p=0.3, size=10)
---
-> array([2, 5, 2, 4, 6, 3, 1, 4, 2, 4])
-> array([18, 22, 16, 10, 12, 16, 12, 20, 13, 16])
-> array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
```
**Построение эмпирической функции распределения:**<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706628.png)
```python
# для каждого значения в выборке доля значений меньших или равных ему
def ecdf(a):  
    x, counts = np.unique(a, return_counts=True) 
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

# параметры нормального распределения
points_number = 250
mu = 100
std = 15


X = norm(loc=mu,scale=std)
# выборка из нормального распределения
samp = pd.DataFrame(
    {'wgh':X.rvs(size=points_number)}
)

# Создаём массивы координат точек x и y эмпирической функции распределения
x, y = ecdf(samp['wgh']) 


# Создаём точки по оси x
xCDF = np.linspace(min(samp['wgh']) - 1, max(samp['wgh']) + 1, points_number) 
yCDF = stats.norm.cdf(xCDF, loc = mu, scale = std) 
# график эмпирической функции распределения
plt.plot(x, y, drawstyle='steps-post') 
# график реальной функции распределения
plt.plot(xCDF, yCDF) 

plt.show()
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706627%201.png)<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706627.png)<br>
Если увеличить количество элементов в выборке, то видно, как график эмпирической функции распределения будет всё лучше приближать график функции распределения нормального распределения.

**Расстояние между эмпирическим и предполагаемым распределением:**<br>
![Pasted image 20250114213522.png](../Вложения/Математические%20операции%20в%20python/file-20251104201706624%201.png)
```python
def ecdf(a):  
    x, counts = np.unique(a, return_counts=True) 
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

points_number = 100
mu = 100
std = 15

norm = stats.norm(loc = mu, scale = std)
samp = norm.rvs(size=points_number)

x, y = ecdf(samp) 
yCDF = norm.cdf(x) 

# сумма квадратов разности между фактическим и прдполагаемым значениями
np.sum((y - yCDF)**2) / len(x)
---
-> 0.0016207068193294727
```
**Демонстрация ЦПТ:**<br>
![Pasted image 20250115192532.png](../Вложения/Математические%20операции%20в%20python/file-20251104201706624.png)
```python
summa = []
# Извлекаем 1000 раз выборку размера 50
# из экспоненциального распределения со средним 1/10
# сохраняем выборочные средние и считаем их среднее
means = []
for j in range(1000):
    
    x = expon(scale=1/10).rvs(50)
    m = np.mean(x)
    
    means.append(m)

np.mean(means)
np.var(means)
---
-> 0.09918970200794265
-> 0.00018929415759760186
```
![Pasted image 20250115184921.png](../Вложения/Математические%20операции%20в%20python/file-20251104201706623.png)
```python
# Подбрасываем 1000 раз по 5 кубиков d6 и считаем сумму очков
# Считаем среднее получившихся сумм
summa = []
for j in range(1000):
    
    x = Randint(1,7).rvs(5)
    s = np.sum(x)
    
    summa.append(s)

np.mean(summa)
np.var(summa)
---
-> 17.557
-> 14.866751
```
**Поиск минимума функции:**
```python
from scipy.optimize import minimize

def f(x):
    return x**2 + 5

minimize(fun=f, x0=10, tol=1e-3, method='Nelder-Mead')
---
-> message: Optimization terminated successfully.
       success: True
        status: 0
           fun: 5.0
             x: [ 0.000e+00]
           nit: 17
          nfev: 34
 final_simplex: (array([[ 0.000e+00],
                       [-9.766e-04]]), array([ 5.000e+00,  5.000e+00]))

```
**Метод максимального правдоподобия:**<br>
У нас есть выборка $X$ с таким распределением:<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706626%201.png)<br>
Эта гистограмма не похожа ни на одну из известных нам плотностей распределения. Она состоит из двух частей, каждая из которых похожа на нормальное распределение. При этом высота пика одного нормального распределения больше, чем другого.

Придумаем функцию плотности для такого распределения. Например, такую:<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706626.png)<br>
Запишем функцию правдоподобия для нашей выборки:<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706625%202.png)<br>
Теперь необходимо найти максимум этой функции:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Наша выборка:
size = 1000
k1 = 0.5
mu1 = 0
sigma1 = 10
mu2 = 100
sigma2 = 20
X = np.concatenate((np.random.normal(mu1, sigma1, int(k1 * size)),
                    np.random.normal(mu2, sigma2, int((1 - k1) * size))), axis=0)

# Функция, предположительно описывающая распределение:
def normal_mixture(params):
    omega, mu1, sigma1, mu2, sigma2 = params
    s = 0
    i = 0
    while (i < len(X)):
        s += np.log(float(omega / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (X[i] - mu1) ** 2 / sigma1 ** 2)
                          + (1 - omega) / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (X[i] - mu2) ** 2 / sigma2**2)))
        i = i + 1
    
    # возвращаем -s, так как ищем не минимум, а максимум
    return -s


def opt():
    # вектор начальных значений
    x0 = np.array([0.5, 0, 50, 200, 100])
    # Оптимизация
    return minimize(normal_mixture, x0, tol=1e-3, method='Nelder-Mead')


estimate_parameters = opt().x
print("Оценка параметра omega: ", estimate_parameters[0])
print("Оценка параметра mu1: ", estimate_parameters[1])
print("Оценка параметра sigma1: ", estimate_parameters[2])
print("Оценка параметра mu2: ", estimate_parameters[3])
print("Оценка параметра sigma2: ", estimate_parameters[4])
---
-> Оценка параметра omega:  0.499121113189275
-> Оценка параметра mu1:  0.006512308135707755
-> Оценка параметра sigma1:  9.73163265305423
-> Оценка параметра mu2:  99.43579943507787
-> Оценка параметра sigma2:  19.654157112382695
```
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706625%201.png)

**Доверительный интервал для Z-распределения:**<br>
![](../Вложения/Математические%20операции%20в%20python/file-20251104201706625.png)
```python
def get_inteval(x, std, alpha):
    
    samp_mean = np.mean(x)
    
    Z = norm(loc=0,scale=1)
    left = Z.ppf(1 - alpha/2) * std / np.sqrt(len(x))
    right = Z.ppf(alpha/2) * std / np.sqrt(len(x))
    
    return samp_mean-left , samp_mean-right

# Выборка
x = np.array([1200, 1500, 1100, 1300, 1400, 1250, 1600, 1100, 1700, 1500])
# СКО ГС
std = 250
left, right = get_inteval(x, std, 0.05)
---
-> (1210.0512419238596, 1519.9487580761404)
```
**Доверительный интервал для t-распределения:**
```python
student = stats.t(df=len(x)-1)

def get_inteval(x, alpha):
    
    std = np.std(x, ddof=1)
    samp_mean = np.mean(x)
    
    student = stats.t(df=len(x)-1)
    left = student.ppf(1 - alpha/2) * std / np.sqrt(len(x))
    right = student.ppf(alpha/2) * std / np.sqrt(len(x))
    
    return samp_mean-left, samp_mean-right
    
# Выборка
x = np.array([25, 29, 31, 34, 30, 30, 32, 28, 28, 33, 29, 24, 30, 24, 34])
left, right = get_inteval(x, 0.05)
---
(27.60188428433719, 31.19811571566281)
```

**Бутстреп (пример с оценкой вариабельности медианы):**
```python
def bootstrap(sample,n,k):
    
    bootstrap_medians = []
    for i in range(k):
        new_sample = np.random.choice(a=sample, size=n, replace=True) 
        median = np.median(new_sample)
        bootstrap_medians.append(median)
    
    return np.mean(bootstrap_medians), np.std(bootstrap_medians)

times = pd.read_csv('https://code.s3.yandex.net/Math/datasets/Times.csv', header=None).values.flatten().tolist()

median_mean, median_std = bootstrap(sample=times,n=len(times),k=10000)
print("Оценка медианы из исходного набора данных:", np.median(times))
print("Среднее значение медианы по всем поднаборам данных:", median_mean)
print("Оценка стандартного отклонения медианы:", median_std)
---
-> Оценка медианы из исходного набора данных: 50.25951632518907
-> Среднее значение медианы по всем поднаборам данных: 50.27423459651471
-> Оценка стандартного отклонения медианы: 0.7410657382148823
```
**Доверительный интервал для бутстрапированной величины:**
```python
Z = norm(loc=0, scale=1)
alpha = 0.05
z_value = Z.ppf(q=1-alpha/2)

me_left = median_mean - z_value * median_std / np.sqrt(k)
me_right = median_mean + z_value * median_std / np.sqrt(k)

me_left, me_right
---
(50.23673218487899, 50.31173700815043)
```

**Уменьшение размерности данных методом главных компонент (вручную):**
```python
import numpy as np
from matplotlib import pyplot as plt

def pca(data, n_components):
    
    # центрируем данные
    data_centered = data - np.mean(data, axis=0) 
    # вычисляем матрицу ковариации
    cov_mat = np.cov(data_centered, rowvar = False) 
    # вычисляем ее собственные значения и векторы
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat) 
    # сортируем собственные значения по убыванию
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    # сортируем собственные векторы
    sorted_eigenvectors = eigen_vectors[:,sorted_index] 
    # отбираем первые n собственных векторов
    eigenvector_subset = sorted_eigenvectors[:,0:n_components] 
    # переводим даные в новое пространство меньшей размерности
    data_reduced = data_centered @ eigenvector_subset
    
    return data_reduced

data = np.array([
        [1, 4, 5, 5, 12],
        [3, 6, 2, 2, 5],
        [2, 10, 12, 0, 2],
        [3, 9, 6, 2, 10],
        [2, 5, 0, 6, 25],
        [0, 8, 3, 3, 14],
        [1, 8, 9, 2, 12],
        [1, 6, 2, 3, 23],
        [2, 8, 10, 7, 26],
        [0, 5, 7, 2, 16]
])

data_reduced = pca(data, 2)
---
-> array([[ 1.59218949e+00, -1.89140643e+00],
       [ 8.60104975e+00, -5.65599840e+00],
       [ 1.41164338e+01,  4.20375877e+00],
       [ 4.87811403e+00, -2.59195015e-03],
       [-1.18228772e+01, -3.55118381e+00],
       [ 1.02427079e-01, -2.18054748e+00],
       [ 3.36409826e+00,  2.96202031e+00],
       [-8.85035613e+00, -1.83832754e+00],
       [-1.08175105e+01,  6.87919355e+00],
       [-1.16356857e+00,  1.07508299e+00]])
```
**Или при помощи SCD:**
```python
def pca_svd(data, n_components):
    
    data_centered = data - data.mean(axis=0)
    
    U, S, Vt = np.linalg.svd(data_centered)
    data_reduced = data_centered @ Vt.T[:, 0:n_components]

    return data_reduced

data_reduced = pca(data, 2)
---
-> array([[ 1.59218949e+00, -1.89140643e+00],
       [ 8.60104975e+00, -5.65599840e+00],
       [ 1.41164338e+01,  4.20375877e+00],
       [ 4.87811403e+00, -2.59195015e-03],
       [-1.18228772e+01, -3.55118381e+00],
       [ 1.02427079e-01, -2.18054748e+00],
       [ 3.36409826e+00,  2.96202031e+00],
       [-8.85035613e+00, -1.83832754e+00],
       [-1.08175105e+01,  6.87919355e+00],
       [-1.16356857e+00,  1.07508299e+00]])
```