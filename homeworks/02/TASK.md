# Знакомство с библиотеками

## Настройка окружения
Предполагается что мы работаем в среде как минимум `Ubuntu 22.04`

Установка Eigen3: 
```bash
sudo apt install libeigen3-dev
```

Установка OpenBlas:
```bash
sudo apt-get install libopenblas-dev
```

Установка libtorch:
- https://docs.pytorch.org/cppdocs/installing.html
- https://pytorch.org/get-started/locally/

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.14.1%2Bcpu.zip
unzip libtorch-shared-with-deps-latest.zip
```

При использовании CMake:
```bash
cmake -D<Your path>/libtorch/share/cmake/Torch ..
```

Загрузить dataset MNIST можно отсюда:

https://github.com/mrgloom/MNIST-dataset-in-different-formats/raw/refs/heads/master/data/Original%20dataset/train-images.idx3-ubyte

https://github.com/mrgloom/MNIST-dataset-in-different-formats/raw/refs/heads/master/data/Original%20dataset/train-labels.idx1-ubyte

Файлы надо переименовать в:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte

---

## Задания

### Часть 1. Eigen3

Eigen — header-only библиотека линейной алгебры.

1. **Основы.** Дополни `eigen.cpp`: создай матрицы `Matrix3d` и `Vector3d`, выполни сложение, умножение матрицы на вектор, транспонирование. Выведи размеры и элементы. Разберись, чем отличаются статические (`Matrix2f`) и динамические (`MatrixXd`) матрицы.

2. **Блоки и производительность.** Для cлучая больших матриц (e.g. `2000x2000`) сравни произвдительность `C = A * B` с наивным умножением тройным циклом. Объясни, что такое *lazy evaluation* (`.eval()`).

## Часть 2. OpenBLAS

OpenBLAS — высокопроизводительная реализация BLAS.

1. **Понять layout.** Разбери `sgemm.cpp`: что такое *leading dimension* (lda/ldb/ldc), разница `CblasColMajor` и `CblasRowMajor`.

2. **Наивный GEMM vs cblas_sgemm.** Напиши свою функцию `matmul_naive(A, B)` тройным циклом. Заполни матрицы `128x128`, `512x512`, `1024x1024` случайными значениями и сравни результат `cblas_sgemm` с наивным (проверь равенство с допуском `1e-2`). Посчитай скорость в GFLOPs = `2*m*n*k / время`.

3. **Многопоточность**. Используй:
```cpp
setenv("OPENBLAS_NUM_THREADS",std::to_string(threads).c_str(),1);
openblas_set_num_threads(threads);
```
Чтобы изменить количество потоков которое используется OpenBLAS, сравно производительность на больших размерах матриц.

## Часть 3. libtorch (`torch/`)

libtorch — C++ API PyTorch: тензоры, autograd, датасеты и нейросетевые модули.

1. **Тензоры.** Создай тензоры `tensors.cpp` с помощью `torch::randn`, `torch::arange`, `torch::zeros`, используй арифметический операторы, `reshape`, транспонирование, `mm` (matmul). Познакомься с устройством  тензора: `sizes()`, `dtype()`, `device()`, `.item<float>()`.

2. **Датасет и DataLoader.** Загрузи MNIST `dataloader.cpp` через `torch::data::datasets::MNIST`, примени `Stack<>`, сделай `make_data_loader` с `batch_size=64`. Выведи размер батча, min/max значений пикселей и посчитай среднее по датасету (это шаг к нормализации).

3. **Полная сеть и метрика.** К имеющемуся в `torch.cpp` MLP добавь: подсчёт *accuracy* на    обучающей выборке после каждой эпохи. Сохрани `net.pt` и загрузи обратно, проверь, что  предсказание не изменилось.