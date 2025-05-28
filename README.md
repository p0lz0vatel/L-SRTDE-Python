# L-SRTDE: Success Rate-based Adaptive Differential Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Перевод на язык Python оптимизированной реализации алгоритма L-SRTDE для соревнования GNBG 2024, основанная на адаптивном дифференциальном развитии с контролем скорости успеха.

## 🔥 Особенности

- **Адаптивные параметры**: Динамическая настройка F и Cr на основе success rate
- **Две популяции**: Совместная работа populations new и top
- **Поддержка многопоточности**: Параллельная оценка особей

## 📦 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/L-SRTDE_GNBG-24.git
cd L-SRTDE_GNBG-24
```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## 🚀 Использование

Основной скрипт:
```bash
python L-SRTDE.py
```
Параметры запуска (через аргументы командной строки):
```bash
python L-SRTDE.py --dim 30 --runs 31 --pop_size 10
```

## 📊 Результаты

Производительность на тестовых функциях GNBG:
Функция	Средняя ошибка	FEs до сходимости
F1	0 ± 0	67,714 ± 975
F5	0.015 ± 0.028	277,624 ± 120,136
F16	632.11 ± 125.87	970,092 ± 163,813

