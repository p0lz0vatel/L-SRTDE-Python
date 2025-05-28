# L-SRTDE: Success Rate-based Adaptive Differential Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/numba-0.57+-orange.svg)](https://numba.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Оптимизированная реализация алгоритма L-SRTDE для соревнования GNBG 2024, основанная на адаптивном дифференциальном развитии с контролем скорости успеха.

## 🔥 Особенности

- **Высокая производительность**: Векторизованные операции и JIT-компиляция через Numba
- **Адаптивные параметры**: Динамическая настройка F и Cr на основе success rate
- **Две популяции**: Совместная работа populations new и top
- **Поддержка многопоточности**: Параллельная оценка особей

## 📦 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/L-SRTDE_GNBG-24.git
cd L-SRTDE_GNBG-24
```
