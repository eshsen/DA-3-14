import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
import argparse


def setup_argparse():
    """
    Настравивает парсер аргументов командной строки.

    Возвращает
    ---------
    argparse.ArgumentParser
        Объект парсера с заданными параметрами:
        - length: длина временного ряда
        - seasonal_period: период сезонности
        - seasonal_strength: амплитуда сезонного сигнала
        - noise_std: стандартное отклонение шума
        - random_seed: случайное зерно для генерации
        - max_lags: максимальное число логов для ACF
        - threshold: порог для детекции значимого пика сезонности
        - save_path: базовый путь для сохранения графиков
    """
    parser = argparse.ArgumentParser(
        description='Анализ сезонности временного ряда через ACF')
    parser.add_argument('--length', type=int, default=1000,
                        help='Длина временного ряда (по умолчанию 1000)')
    parser.add_argument('--seasonal_period', type=int, default=12,
                        help='Период сезонности (по умолчанию 12)')
    parser.add_argument('--seasonal_strength', type=float, default=1.5,
                        help='Амплитуда сезонного сигнала (по умолчанию 1.5)')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='Стандартное отклонение шума (по умолчанию 1.0)')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Случайное зерно для генерации (по умолчанию 0)')
    parser.add_argument('--max_lags', type=int, default=48,
                        help='Максимальное число лагов для ACF (по умолчанию 48)')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Порог для детекции значимого пика сезонности (по умолчанию 0.25)')
    parser.add_argument('--save_path', default=None,
                        help='Базовый путь для сохранения графиков.'
                             'Если указан, графики сохраняются в два файла: '
                             'name_acf.png и name_acf_with_peaks.png')
    return parser


def generate_synthetic_series(n, seasonal_period, seasonal_strength, noise_std, random_seed):
    """
    Генерирует синтетический временной ряд с сезонностью и шумом.

    Параметры
    ---------
    n : int
        Длина временного ряда.
    seasonal_period : int
        Период сезонности.
    seasonal_strength : float
        Амплитуда сезонного компонента.
    noise_std : float
        Стандартное отклонение шумовой компоненты.
    random_seed : int
        Начальное значение генератора случайных чисел.

    Возвращает
    ---------
    pandas.Series 
        Синтетический временной ряд с индексом от 0 до n-1.
    """
    np.random.seed(random_seed)
    t = np.arange(n)
    seasonal = seasonal_strength * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(scale=noise_std, size=n)
    data = seasonal + noise
    return pd.Series(data, index=pd.RangeIndex(start=0, stop=n))


def calculate_acf(ts, max_lags):
    """
    Рассчитывает ACF временного ряда ts до max_lags.

    Параметры
    ---------
    ts : pandas.Series
        Изначальный синтетический времнной ряд
    max_lags : int
        Максимальное количество лагов для расчёта ACF

    Возвращает
    ---------
    numpy.ndarray
        Массив значений ACF от лага 0 до max_lags.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("ts должен быть pandas.Series")
    if max_lags <= 0 or not isinstance(max_lags, int):
        raise ValueError("max_lags должен быть положительным целым числом")
    # Вычисляем ACF с использованием FFT для ускорения
    acf_values = acf(ts, nlags=max_lags, fft=True)
    return acf_values


def find_acf_peaks(acf_values, target_lags):
    """
    Находит пики ACF на целевых лагах.

    Параметры
    ---------
    acf_values : List[float]
        Массив значений автокорреляционной функции (начиная с лага 0).
    target_lags : List[int]
        Список лагов, на которых нужно искать пики.

    Возвращает
    ---------
    List[Tuple[int, float]]
        Список кортежей (лаг, значение ACF) для найденных пиков на целевых лагах.
    """
    peaks, _ = find_peaks(acf_values[1:])
    peaks_lags = peaks + 1
    found_peaks = [(lag, acf_values[lag])
                   for lag in peaks_lags if lag in target_lags]
    return found_peaks


def plot_acf(lags, acf_values, peaks=None, title='Autocorrelation Function (ACF)', save_path=None):
    """
    Строит график ACF, опционально выделяет пики.

    Параметры
    ---------
    lags : array-like
        Список или массив значений лагов.
    acf_values : array-like
        Значения автокорреляционной функции для соответствующих лагов.
    peaks : list of tuples, optional
        Список кортежей (лаг, значение), отмечающих пики на графике.
    title : str, optional
        Заголовок графика.
    save_path : str или None, optional
        Путь для сохранения изображения. Если None, график отображается.
    """
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=lags, y=acf_values, marker='o', color='b')
    if peaks:
        for lag, val in peaks:
            plt.plot(lag, val, 'ro', markersize=8)
            plt.annotate(f'lag={lag}', xy=(lag, val), xytext=(lag+1, val),
                         fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.5))
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def detect_seasonality(found_peaks, threshold):
    """
    Определяет наличие сезонности по найденным пикам.

    Параметры
    ---------
    found_peaks : List[Tuple[int, float]]
        Список кортежей (лаг, значение ACF) для найденных пиков на целевых лагах.
    threshold : float
        Порог для детекции значимого пика сезонности.

    Возвращает
    ---------
    lag : int или None
        Сезонный период или None.
    """
    for lag, value in found_peaks:
        if value > threshold:
            return lag
    return None


def main():
    """
    Основная функция для всех задач
    """
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        ts = generate_synthetic_series(
            n=args.length,
            seasonal_period=args.seasonal_period,
            seasonal_strength=args.seasonal_strength,
            noise_std=args.noise_std,
            random_seed=args.random_seed
        )
        save_base = args.save_path
        acf_values = calculate_acf(ts, max_lags=args.max_lags)
        lags = np.arange(args.max_lags + 1)

        if save_base:
            save_acf = f"{save_base}_acf.png"
        else:
            save_acf = None
        plot_acf(lags, acf_values,
                 title='Autocorrelation Function (ACF)', save_path=save_acf)

        target_lags = list(
            range(args.seasonal_period, args.max_lags + 1, args.seasonal_period))
        found_peaks = find_acf_peaks(acf_values, target_lags)
        print("Пики ACF на заданных лагах:")
        for lag, value in found_peaks:
            print(f"Lag {lag}: ACF = {value:.4f}")

        if save_base:
            save_acf_with_peaks = f"{save_base}_acf_with_peaks.png"
        else:
            save_acf_with_peaks = None
        plot_acf(lags, acf_values, peaks=found_peaks,
                 title='Autocorrelation Function (ACF) with peaks', save_path=save_acf_with_peaks)

        seasonal_period_estimate = detect_seasonality(
            found_peaks, args.threshold)
        if seasonal_period_estimate is not None:
            print(
                f"Сезонность обнаружена. Приблизительный сезонный период: {seasonal_period_estimate} лагов."
            )
        else:
            print(
                "Сезонность не обнаружена на заданном наборе лагов или пики не превышают порог.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
