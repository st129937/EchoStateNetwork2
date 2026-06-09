import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from esn_core import EchoStateNetwork
import data_utils as du

SEED = 42

# ==== Эксперимент 1 ====
def run_esp_practical_verification():
    u = np.random.RandomState(SEED).uniform(-0.1, 0.1, 1000)
    plt.figure(figsize=(8, 4))
    for rho in [0.7, 0.95, 1.2]:
        m = EchoStateNetwork(1, 200, 1, rho=rho, input_scaling=0.001, leaking_rate=1.0, seed=SEED)
        x1, x2 = np.zeros(200), np.random.RandomState(1).rand(200)
        diffs = []
        for t in range(len(u)):
            x1 = m._update(x1, [u[t]])
            x2 = m._update(x2, [u[t]])
            diffs.append(np.linalg.norm(x1 - x2))
        plt.plot(diffs, label=rf"$\rho$={rho}")
    plt.yscale('log'); plt.xlabel("Time steps"); plt.ylabel("State Diff")
    plt.title("ESP Check: State Convergence")
    plt.legend(); plt.grid(True)
    plt.savefig("esp_practical_verification.pdf")
    plt.close()
    print("Эксперимент 1 завершён, график esp_practical_verification.pdf")

# ==== Эксперимент 2: измерение MC для разных геометрий ====
def compute_mc(esn, u, max_delay=100, washout=200, rcond=1e-6):
    """Вычисление суммарной ёмкости памяти (MC) для заданного резервуара.
       Добавлен параметр rcond для псевдообращения (устойчивость)."""
    states = []
    x = np.zeros(esn.n_res)
    for t in range(len(u)):
        x = esn._update(x, [u[t]])
        states.append(np.concatenate([[1.0], [u[t]], x]))
    X = np.array(states)[washout:]
    # Псевдообратная матрица с подавлением малых сингулярных чисел
    pinvX = np.linalg.pinv(X, rcond=rcond)
    mc_sum = 0.0
    for k in range(1, min(max_delay, len(u)-washout)):
        target = u[washout-k : len(u)-k]
        w = pinvX @ target
        pred = X @ w
        if np.std(pred) > 1e-9 and np.std(target) > 1e-9:
            corr = np.corrcoef(target, pred)[0,1]
            mc_sum += corr**2
    return mc_sum

def run_mc_geometry_comparison():
    print("Эксперимент 2: измерение MC для разных геометрий, ρ, LR, input_scaling")
    n_res = 200
    u = np.random.RandomState(SEED).uniform(-1, 1, 4000)
    
    geometries = ['random', 'orthogonal', 'cyclic']
    rhos = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.02]
    lrs = [0.1, 0.3, 0.5, 0.7, 1.0]
    scales = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    default_rho = 0.9
    default_lr = 0.5
    default_scale = 0.2
    
    results = []
    
    for geom in geometries:
        print(f"\n=== Геометрия: {geom} ===")
        # Серия 1: варьируем ρ
        for rho in rhos:
            print(f"  rho={rho:.2f} (LR={default_lr}, scale={default_scale})...", end='', flush=True)
            try:
                esn = EchoStateNetwork(1, n_res, 1, reservoir_type=geom, rho=rho,
                                       leaking_rate=default_lr, input_scaling=default_scale,
                                       beta=0.0, seed=SEED)
                mc = compute_mc(esn, u)
                results.append({'geometry': geom, 'param': 'rho', 'value': rho, 'MC': mc})
                print(f" MC={mc:.2f}")
            except Exception as e:
                print(f" ошибка: {e}")
                results.append({'geometry': geom, 'param': 'rho', 'value': rho, 'MC': 0.0})
        
        # Серия 2: варьируем leaking_rate
        for lr in lrs:
            print(f"  LR={lr:.2f} (rho={default_rho}, scale={default_scale})...", end='', flush=True)
            try:
                esn = EchoStateNetwork(1, n_res, 1, reservoir_type=geom, rho=default_rho,
                                       leaking_rate=lr, input_scaling=default_scale,
                                       beta=0.0, seed=SEED)
                mc = compute_mc(esn, u)
                results.append({'geometry': geom, 'param': 'leaking_rate', 'value': lr, 'MC': mc})
                print(f" MC={mc:.2f}")
            except Exception as e:
                print(f" ошибка: {e}")
                results.append({'geometry': geom, 'param': 'leaking_rate', 'value': lr, 'MC': 0.0})
        
        # Серия 3: варьируем input_scaling
        for scale in scales:
            print(f"  scale={scale:.2f} (rho={default_rho}, LR={default_lr})...", end='', flush=True)
            try:
                esn = EchoStateNetwork(1, n_res, 1, reservoir_type=geom, rho=default_rho,
                                       leaking_rate=default_lr, input_scaling=scale,
                                       beta=0.0, seed=SEED)
                mc = compute_mc(esn, u)
                results.append({'geometry': geom, 'param': 'input_scaling', 'value': scale, 'MC': mc})
                print(f" MC={mc:.2f}")
            except Exception as e:
                print(f" ошибка: {e}")
                results.append({'geometry': geom, 'param': 'input_scaling', 'value': scale, 'MC': 0.0})
    
    # Сохранение результатов
    df = pd.DataFrame(results)
    df.to_csv('mc_full_results.csv', index=False)
    print("\nПолная таблица сохранена в mc_full_results.csv")
    
    # Построение графиков для каждого параметра
    for param in ['rho', 'leaking_rate', 'input_scaling']:
        plt.figure(figsize=(10, 6))
        for geom in geometries:
            sub = df[(df.geometry == geom) & (df.param == param)].dropna()
            if not sub.empty:
                sub_sorted = sub.sort_values('value')
                plt.plot(sub_sorted['value'], sub_sorted['MC'], marker='o', label=geom)
        xlabel = {'rho': r'$\rho(\mathbf{W})$', 'leaking_rate': 'Leaking rate', 'input_scaling': 'Input scaling'}[param]
        plt.xlabel(xlabel)
        plt.ylabel('MC')
        plt.title(f'Зависимость ёмкости памяти MC от {xlabel}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'mc_{param}.pdf')
        plt.close()
    
    # Генерация LaTeX-таблиц из результатов эксперимента
    # 1. Таблица для спектрального радиуса
    pivot_rho = df[df.param == 'rho'].pivot(index='value', columns='geometry', values='MC').round(2)
    with open('mc_rho_table.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Зависимость ёмкости памяти от спектрального радиуса $\\rho(\\mathbf{W})$ ($N=200$, $\\alpha=0.5$, $s=0.2$)}\n")
        f.write("\\label{tab:mc_rho}\n")
        f.write(pivot_rho.to_latex() + "\n")
        f.write("\\end{table}\n")

    # 2. Таблица для коэффициента утечки
    pivot_lr = df[df.param == 'leaking_rate'].pivot(index='value', columns='geometry', values='MC').round(2)
    with open('mc_lr_table.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Зависимость ёмкости памяти от коэффициента утечки $\\alpha$ ($N=200$, $\\rho=0.9$, $s=0.2$)}\n")
        f.write("\\label{tab:mc_lr}\n")
        f.write(pivot_lr.to_latex() + "\n")
        f.write("\\end{table}\n")

    # 3. Таблица для масштаба входа
    pivot_scale = df[df.param == 'input_scaling'].pivot(index='value', columns='geometry', values='MC').round(2)
    with open('mc_scale_table.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Зависимость ёмкости памяти от масштаба входа $s$ ($N=200$, $\\rho=0.9$, $\\alpha=0.5$)}\n")
        f.write("\\label{tab:mc_scale}\n")
        f.write(pivot_scale.to_latex() + "\n")
        f.write("\\end{table}\n")
    
    print("Эксперимент 2 завершён.")

# ==== Эксперимент 3: предсказание временных рядов ====
def run_time_series_prediction():
    esn_params = {
        'n_res': 500,
        'rho': 0.95,
        'leaking_rate': 0.5,
        'input_scaling': 0.1,
        'beta': 1e-6,
        'washout': 200
    }

    datasets = [
        ('NARMA-10', du.get_narma10, (4000, 1000)),
        ('Mackey-Glass', du.get_mackey_glass, (4000, 1000))  # теперь с delta_t=0.1
    ]

    all_results = []
    for name, gen_func, (train_len, test_len) in datasets:
        print(f"\n=== {name} ===")
        total = train_len + test_len
        u, y = gen_func(n=total, seed=SEED)
        tr_u, te_u = u[:train_len], u[train_len:]
        tr_y, te_y = y[:train_len], y[train_len:]

        # AR baseline (порядок 15)
        lags = 15
        y_true_ar, pred_ar = du.get_ar_baseline(tr_y.flatten(), te_y.flatten(), lags=lags)

        # ESN
        esn = EchoStateNetwork(1, esn_params['n_res'], 1,
                               reservoir_type='random',
                               rho=esn_params['rho'],
                               leaking_rate=esn_params['leaking_rate'],
                               input_scaling=esn_params['input_scaling'],
                               beta=esn_params['beta'],
                               seed=SEED)
        esn.fit(tr_u, tr_y, washout=esn_params['washout'])
        esn.save_matrices(folder="weights")
        pred_esn_full = esn.predict(te_u)
        pred_esn = pred_esn_full[lags:]
        y_true_esn = te_y[lags:]

        # Метрики с защитой от inf/NaN
        if np.std(y_true_esn) > 1e-12:
            nrmse_esn = du.nrmse(y_true_esn, pred_esn)
            mae_esn = du.mae_metric(y_true_esn, pred_esn)
        else:
            nrmse_esn = np.nan
            mae_esn = np.nan
            print(f"Предупреждение: нулевая дисперсия целевого ряда для {name} (ESN)")

        if np.std(y_true_ar) > 1e-12:
            nrmse_ar = du.nrmse(y_true_ar, pred_ar)
            mae_ar = du.mae_metric(y_true_ar, pred_ar)
        else:
            nrmse_ar = np.nan
            mae_ar = np.nan
            print(f"Предупреждение: нулевая дисперсия целевого ряда для {name} (AR)")

        # Если получились inf/NaN, пропускаем добавление в all_results или заменяем на None
        if np.isinf(nrmse_esn) or np.isnan(nrmse_esn):
            print(f"Ошибка: NRMSE для {name} (ESN) = {nrmse_esn}, пропускаем")
            continue

        all_results.append({'Dataset': name, 'Model': 'ESN', 'NRMSE': nrmse_esn, 'MAE': mae_esn})
        all_results.append({'Dataset': name, 'Model': 'AR(15)', 'NRMSE': nrmse_ar, 'MAE': mae_ar})

        # График (первые 200 точек, чтобы было наглядно)
        plot_len = min(200, len(y_true_esn))
        plt.figure(figsize=(12,5))
        plt.plot(y_true_esn[:plot_len], label='Target', alpha=0.7)
        plt.plot(pred_esn[:plot_len], '--', label=f'ESN (NRMSE={nrmse_esn:.4f})')
        plt.plot(pred_ar[:plot_len], ':', label=f'AR(15) (NRMSE={nrmse_ar:.4f})')
        plt.title(f'Prediction on {name} (first {plot_len} steps)')
        plt.legend()
        plt.grid(True)
        fname = f'prediction_{name.lower().replace("-","_")}.pdf'
        plt.savefig(fname)
        plt.close()

    # Сохраняем таблицу
    df = pd.DataFrame(all_results)
    df.to_csv('prediction_results.csv', index=False)
    df.to_latex('prediction_table.tex', index=False, float_format="%.4f", encoding='utf-8')
    print(df.to_string(index=False))

def run_ablation_analysis():
    """Абляционный анализ: влияние параметров на NRMSE (NARMA-10) без построения графиков"""
    print("\n=== Абляционный анализ ===")
    u, y = du.get_narma10(n=5000, seed=SEED)
    split = int(0.8 * len(u))
    tr_u, te_u = u[:split], u[split:]
    tr_y, te_y = y[:split], y[split:]

    base = {
        'n_res': 300,
        'reservoir_type': 'random',
        'rho': 0.9,
        'leaking_rate': 0.5,
        'input_scaling': 0.2,
        'sparsity': 0.1,
        'beta': 1e-6,
        'washout': 200
    }

    param_ranges = {
        'rho': [0.7, 0.8, 0.9, 0.95, 0.99, 1.02],
        'leaking_rate': [0.1, 0.3, 0.5, 0.7, 1.0],
        'input_scaling': [0.05, 0.1, 0.2, 0.5, 1.0],
        'sparsity': [0.02, 0.05, 0.1, 0.2, 0.5],
        'beta': [1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
        'washout': [100, 200, 500, 1000]
    }

    results = []
    for param_name, values in param_ranges.items():
        for val in values:
            print(f"  {param_name} = {val}...", end='', flush=True)
            kwargs = base.copy()
            kwargs[param_name] = val
            try:
                esn = EchoStateNetwork(
                    n_inputs=1, n_res=kwargs['n_res'], n_outputs=1,
                    reservoir_type=kwargs['reservoir_type'],
                    rho=kwargs['rho'], leaking_rate=kwargs['leaking_rate'],
                    input_scaling=kwargs['input_scaling'], sparsity=kwargs['sparsity'],
                    beta=kwargs['beta'], seed=SEED
                )
                esn.fit(tr_u, tr_y, washout=kwargs['washout'])
                pred = esn.predict(te_u)
                min_len = min(len(pred), len(te_y))
                nrmse_val = du.nrmse(te_y[:min_len], pred[:min_len])
                # MC вычисляем на отдельном сигнале
                u_mc = np.random.RandomState(SEED).uniform(-1, 1, 3000)
                mc_val = compute_mc(esn, u_mc, max_delay=100)
                results.append({'param': param_name, 'value': val, 'NRMSE': nrmse_val, 'MC': mc_val})
                print(f" NRMSE={nrmse_val:.4f}, MC={mc_val:.2f}")
            except Exception as e:
                print(f" ошибка: {e}")
                results.append({'param': param_name, 'value': val, 'NRMSE': np.nan, 'MC': np.nan})

    df = pd.DataFrame(results)
    df.to_csv('ablation_results.csv', index=False)

    df_table = df[['param', 'value', 'NRMSE', 'MC']].copy()
    param_names = {
        'rho': r'$\rho$',
        'leaking_rate': r'$\alpha$',
        'input_scaling': '$s$',
        'sparsity': 'sparsity',
        'beta': r'$\beta$',
        'washout': 'washout'
    }
    df_table['param'] = df_table['param'].map(param_names)
    df_table = df_table.sort_values(['param', 'value'])
    # Форматируем числа: NRMSE до 4 знаков, MC до 2 знаков
    df_table['NRMSE'] = df_table['NRMSE'].map(lambda x: f"{x:.4f}" if not np.isnan(x) else '---')
    df_table['MC'] = df_table['MC'].map(lambda x: f"{x:.2f}" if not np.isnan(x) else '---')
    
    # Генерация содержимого таблицы
    with open('ablation_table_content.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcrr}\n")
        f.write("\\toprule\n")
        f.write("Параметр & Значение & NRMSE & MC \\\\\n")
        f.write("\\midrule\n")
        for _, row in df_table.iterrows():
            f.write(f"{row['param']} & {row['value']} & {row['NRMSE']} & {row['MC']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print("Таблица сохранена в ablation_table_content.tex")
    
    print("Абляционный анализ завершён. Результаты в ablation_results.csv и ablation_table.tex")


if __name__ == "__main__":
    print("Запуск эксперимента 1 (ESP проверка)...")
    run_esp_practical_verification()
    print("Запуск эксперимента 2 (MC для разных геометрий)...")
    run_mc_geometry_comparison()
    print("Запуск эксперимента 3 (предсказание рядов)...")
    run_time_series_prediction()
    print("Все эксперименты завершены.")
    run_ablation_analysis()