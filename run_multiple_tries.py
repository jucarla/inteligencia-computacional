import subprocess
import os
import time
from datetime import datetime
import pygame
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from main import World

def create_map_screenshot(seed, output_dir):
    """Cria uma captura de tela do mapa inicial para uma seed específica"""
    # Inicializa pygame para cada mapa - isso é necessário porque main.py também usa pygame
    # e faz o quit() quando termina, então precisamos reinicializar para cada mapa
    pygame.init()
    world = World(seed)
    os.makedirs(output_dir, exist_ok=True)
    world.draw_world()
    screenshot_path = os.path.join(output_dir, f"map_{seed}.png")
    pygame.image.save(world.screen, screenshot_path)
    pygame.quit()
    return screenshot_path

def run_game_with_seed(seed, try_number, algorithm, delay=None):
    """Executa o jogo com uma seed específica e retorna o output"""
    try:
        # Fixando o peso em 2.8
        weight = 2.8
        cmd = ['python', 'main.py', '--seed', str(seed), '--weight', str(weight), '--pathfinding_algorithm', algorithm]
        if delay is not None:
            cmd.extend(['--delay', str(delay)])
            
        print(f"Executando comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Debug: imprimir o código de retorno e stderr se houver erro
        if result.returncode != 0:
            print(f"Erro ao executar o comando (código {result.returncode})")
            print(f"STDERR: {result.stderr}")
            return None
            
        output_lines = result.stdout.split('\n')
        final_score = None
        final_steps = None
        final_battery = None
        
        for line in output_lines:
            if "Pontuação final:" in line:
                final_score = int(line.split(":")[1].strip())
            elif "Total de passos:" in line:
                final_steps = int(line.split(":")[1].strip())
            elif "Bateria:" in line and "Entregas:" in line:
                # Extrai o valor da bateria da última linha de status
                final_battery = int(line.split("Bateria:")[1].split(",")[0].strip())
        
        return {
            'seed': seed,
            'try_number': try_number,
            'algorithm': algorithm,
            'weight': weight,
            'delay': delay if delay is not None else 100,  # Valor padrão se não especificado
            'score': final_score,
            'steps': final_steps,
            'battery': final_battery,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Erro ao executar jogo com seed {seed}: {str(e)}")
        return None

def generate_seeds(mode='random', num_seeds=10, start=None, end=None):
    """
    Gera seeds baseado no modo especificado
    mode: 'random' ou 'range'
    num_seeds: número de seeds para gerar (usado apenas no modo random)
    start: valor inicial do intervalo (usado apenas no modo range)
    end: valor final do intervalo (usado apenas no modo range)
    """
    if mode == 'random':
        return [random.randint(1, 10000) for _ in range(num_seeds)]
    elif mode == 'range':
        if start is None or end is None or start >= end:
            raise ValueError("No modo 'range', start e end devem ser especificados e start deve ser menor que end")
        return list(range(start, end + 1))
    else:
        raise ValueError("Modo deve ser 'random' ou 'range'")

def create_visualizations(results_df, output_dir):
    """Cria visualizações dos resultados"""
    # Verificação básica para garantir que temos dados
    if results_df.empty:
        print("Aviso: Sem dados para criar visualizações")
        return
        
    # Configuração do estilo
    plt.style.use('default')  # Usando estilo padrão do matplotlib
    sns.set_palette("husl")
    
    # Paleta de cores específica para os algoritmos
    algorithm_colors = {
        'astar': 'green',
        'greedy': 'orange',
        'dijkstra': 'blue'
    }
    
    # Gráfico de barras comparando algoritmos por seed
    plt.figure(figsize=(14, 8))
    
    # Preparar dados para o gráfico
    # Agrupar por seed e algoritmo e calcular a média da pontuação
    grouped_data = results_df.groupby(['seed', 'algorithm'])['score'].mean().reset_index()
    
    # Pivotear para ter seeds como índice e algoritmos como colunas
    pivot_data = grouped_data.pivot(index='seed', columns='algorithm', values='score')
    
    # Desenhar as barras agrupadas
    bar_width = 0.25
    seeds = pivot_data.index
    x = np.arange(len(seeds))
    
    # Verificar quais algoritmos estão disponíveis nos dados
    available_algorithms = pivot_data.columns.tolist()
    
    # Desenhar barras para cada algoritmo
    for i, algorithm in enumerate(available_algorithms):
        plt.bar(x + i*bar_width - bar_width, 
                pivot_data[algorithm], 
                bar_width, 
                label=algorithm.capitalize(), 
                color=algorithm_colors.get(algorithm, f'C{i}'))
    
    # Configurações do gráfico
    plt.xlabel('Seed do Mapa', fontsize=12)
    plt.ylabel('Pontuação Média', fontsize=12)
    plt.title('Comparação da Pontuação por Algoritmo de Pathfinding (Peso fixo: 2.8)', fontsize=14)
    plt.xticks(x, seeds)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, algorithm in enumerate(available_algorithms):
        for j, score in enumerate(pivot_data[algorithm]):
            plt.text(j + i*bar_width - bar_width, score + 5, f'{int(score)}', 
                     ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'))
    plt.close()
    
    # Boxplot da distribuição de pontuações por algoritmo
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='algorithm', y='score', palette=algorithm_colors)
    plt.title('Distribuição de Pontuações por Algoritmo de Pathfinding')
    plt.xlabel('Algoritmo')
    plt.ylabel('Pontuação')
    plt.savefig(os.path.join(output_dir, 'score_boxplot_by_algorithm.png'))
    plt.close()
    
    # Boxplot da distribuição de passos por algoritmo
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='algorithm', y='steps')
    plt.title('Distribuição de Passos por Algoritmo de Pathfinding')
    plt.xlabel('Algoritmo')
    plt.ylabel('Número de Passos')
    plt.savefig(os.path.join(output_dir, 'steps_boxplot_by_algorithm.png'))
    plt.close()
    
    # Boxplot da distribuição de bateria final por algoritmo
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='algorithm', y='battery')
    plt.title('Distribuição de Bateria Final por Algoritmo de Pathfinding')
    plt.xlabel('Algoritmo')
    plt.ylabel('Bateria Final')
    plt.savefig(os.path.join(output_dir, 'battery_boxplot_by_algorithm.png'))
    plt.close()
    
    # Scatter plot comparando passos vs score, colorizado por algoritmo
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='steps', y='score', hue='algorithm', palette=algorithm_colors, s=100, alpha=0.7)
    plt.title('Relação entre Número de Passos e Pontuação por Algoritmo')
    plt.xlabel('Número de Passos')
    plt.ylabel('Pontuação')
    plt.savefig(os.path.join(output_dir, 'steps_vs_score.png'))
    plt.close()

def main():
    # Configuração
    num_tries = 1  # Número de tentativas por seed
    num_seeds = 10  # Número de seeds diferentes para tentar (usado apenas no modo random)
    algorithms = ['astar', 'greedy', 'dijkstra']  # Algoritmos de pathfinding a testar
    fixed_delay = 1  # Delay para todas as tentativas (ms)
    output_dir = "results"
    map_dir = "maps"
    
    # Configuração da geração de seeds
    seed_mode = 'random'  # 'random' ou 'range'
    seed_start = 1  # Usado apenas no modo range
    seed_end = 10   # Usado apenas no modo range
    
    # Criar diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    
    # Gerar seeds baseado no modo
    #seeds = generate_seeds(mode=seed_mode, num_seeds=num_seeds, start=seed_start, end=seed_end)
    seeds = [8250, 1488, 4827, 9294, 3109, 3402, 4258, 2053, 6350, 6198]
    # Criar timestamp para nomear os arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar pasta específica para esta execução
    execution_dir = os.path.join(output_dir, f"execution_{timestamp}")
    charts_dir = os.path.join(execution_dir, "charts")
    os.makedirs(execution_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    # Criar arquivo de resultados com timestamp
    results_file = os.path.join(execution_dir, f"results.txt")
    
    # Lista para armazenar todos os resultados
    all_results = []
    
    try:
        with open(results_file, 'w') as f:
            f.write(f"Resultados do Jogo - Iniciado em {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Delay fixo: {fixed_delay}ms\n")
            f.write(f"Peso fixo: 2.8\n")
            f.write(f"Modo de geração de seeds: {seed_mode}\n")
            if seed_mode == 'range':
                f.write(f"Intervalo de seeds: {seed_start} a {seed_end}\n")
            else:
                f.write(f"Número de seeds: {num_seeds}\n")
            f.write("\n")
            
            for seed in seeds:
                f.write(f"Seed: {seed}\n")
                f.write("-" * 30 + "\n")
                
                # Criar e salvar screenshot do mapa
                map_path = create_map_screenshot(seed, map_dir)
                f.write(f"Screenshot do mapa salvo como: {map_path}\n")
                
                # Executar múltiplas tentativas para cada algoritmo
                for algorithm in algorithms:
                    f.write(f"\nAlgoritmo: {algorithm}\n")
                    f.write("-" * 20 + "\n")
                    
                    for try_num in range(1, num_tries + 1):
                        print(f"Executando seed {seed}, algoritmo {algorithm}, tentativa {try_num}/{num_tries}")
                        result = run_game_with_seed(seed, try_num, algorithm, fixed_delay)
                        
                        if result:
                            all_results.append(result)
                            f.write(f"Tentativa {try_num}:\n")
                            f.write(f"  Pontuação: {result['score']}\n")
                            f.write(f"  Passos: {result['steps']}\n")
                            f.write(f"  Bateria Final: {result['battery']}\n")
                            f.write(f"  Timestamp: {result['timestamp']}\n")
                            f.write("\n")
                
                f.write("\n")
    
        # Criar DataFrame com todos os resultados
        results_df = pd.DataFrame(all_results)
        
        # Criar visualizações
        create_visualizations(results_df, charts_dir)
        
        print(f"\nResultados salvos em: {results_file}")
        print(f"Screenshots dos mapas salvos em: {map_dir}")
        print(f"Gráficos salvos em: {charts_dir}")
        print(f"Seeds usados: {seeds}")
    
    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == "__main__":
    main() 