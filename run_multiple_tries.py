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

def run_game_with_seed(seed, try_number, weight, delay=None):
    """Executa o jogo com uma seed específica e retorna o output"""
    try:
        cmd = ['python', 'main.py', '--seed', str(seed), '--weight', str(weight)]
        if delay is not None:
            cmd.extend(['--delay', str(delay)])
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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
    
    # Paleta de cores fixada para heatmaps, cobrindo a escala fixa de -620 até 200
    fixed_cmap = plt.cm.get_cmap('YlOrRd')
    fixed_vmin = -200
    fixed_vmax = 150
    
    # 1. Box Plot de Pontuações por Seed
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='seed', y='score')
    plt.title('Distribuição de Pontuações por Mapa')
    plt.xlabel('Seed do Mapa')
    plt.ylabel('Pontuação')
    plt.savefig(os.path.join(output_dir, 'scores_by_seed.png'))
    plt.close()
    
    # 2. Box Plot de Pontuações por Peso
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='weight', y='score')
    plt.title('Distribuição de Pontuações por Peso')
    plt.xlabel('Peso do Jogador')
    plt.ylabel('Pontuação')
    plt.savefig(os.path.join(output_dir, 'scores_by_weight.png'))
    plt.close()
    
    # 3. Heatmap de Pontuações (Seed vs Peso)
    try:
        pivot_df = results_df.pivot_table(values='score', index='seed', columns='weight', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        # Usar escala fixa para o heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap=fixed_cmap, vmin=fixed_vmin, vmax=fixed_vmax)
        plt.title('Pontuação Média por Mapa e Peso (Escala fixada: -200 a 150)')
        plt.xlabel('Peso do Jogador')
        plt.ylabel('Seed do Mapa')
        plt.savefig(os.path.join(output_dir, 'score_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Erro ao criar heatmap: {e}")
    
    # 4. Gráfico de Linha de Pontuações ao Longo das Tentativas
    plt.figure(figsize=(12, 6))
    for seed in results_df['seed'].unique():
        seed_data = results_df[results_df['seed'] == seed]
        plt.plot(seed_data['try_number'], seed_data['score'], label=f'Mapa {seed}')
    plt.title('Evolução da Pontuação ao Longo das Tentativas')
    plt.xlabel('Número da Tentativa')
    plt.ylabel('Pontuação')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'score_evolution.png'))
    plt.close()
    
    # 5. Gráfico de Barras de Média de Pontuação por Peso
    plt.figure(figsize=(12, 6))
    weight_means = results_df.groupby('weight')['score'].mean()
    
    # Destacar barras com pontuação média positiva
    colors = ['green' if score > 0 else 'red' for score in weight_means.values]
    
    weight_means.plot(kind='bar', color=colors)
    plt.title('Pontuação Média por Peso do Jogador')
    plt.xlabel('Peso do Jogador')
    plt.ylabel('Pontuação Média')
    
    # Adicionar linha horizontal no zero para referência
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Ajustar o limite do eixo y para manter consistência com a escala fixa
    plt.ylim(fixed_vmin, fixed_vmax)
    
    plt.savefig(os.path.join(output_dir, 'average_score_by_weight.png'))
    plt.close()
    
    # 6. Gráfico de Bateria Final por Peso
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='weight', y='battery')
    plt.title('Distribuição da Bateria Final por Peso')
    plt.xlabel('Peso do Jogador')
    plt.ylabel('Bateria Final')
    plt.savefig(os.path.join(output_dir, 'final_battery_by_weight.png'))
    plt.close()
    
    # 7. Gráfico de Bateria Final por Seed
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='seed', y='battery')
    plt.title('Distribuição da Bateria Final por Mapa')
    plt.xlabel('Seed do Mapa')
    plt.ylabel('Bateria Final')
    plt.savefig(os.path.join(output_dir, 'final_battery_by_seed.png'))
    plt.close()
    
    # 8. Heatmap de Bateria Final (Seed vs Peso)
    try:
        battery_pivot = results_df.pivot_table(values='battery', index='seed', columns='weight', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(battery_pivot, annot=True, fmt='.0f', cmap='Blues')
        plt.title('Bateria Final Média por Mapa e Peso')
        plt.xlabel('Peso do Jogador')
        plt.ylabel('Seed do Mapa')
        plt.savefig(os.path.join(output_dir, 'battery_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Erro ao criar heatmap de bateria: {e}")
    
    # 9. Gráfico separado para destacar os melhores pesos
    try:
        # Identificar pesos que tiveram pontuações positivas em todos os mapas
        best_weights = []
        for weight in results_df['weight'].unique():
            weight_data = results_df[results_df['weight'] == weight]
            if all(weight_data['score'] > 0):
                best_weights.append(weight)
                
        if best_weights:
            # Informar quais pesos foram bons para todos os mapas
            print(f"Pesos com resultados positivos em todos os mapas: {best_weights}")
            
            # Salvar essa informação em um arquivo de texto
            with open(os.path.join(output_dir, 'best_weights.txt'), 'w') as f:
                f.write("Pesos com resultados positivos em todos os mapas:\n")
                for weight in best_weights:
                    f.write(f"- {weight}\n")
                    
            # Filtrar apenas os pesos bons
            best_df = results_df[results_df['weight'].isin(best_weights)]
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=best_df, x='weight', y='score')
            plt.title('Distribuição de Pontuações dos Melhores Pesos')
            plt.xlabel('Peso do Jogador (Positivo em Todos Mapas)')
            plt.ylabel('Pontuação')
            plt.savefig(os.path.join(output_dir, 'best_weights_scores.png'))
            plt.close()
    except Exception as e:
        print(f"Erro ao criar gráfico de melhores pesos: {e}")
        
    # 10. Comparação de peso vs passos (tempo para solução)
    plt.figure(figsize=(12, 6))
    step_pivot = results_df.pivot_table(values='steps', index='seed', columns='weight', aggfunc='mean')
    sns.heatmap(step_pivot, annot=True, fmt='.0f', cmap='Greens', vmin=60, vmax=250)
    plt.title('Média de Passos por Mapa e Peso (Escala fixada: 60 a 250)')
    plt.xlabel('Peso do Jogador')
    plt.ylabel('Seed do Mapa')
    plt.savefig(os.path.join(output_dir, 'steps_heatmap.png'))
    plt.close()
    
    # 11. NOVOS GRÁFICOS DE CORRELAÇÃO
    
    # Gráfico de correlação: Bateria Final vs Score
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='battery', y='score', hue='weight', palette='viridis', s=100, alpha=0.7)
    plt.title('Correlação entre Bateria Final e Pontuação')
    plt.xlabel('Bateria Final')
    plt.ylabel('Pontuação')
    # Adicionar linha de tendência
    sns.regplot(data=results_df, x='battery', y='score', scatter=False, color='red')
    plt.savefig(os.path.join(output_dir, 'correlation_battery_score.png'))
    plt.close()
    
    # Gráfico de correlação: Passos vs Score 
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='steps', y='score', hue='weight', palette='viridis', s=100, alpha=0.7)
    plt.title('Correlação entre Número de Passos e Pontuação')
    plt.xlabel('Número de Passos')
    plt.ylabel('Pontuação')
    # Adicionar linha de tendência
    sns.regplot(data=results_df, x='steps', y='score', scatter=False, color='red')
    # Definir limites do eixo y entre -150 e 150
    plt.ylim(-150, 150)
    plt.savefig(os.path.join(output_dir, 'correlation_steps_score.png'))
    plt.close()
    
    # Gráfico de correlação: Bateria vs Passos
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='battery', y='steps', hue='weight', palette='viridis', s=100, alpha=0.7)
    plt.title('Correlação entre Bateria Final e Número de Passos')
    plt.xlabel('Bateria Final')
    plt.ylabel('Número de Passos')
    # Adicionar linha de tendência
    sns.regplot(data=results_df, x='battery', y='steps', scatter=False, color='red')
    plt.savefig(os.path.join(output_dir, 'correlation_battery_steps.png'))
    plt.close()
    
    # Matriz de correlação entre todas as variáveis numéricas
    plt.figure(figsize=(12, 10))
    correlation_matrix = results_df[['weight', 'score', 'steps', 'battery']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Matriz de Correlação entre Variáveis')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 3D plot: Bateria, Passos e Score
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Valores p/ scatter plot 3D
        xs = results_df['battery']
        ys = results_df['steps']
        zs = results_df['score']
        
        # Colorir por peso
        weights = results_df['weight']
        
        sc = ax.scatter(xs, ys, zs, c=weights, marker='o', s=50, cmap='viridis', alpha=0.7)
        
        ax.set_xlabel('Bateria Final')
        ax.set_ylabel('Número de Passos')
        ax.set_zlabel('Pontuação')
        ax.set_title('Relação 3D: Bateria, Passos e Pontuação')
        
        # Adicionar barra de cores para indicar o peso
        cbar = plt.colorbar(sc)
        cbar.set_label('Peso do Jogador')
        
        plt.savefig(os.path.join(output_dir, 'correlation_3d.png'))
        plt.close()
    except Exception as e:
        print(f"Erro ao criar gráfico 3D: {e}")
    
    # Pairplot - matriz de gráficos de dispersão para todas as combinações
    try:
        sns.pairplot(results_df[['weight', 'score', 'steps', 'battery']], diag_kind='kde')
        plt.suptitle('Relações entre Peso, Pontuação, Passos e Bateria', y=1.02)
        plt.savefig(os.path.join(output_dir, 'pairplot.png'))
        plt.close()
    except Exception as e:
        print(f"Erro ao criar pairplot: {e}")

def main():
    # Configuração
    num_tries = 1  # Número de tentativas por seed
    num_seeds = 10  # Número de seeds diferentes para tentar (usado apenas no modo random)
    weights = np.arange(2,4, 0.2)  # Pesos de 0.1 a 10  com passo de 0.2
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
                
                # Executar múltiplas tentativas para cada peso
                for weight in weights:
                    f.write(f"\nPeso do Jogador: {weight}\n")
                    f.write("-" * 20 + "\n")
                    
                    for try_num in range(1, num_tries + 1):
                        print(f"Executando seed {seed}, peso {weight}, tentativa {try_num}/{num_tries}")
                        result = run_game_with_seed(seed, try_num, weight, fixed_delay)
                        
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