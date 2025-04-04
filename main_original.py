import pygame
import random
import heapq
import sys
import argparse
import os
import time
import imageio
from abc import ABC, abstractmethod

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    Para criar uma nova estratégia de jogador, basta herdar dessa classe e implementar o método escolher_alvo.
    """
    def __init__(self, position):
        self.position = position  # Posição no grid [x, y]
        self.cargo = 0            # Número de pacotes atualmente carregados
        self.battery = 70         # Nível da bateria

    @abstractmethod
    def escolher_alvo(self, world):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world para acesso a pacotes e metas.
        """
        pass

class DefaultPlayer(BasePlayer):
    """
    Implementação padrão do jogador.
    Se não estiver carregando pacotes (cargo == 0), escolhe o pacote mais próximo.
    Caso contrário, escolhe a meta (entrega) mais próxima.
    """
    def escolher_alvo(self, world):
        sx, sy = self.position
        # Se não estiver carregando pacote e houver pacotes disponíveis:
        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best
        else:
            # Se estiver carregando ou não houver mais pacotes, vai para a meta de entrega (se existir)
            if world.goals:
                best = None
                best_dist = float('inf')
                for goal in world.goals:
                    d = abs(goal[0] - sx) + abs(goal[1] - sy)
                    if d < best_dist:
                        best_dist = d
                        best = goal
                return best
            else:
                return None

# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Parâmetros do grid e janela
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        # Cria uma matriz 2D para planejamento de caminhos:
        # 0 = livre, 1 = obstáculo
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        # Geração de obstáculos com padrão de linha (assembly line)
        self.generate_obstacles()
        # Gera a lista de paredes a partir da matriz
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        # Número total de itens (pacotes) a serem entregues
        self.total_items = 4

        # Geração dos locais de coleta (pacotes)
        self.packages = []
        # Aqui geramos 5 locais para coleta, garantindo uma opção extra
        while len(self.packages) < self.total_items + 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        # Geração dos locais de entrega (metas)
        self.goals = []
        while len(self.goals) < self.total_items:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.goals and [x, y] not in self.packages:
                self.goals.append([x, y])

        # Cria o jogador usando a classe DefaultPlayer
        self.player = self.generate_player()

        # Coloca o recharger (recarga de bateria) próximo ao centro (região 3x3)
        self.recharger = self.generate_recharger()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")
        
        # Inicializa fontes para texto
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.font_bold = pygame.font.SysFont('Arial', 16, bold=True)

        # Carrega imagens para pacote, meta e recharger a partir de arquivos
        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        self.recharger_image = pygame.image.load("images/charging-station.png")
        self.recharger_image = pygame.transform.scale(self.recharger_image, (self.block_size, self.block_size))

        # Cores utilizadas para desenho (caso a imagem não seja usada)
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)
        
        # Cores para o indicador de bateria
        self.battery_full_color = (0, 200, 0)  # Verde
        self.battery_medium_color = (200, 200, 0)  # Amarelo
        self.battery_low_color = (200, 0, 0)  # Vermelho
        self.battery_bg_color = (50, 50, 50)  # Cinza escuro
        
        # Variáveis para mostrar alertas de emergência
        self.show_emergency_alert = False
        self.alert_start_time = 0

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem:
         - Cria vários segmentos horizontais curtos com lacunas.
         - Cria vários segmentos verticais curtos com lacunas.
         - Cria um obstáculo em bloco grande (4x4 ou 6x6) simulando uma estrutura de suporte.
        """
        # Barragens horizontais curtas:
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Barragens verticais curtas:
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Obstáculo em bloco grande: bloco de tamanho 4x4 ou 6x6.
        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        # Cria o jogador em uma célula livre que não seja de pacote ou meta.
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages and [x, y] not in self.goals:
                return DefaultPlayer([x, y])

    def generate_recharger(self):
        # Coloca o recharger próximo ao centro
        center = self.maze_size // 2
        while True:
            x = random.randint(center - 1, center + 1)
            y = random.randint(center - 1, center + 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages and [x, y] not in self.goals and [x, y] != self.player.position:
                return [x, y]

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None, steps=0, score=0):
        self.screen.fill(self.ground_color)
        # Desenha os obstáculos (paredes)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        # Desenha os locais de coleta (pacotes) utilizando a imagem
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        # Desenha os locais de entrega (metas) utilizando a imagem
        for goal in self.goals:
            x, y = goal
            self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        # Desenha o recharger utilizando a imagem
        if self.recharger:
            x, y = self.recharger
            self.screen.blit(self.recharger_image, (x * self.block_size, y * self.block_size))
        # Desenha o caminho, se fornecido
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        # Desenha o jogador (retângulo colorido)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        
        # Desenha o indicador de bateria e estatísticas
        self.draw_battery_indicator()
        self.draw_game_stats(steps, score)
        
        pygame.display.flip()
        
    def draw_battery_indicator(self):
        # Configuração do indicador de bateria
        battery_width = 150
        battery_height = 20
        x_pos = 10
        y_pos = 10
        border = 2
        
        # Determina a cor baseada no nível da bateria
        battery_percent = max(0, min(100, self.player.battery / 70 * 100))
        if battery_percent > 60:
            color = (0, 200, 0)  # Verde
        elif battery_percent > 30:
            color = (200, 200, 0)  # Amarelo
        else:
            color = (200, 0, 0)  # Vermelho
            
        # Desenha a borda do indicador
        border_rect = pygame.Rect(x_pos, y_pos, battery_width, battery_height)
        pygame.draw.rect(self.screen, (50, 50, 50), border_rect)
        
        # Desenha o nível atual da bateria
        fill_width = int((battery_width - 2 * border) * (battery_percent / 100))
        fill_rect = pygame.Rect(x_pos + border, y_pos + border, 
                               fill_width, battery_height - 2 * border)
        pygame.draw.rect(self.screen, color, fill_rect)
        
        # Adiciona o texto de porcentagem e valor da bateria
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 16)
            
        battery_text = f"Bateria: {self.player.battery}/70"
        text_surface = self.font.render(battery_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (x_pos + battery_width + 10, y_pos))
    
    def draw_game_stats(self, steps, score):
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 16)
            
        # Texto para passos
        steps_text = f"Passos: {steps}"
        steps_surface = self.font.render(steps_text, True, (0, 0, 0))
        self.screen.blit(steps_surface, (10, 40))
        
        # Texto para pontuação
        score_text = f"Pontuação: {score}"
        score_surface = self.font.render(score_text, True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 65))
        
        # Texto para carga
        cargo_text = f"Pacotes: {self.player.cargo}"
        cargo_surface = self.font.render(cargo_text, True, (0, 0, 0))
        self.screen.blit(cargo_surface, (10, 90))

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None, delay=100, record=False):
        self.world = World(seed)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = delay  # milissegundos entre movimentos
        self.path = []
        self.num_deliveries = 0  # contagem de entregas realizadas
        
        # Configurações para gravação de vídeo
        self.record = record
        self.frames = []  # Lista para armazenar os frames capturados

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def game_loop(self):
        # Cria diretório para salvar o vídeo se estiver em modo de gravação
        if self.record:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_dir = f"videos"
            os.makedirs(video_dir, exist_ok=True)
            video_filename = f"{video_dir}/game_recording_original_{timestamp}.gif"
            print(f"Gravando vídeo para: {video_filename}")
            
        # O jogo termina quando o número de entregas realizadas é igual ao total de itens.
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            # Utiliza a estratégia do jogador para escolher o alvo
            target = self.world.player.escolher_alvo(self.world)
            if target is None:
                self.running = False
                break

            self.path = self.astar(self.world.player.position, target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", target)
                self.running = False
                break

            # Segue o caminho calculado
            for pos in self.path:
                self.world.player.position = pos
                self.steps += 1
                # Consumo da bateria: -1 por movimento se bateria >= 0, caso contrário -5
                self.world.player.battery -= 1
                if self.world.player.battery >= 0:
                    self.score -= 1
                else:
                    self.score -= 5
                # Recarrega a bateria se estiver no recharger
                if self.world.recharger and pos == self.world.recharger:
                    self.world.player.battery = 60
                    print("Bateria recarregada!")
                self.world.draw_world(self.path, self.steps, self.score)
                
                # Captura o frame atual para o vídeo se estiver em modo de gravação
                if self.record:
                    # Converte a tela do pygame para um array numpy
                    frame = pygame.surfarray.array3d(self.world.screen)
                    # Transpõe a matriz para o formato correto
                    frame = frame.transpose([1, 0, 2])
                    self.frames.append(frame)
                
                pygame.time.wait(self.delay)

            # Ao chegar ao alvo, processa a coleta ou entrega:
            if self.world.player.position == target:
                # Se for local de coleta, pega o pacote.
                if target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(target)
                    print("Pacote coletado em", target, "Cargo agora:", self.world.player.cargo)
                # Se for local de entrega e o jogador tiver pelo menos um pacote, entrega.
                elif target in self.world.goals and self.world.player.cargo > 0:
                    self.world.player.cargo -= 1
                    self.num_deliveries += 1
                    self.world.goals.remove(target)
                    self.score += 50
                    print("Pacote entregue em", target, "Cargo agora:", self.world.player.cargo)
                # Atualiza o display após coleta/entrega
                self.world.draw_world(self.path, self.steps, self.score)
                
                # Captura também esse frame
                if self.record:
                    frame = pygame.surfarray.array3d(self.world.screen)
                    frame = frame.transpose([1, 0, 2])
                    self.frames.append(frame)
                    
            print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, Bateria: {self.world.player.battery}, Entregas: {self.num_deliveries}")

        print("Fim de jogo!")
        print("Pontuação final:", self.score)
        print("Total de passos:", self.steps)
        
        # Salva o vídeo gravado
        if self.record and self.frames:
            print(f"Salvando vídeo do jogo...")
            # Reduce FPS if too many frames (limit to ~15 seconds at 24fps)
            total_frames = len(self.frames)
            fps = min(24, max(10, total_frames // 15))
            
            # Salva o GIF
            imageio.mimsave(video_filename, self.frames, fps=fps)
            print(f"Vídeo salvo em: {video_filename}")
        
        pygame.quit()

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delivery Bot: Navegue no grid, colete pacotes e realize entregas."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Valor do seed para recriar o mesmo mundo (opcional)."
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Peso do jogador (não utilizado na versão original, apenas para compatibilidade)."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Delay entre movimentos em milissegundos."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Grava o vídeo do jogo."
    )
    args = parser.parse_args()
    
    maze = Maze(seed=args.seed, delay=args.delay, record=args.record)
    maze.game_loop()

