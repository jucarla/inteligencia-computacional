import pygame
import random
import heapq
import sys
import argparse
import os
import time
import imageio
from abc import ABC, abstractmethod

# Importamos OpenCV apenas se disponível
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

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
    
class SmartBatteryPlayer(BasePlayer):
    """
    Implementação inteligente do jogador que usa análise de custo-benefício e thresholds dinâmicos.
    - Calcula o valor de cada ação baseado em distância e pontuação
    - Ajusta thresholds de bateria baseado na distância do recharger
    - Prioriza entregas próximas mesmo com bateria baixa
    - Considera pegar múltiplos pacotes no caminho
    - Considera custo de recarga vs benefício da entrega
    """
    def __init__(self, position, weight=2.8):
        super().__init__(position)
        self.weight = weight  # Peso para ajustar os parâmetros do jogador
        self.base_battery_threshold = 25 * weight  # Limite base para recarregar
        self.base_min_battery = 15 * weight       # Bateria mínima base
        self.delivery_points = 50        # Pontos por entrega
        self.step_cost = 1               # Custo de um passo normal
        self.no_battery_step_cost = 5    # Custo de um passo sem bateria
        self.max_cargo = 4              # Máximo de pacotes que pode carregar
        self.nearby_package_range = 5 * weight    # Distância para considerar pacotes próximos
        self.max_path_deviation = 3 * weight      # Desvio máximo permitido do caminho direto

    def distance_to(self, pos1, pos2):
        """Calcula a distância de Manhattan entre duas posições"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_on_path(self, point, start, end, max_deviation=None):
        """Verifica se um ponto está próximo ao caminho entre start e end"""
        if max_deviation is None:
            max_deviation = self.max_path_deviation

        # Calcula a distância direta entre start e end
        direct_dist = self.distance_to(start, end)
        
        # Calcula a distância passando pelo ponto
        detour_dist = self.distance_to(start, point) + self.distance_to(point, end)
        
        # Verifica se o ponto está dentro de um retângulo expandido entre start e end
        min_x = min(start[0], end[0]) - max_deviation
        max_x = max(start[0], end[0]) + max_deviation
        min_y = min(start[1], end[1]) - max_deviation
        max_y = max(start[1], end[1]) + max_deviation
        
        in_bounds = (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y)
        
        # Considera que está no caminho se o desvio for aceitável E estiver dentro dos limites
        return detour_dist <= direct_dist + max_deviation and in_bounds

    def find_nearby_packages(self, current_pos, world):
        """Encontra pacotes próximos que valem a pena pegar"""
        nearby_packages = []
        for pkg in world.packages:
            pkg_dist = self.distance_to(current_pos, pkg)
            if pkg_dist <= self.nearby_package_range:
                # Calcula o valor do pacote baseado na distância e possíveis entregas
                best_goal_dist = float('inf')
                for goal in world.goals:
                    goal_dist = self.distance_to(pkg, goal)
                    best_goal_dist = min(best_goal_dist, goal_dist)
                
                # Valor do pacote é maior quanto mais próximo estiver e quanto mais próxima for a entrega
                pkg_value = self.delivery_points - (pkg_dist * 2) - (best_goal_dist * 0.5)
                nearby_packages.append((pkg_value, pkg))
        
        # Retorna pacotes ordenados por valor
        return sorted(nearby_packages, reverse=True)

    def calculate_path_value(self, current_pos, target_pos, world):
        """Calcula o valor do caminho considerando pacotes que podem ser pegos no trajeto"""
        base_value = self.calculate_delivery_value(current_pos, target_pos, world.recharger)
        extra_value = 0
        available_cargo = self.max_cargo - self.cargo

        if available_cargo > 0:
            # Procura pacotes que estão no caminho
            for pkg in world.packages:
                if self.is_on_path(pkg, current_pos, target_pos):
                    # Para cada pacote no caminho, procura a entrega mais próxima
                    best_goal_dist = float('inf')
                    for goal in world.goals:
                        goal_dist = self.distance_to(pkg, goal)
                        if goal_dist < best_goal_dist:
                            best_goal_dist = goal_dist
                    
                    # Adiciona valor extra considerando a proximidade do pacote e da possível entrega
                    pkg_dist = self.distance_to(current_pos, pkg)
                    path_alignment = 1.0 - (pkg_dist / (self.distance_to(current_pos, target_pos) + 1))
                    extra_value += (self.delivery_points * path_alignment) - (best_goal_dist * 0.5)
                    
                    available_cargo -= 1
                    if available_cargo == 0:
                        break

        return base_value + extra_value

    def is_last_delivery(self, world):
        """Verifica se estamos na última entrega possível"""
        return len(world.goals) == 1

    def calculate_delivery_value(self, current_pos, goal_pos, recharger_pos):
        """Calcula o valor líquido de uma entrega considerando custos"""
        # Distância até a entrega
        delivery_dist = self.distance_to(current_pos, goal_pos)
        
        # Distância até o recharger após a entrega
        recharger_dist = self.distance_to(goal_pos, recharger_pos)
        
        # Custo total de bateria e pontos
        total_steps = delivery_dist + recharger_dist
        battery_cost = total_steps
        
        # Se não tiver bateria suficiente, calcula custo extra
        if self.battery < delivery_dist:
            battery_cost += (delivery_dist - self.battery) * 4  # 4 pontos extras por passo sem bateria
        
        # Valor líquido = pontos da entrega - custos
        return self.delivery_points - battery_cost - total_steps

    def get_dynamic_thresholds(self, recharger_dist):
        """Ajusta thresholds baseado na distância do recharger"""
        # Quanto mais longe do recharger, mais conservador com a bateria
        battery_threshold = self.base_battery_threshold + (recharger_dist // 2)
        min_battery = self.base_min_battery + (recharger_dist // 3)
        return min(battery_threshold, 60), min(min_battery, 40)

    def should_try_last_delivery(self, current_pos, goal_pos, recharger_pos, world, need_package=False):
        """
        Decide se deve tentar fazer a última entrega diretamente ou recarregar primeiro.
        """
        delivery_dist = self.distance_to(current_pos, goal_pos)
        if need_package:  # Se precisar pegar pacote, considera a menor distância possível
            min_pkg_dist = float('inf')
            for pkg in world.packages:
                pkg_dist = self.distance_to(current_pos, pkg)
                goal_dist = self.distance_to(pkg, goal_pos)
                total_dist = pkg_dist + goal_dist
                min_pkg_dist = min(min_pkg_dist, total_dist)
            delivery_dist = min_pkg_dist if min_pkg_dist != float('inf') else delivery_dist

        # Se já estiver no recharger, verifica se tem bateria suficiente para a entrega
        if current_pos == recharger_pos:
            return self.battery >= delivery_dist

        recharger_dist = self.distance_to(current_pos, recharger_pos)
        # Para a última entrega, apenas verifica se tem bateria suficiente para chegar até o objetivo
        # sem considerar a volta para o recharger (como estava originalmente)
        return (self.battery >= delivery_dist)

    def escolher_alvo(self, world):
        if not world.recharger:
            return None

        current_pos = self.position
        recharger_dist = self.distance_to(current_pos, world.recharger)
        battery_threshold, min_battery = self.get_dynamic_thresholds(recharger_dist)

        # Verifica se é a última entrega
        is_last = self.is_last_delivery(world)
        
        # Condição de recarga imediata: bateria igual ou menor à distância para recharger
        if self.battery <= recharger_dist:
            print(f"[ALERTA] RECARGA IMEDIATA! Bateria: {self.battery}, Distância ao recharger: {recharger_dist}")
            world.show_emergency_alert = True  # Sinaliza para mostrar alerta na tela
            world.alert_start_time = pygame.time.get_ticks()  # Marca o tempo de início do alerta
            return world.recharger
            
        # Se for a última entrega
        if is_last:
            goal = world.goals[0]  # Sabemos que só tem uma meta
            
            if self.cargo > 0:  # Se já tem o pacote
                # Verifica se vale a pena tentar entrega direta
                if self.should_try_last_delivery(current_pos, goal, world.recharger, world):
                    print(f"Última entrega: Indo entregar com pacote! Bateria: {self.battery}, Distância: {self.distance_to(current_pos, goal)}")
                    return goal
                else:
                    print(f"Última entrega: Precisa recarregar antes! Bateria: {self.battery}, Dist.Recharger: {recharger_dist}")
                    return world.recharger

            elif world.packages:  # Se precisa pegar um pacote
                # Primeiro verifica se vale a pena tentar entrega direta
                if self.should_try_last_delivery(current_pos, goal, world.recharger, world, need_package=True):
                    # Encontra o melhor pacote considerando o caminho total
                    best_pkg = None
                    best_total_dist = float('inf')
                    
                    for pkg in world.packages:
                        pkg_dist = self.distance_to(current_pos, pkg)
                        goal_dist = self.distance_to(pkg, goal)
                        total_dist = pkg_dist + goal_dist
                        
                        if total_dist < best_total_dist and self.battery >= total_dist:
                            best_total_dist = total_dist
                            best_pkg = pkg
                    
                    if best_pkg:
                        print(f"Última entrega: Indo pegar pacote! Bateria: {self.battery}, Distância total: {best_total_dist}")
                        return best_pkg
                
                # Se não dá para fazer entrega direta, tenta recarregar
                print(f"Última entrega: Precisa recarregar antes! Bateria: {self.battery}, Dist.Recharger: {recharger_dist}")
                return world.recharger
                
            print(f"Última entrega: Não encontrou caminho viável! Bateria: {self.battery}")
            return None

        # LÓGICA PARA ENTREGAS NORMAIS (não-finais) - Evitar bateria negativa
        
        # Condição de bateria crítica para entregas normais: 
        # se bateria for menor ou igual à distância para recharger + 5
        # Sempre vai para o carregador, a não ser que esteja muito próximo do objetivo (raio <= 3)
        if self.battery <= recharger_dist + 5:
            # Verifica se há algum objetivo (pacote ou entrega) muito próximo
            muito_proximo = False
            
            # Se estiver carregando pacotes, verifica se há alguma entrega muito próxima
            if self.cargo > 0 and world.goals:
                for goal in world.goals:
                    if self.distance_to(current_pos, goal) <= 3:
                        muito_proximo = True
                        print(f"[ENTREGA PRÓXIMA] Completando entrega antes de recarregar! Bateria: {self.battery}, Distância: {self.distance_to(current_pos, goal)}")
                        return goal
            
            # Se não estiver carregando pacotes, verifica se há algum pacote muito próximo
            elif world.packages:
                for pkg in world.packages:
                    if self.distance_to(current_pos, pkg) <= 3:
                        muito_proximo = True
                        print(f"[PACOTE PRÓXIMO] Coletando pacote antes de recarregar! Bateria: {self.battery}, Distância: {self.distance_to(current_pos, pkg)}")
                        return pkg
            
            # Se não tiver objetivo muito próximo, vai para o recharger
            if not muito_proximo:
                print(f"[ALERTA] Bateria crítica! Indo recarregar. Bateria: {self.battery}, Distância ao recharger: {recharger_dist}")
                return world.recharger

        # Se não for a última entrega, aplica regra de recarga oportunista
        if recharger_dist <= 3 and self.battery < 45:
            return world.recharger

        # Primeiro, verifica se há pacotes muito próximos que valem a pena pegar
        if self.cargo < self.max_cargo:
            nearby_packages = self.find_nearby_packages(current_pos, world)
            for value, pkg in nearby_packages:
                pkg_dist = self.distance_to(current_pos, pkg)
                # Se o pacote estiver próximo e tivermos bateria suficiente
                if pkg_dist <= self.nearby_package_range and (self.battery >= pkg_dist or pkg_dist <= 3):
                    return pkg

        # Se estiver carregando pacotes
        if self.cargo > 0:
            if world.goals:
                best_goal = None
                best_value = float('-inf')
                
                for goal in world.goals:
                    value = self.calculate_path_value(current_pos, goal, world)
                    delivery_dist = self.distance_to(current_pos, goal)
                    
                    # Para entregas normais (não-finais):
                    # 1. Se o alvo estiver muito próximo (<=3), podemos ir diretamente
                    # 2. OU se a bateria for suficiente para o objetivo + distância para o recharger + 5 (margem)
                    recharger_dist_from_goal = self.distance_to(goal, world.recharger)
                    total_dist_needed = delivery_dist + recharger_dist_from_goal
                    
                    if delivery_dist <= 3 or self.battery >= total_dist_needed + 5:
                        if value > best_value:
                            best_value = value
                            best_goal = goal
                
                if best_goal:
                    return best_goal

            # Se não encontrou destino seguro, recarrega
            print(f"Nenhuma entrega segura encontrada. Indo recarregar. Bateria: {self.battery}")
            return world.recharger

        # Se não estiver carregando ou puder pegar mais pacotes
        elif world.packages:
            best_pkg = None
            best_value = float('-inf')
            
            for pkg in world.packages:
                pkg_dist = self.distance_to(current_pos, pkg)
                
                # Para coleta normal (não-final):
                # 1. Se o pacote estiver muito próximo (<=3), podemos ir diretamente
                # 2. OU se a bateria for suficiente para o pacote + distância para o recharger + 5 (margem)
                pkg_to_recharger = self.distance_to(pkg, world.recharger)
                total_dist_needed = pkg_dist + pkg_to_recharger
                
                if pkg_dist <= 3 or self.battery >= total_dist_needed + 5:
                    value = self.calculate_path_value(current_pos, pkg, world)
                    if value > best_value:
                        best_value = value
                        best_pkg = pkg
            
            if best_pkg:
                return best_pkg

            # Se não encontrou pacote seguro, recarrega
            print(f"Nenhum pacote seguro encontrado. Indo recarregar. Bateria: {self.battery}")
            return world.recharger

        if self.battery < min_battery:
            return world.recharger

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

        # Cria o jogador usando a classe SmartBatteryPlayer (pode ser substituído por outra implementação)
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
                return SmartBatteryPlayer([x, y])

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
        
        # Desenha o limite mínimo de bateria
        if hasattr(self.player, 'base_min_battery'):
            min_battery = min(70, self.player.base_min_battery)
            min_x = x_pos + border + int((battery_width - 2 * border) * (min_battery / 70))
            pygame.draw.line(self.screen, (200, 0, 0), 
                           (min_x, y_pos), (min_x, y_pos + battery_height), 2)
    
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
        
        # Informação sobre recharge
        if hasattr(self.player, 'distance_to') and self.recharger:
            recharge_dist = self.player.distance_to(self.player.position, self.recharger)
            recharge_text = f"Dist. ao Recharger: {recharge_dist}"
            recharge_surface = self.font.render(recharge_text, True, (0, 0, 0))
            self.screen.blit(recharge_surface, (10, 115))

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None, delay=100, record=False, video_format='gif', pathfinding_algorithm='dijkstra'):
        self.world = World(seed)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = delay  # milissegundos entre movimentos
        self.path = []
        self.num_deliveries = 0  # contagem de entregas realizadas
        
        # Configurações para gravação de vídeo
        self.record = record
        self.video_format = video_format
        self.frames = []  # Lista para armazenar os frames capturados
        
        # Algoritmo de pathfinding
        self.pathfinding_algorithm = pathfinding_algorithm

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def greedy_best_first(self, start, goal):
        maze = self.world.map                                                       #Define o grid
        size = self.world.maze_size                                                 #Define o tamanho do grid
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]                              #Define os nós vizinhos

        open_set = []                                                               #Cria fila de prioridade de nós
        heapq.heappush(open_set, (self.heuristic(start, goal), tuple(start)))       #Prioriza utilizando apenas a distância de Manhattan
        came_from = {}                                                              #Dicionário para relacionar o nó anterior a cada nó
        visited = set()                                                             #Cria um set para os nós já visitados

        while open_set:                                                             
            current = heapq.heappop(open_set)[1]                                    #Retira o nó de maior prioridade no set

            if list(current) == goal:                                               #Se alcançar o destino, recria o caminho feito
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.reverse()
                return path

            visited.add(current)                                                    #Adiciona o nó ao set de visitados

            for dx, dy in neighbors:                                                #Analisa os nós vizinhos
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:             #Verifica se o vizinho está dentro dos limites do grid
                    if maze[neighbor[1]][neighbor[0]] == 1:                         #Verifica se é uma parede (obstáculo). Se for, ignore
                        continue                                                    
                else:                                                               #Se a posição for fora do mapa, ignore
                    continue

                if neighbor in visited:                                             #Se o nó já foi visitado, ignore
                    continue

                if neighbor not in [n[1] for n in open_set]:                        #Se o nó não foi adicionado ao set ainda, adicione
                    came_from[neighbor] = current                                   #Adiciona o nó anterior do vizinho
                    heapq.heappush(open_set, (self.heuristic(neighbor, goal), neighbor))

        return []
    
    def dijkstra(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        oheap = []
        heapq.heappush(oheap, (0, tuple(start)))

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
                    heapq.heappush(oheap, (tentative_g, neighbor))
        return []
    
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
            video_filename = f"{video_dir}/game_recording_{timestamp}.{self.video_format}"
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

            # Seleciona o algoritmo de pathfinding baseado no parâmetro
            if self.pathfinding_algorithm == 'astar':
                self.path = self.astar(self.world.player.position, target)
            elif self.pathfinding_algorithm == 'greedy':
                self.path = self.greedy_best_first(self.world.player.position, target)
            elif self.pathfinding_algorithm == 'dijkstra':
                self.path = self.dijkstra(self.world.player.position, target)
            else:
                # Default para A* se o algoritmo não for reconhecido
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
        help="Peso para ajustar os parâmetros do SmartBatteryPlayer (0.1 a 2.0)."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Delay em milissegundos entre movimentos (padrão: 100)."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Grava o vídeo do jogo (opcional)."
    )
    parser.add_argument(
        "--pathfinding_algorithm",
        type=str,
        default="astar",
        choices=["astar", "greedy", "dijkstra"],
        help="Algoritmo de pathfinding (opções: astar, greedy, dijkstra)."
    )
    args = parser.parse_args()
    
    print(f"Iniciando jogo com: seed={args.seed}, weight={args.weight}, delay={args.delay}, algoritmo={args.pathfinding_algorithm}")
    
    maze = Maze(seed=args.seed, delay=args.delay, record=args.record, pathfinding_algorithm=args.pathfinding_algorithm)
    maze.world.player = SmartBatteryPlayer(maze.world.player.position, args.weight)
    maze.game_loop()

