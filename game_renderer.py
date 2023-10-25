import pygame


class Styling:
    # color palette
    DARK_BLUE = (25, 25, 112)
    MEDIUM_BLUE = (70, 130, 180)
    DARK_RED = (139, 0, 0)
    MEDIUM_RED = (205, 92, 92)

    VERY_DARK_GRAY = (20, 20, 20)
    DARK_GRAY = (50, 50, 50)
    MEDIUM_GRAY = (100, 100, 100)
    LIGHT_GRAY = (150, 150, 150)

    WHITE = (255, 255, 255)

    # Basic Colors
    BACKGROUND_COLOR = VERY_DARK_GRAY
    FONT_COLOR = WHITE
    MATRIX_BORDER_COLOR = MEDIUM_GRAY

    AGENT1_COLOR = DARK_BLUE
    AGENT1_ALT_COLOR = MEDIUM_BLUE
    AGENT2_COLOR = DARK_RED
    AGENT2_ALT_COLOR = MEDIUM_RED

    MATRIX_BACKGROUND_COLOR = DARK_GRAY
    SCOREBOARD_BACKGROUND_COLOR = DARK_GRAY

    # Fonts
    FONT_NAME = "Arial"
    TITLE_FONT_SIZE = 40  # Larger font for the title
    DEFAULT_FONT_SIZE = 18  # Standard font size for other texts
    MATRIX_FONT_SIZE = 20  # Smaller font size for the matrix values

    # Matrix Styling
    MATRIX_CELL_BORDER_WIDTH = 1
    HIGHLIGHT_COLOR = MEDIUM_GRAY  # Updated from direct definition

    # Scoreboard Styling
    SCOREBOARD_HEADER_COLOR = DARK_GRAY
    SCOREBOARD_ROW_COLOR = DARK_GRAY
    SCOREBOARD_BORDER_WIDTH = 1
    SCOREBOARD_BORDER_COLOR = MEDIUM_GRAY

    # Headers & Titles
    HEADER_PADDING = 10  # Padding for headers from the top edge of their section

    # Paddings & Margins
    TITLE_PADDING_TOP = 30  # Padding from the top of the screen to the title
    SECTION_VERTICAL_MARGIN = 80  # Vertical space between different sections (like matrix and scoreboard)


class GameRenderer:
    def __init__(self, game_modes_instance):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(Styling.GAME_TITLE if hasattr(Styling, 'GAME_TITLE') else "Game Theory")

        self.game_modes = game_modes_instance
        self.game = self.game_modes.get_game()
        self.game_type = self.game_modes.game_type  # Storing the game type
        self.action_names = self.game["action_names"]

        num_actions = len(self.action_names)
        self.cell_width = 800 // (num_actions + 2)
        self.cell_height = 40

        # Initialize fonts
        self.font = pygame.font.SysFont(Styling.FONT_NAME, Styling.DEFAULT_FONT_SIZE, bold=False)
        self.font_bold = pygame.font.SysFont(Styling.FONT_NAME, Styling.DEFAULT_FONT_SIZE, bold=True)  # Bold variant
        self.title_font = pygame.font.SysFont(Styling.FONT_NAME, Styling.TITLE_FONT_SIZE, bold=True)  # Make title bold
        self.matrix_font = pygame.font.SysFont(Styling.FONT_NAME, Styling.MATRIX_FONT_SIZE, bold=False)

    def render(self, agent1_action, agent2_action, reward_agent1, reward_agent2,
               agent1_cumulative_reward, agent2_cumulative_reward, avg_reward_agent1, avg_reward_agent2,
               agent1_total_cumulative_reward, agent2_total_cumulative_reward,  # New parameters
               agent1_total_avg_reward, agent2_total_avg_reward,  # New parameters
               episode_counter, current_round):
        self.screen.fill(Styling.BACKGROUND_COLOR)

        # Title
        title = self.title_font.render(f"{self.game_type} - Episode {episode_counter}, Round {current_round}", True, Styling.FONT_COLOR)  # Display episode and round counters
        title_pos = title.get_rect(center=(400, Styling.TITLE_PADDING_TOP))
        self.screen.blit(title, title_pos.topleft)

        # Displaying Payoff Matrix
        matrix_start_y = title_pos.bottom + Styling.HEADER_PADDING
        self.display_payoff_matrix(agent1_action, agent2_action, matrix_start_y)

        # Displaying Scoreboard
        scoreboard_start_y = matrix_start_y + len(self.game["action_names"]) * self.cell_height + Styling.SECTION_VERTICAL_MARGIN
        self.display_scoreboard(agent1_action, agent2_action, reward_agent1, reward_agent2,
                                agent1_cumulative_reward, agent2_cumulative_reward,
                                avg_reward_agent1, avg_reward_agent2,
                                agent1_total_cumulative_reward, agent2_total_cumulative_reward,  # Pass new values
                                agent1_total_avg_reward, agent2_total_avg_reward,  # Pass new values
                                scoreboard_start_y)

        pygame.display.flip()
        pygame.time.wait(300)

    def display_payoff_matrix(self, agent1_action, agent2_action, start_y):
        matrix_width = (len(self.game["action_names"]) + 1) * self.cell_width
        matrix_start_x = (self.screen.get_width() - matrix_width) // 2
        HIGHLIGHT_COLOR = Styling.HIGHLIGHT_COLOR

        # Fill the background for matrix
        pygame.draw.rect(self.screen, Styling.MATRIX_BACKGROUND_COLOR,
                         (matrix_start_x, start_y, matrix_width, len(self.game["action_names"]) * self.cell_height))

        # Displaying the column headers
        for j, action_name in enumerate(self.game["action_names"]):
            # Define a background color for the cell
            cell_bg_color = Styling.MATRIX_BACKGROUND_COLOR
            if j == agent2_action:
                cell_bg_color = Styling.AGENT2_COLOR

            pygame.draw.rect(self.screen, cell_bg_color,
                             (matrix_start_x + (j + 1) * self.cell_width, start_y,
                              self.cell_width, self.cell_height))

            text = self.font.render(action_name, True, Styling.FONT_COLOR)
            text_rect = text.get_rect(
                center=(matrix_start_x + (j + 1.5) * self.cell_width, start_y + 0.5 * self.cell_height))
            self.screen.blit(text, text_rect.topleft)

        # Draw matrix
        for i, action_name_1 in enumerate(self.game["action_names"]):
            # Define a background color for the row
            row_bg_color = Styling.MATRIX_BACKGROUND_COLOR
            if i == agent1_action:
                row_bg_color = Styling.AGENT1_COLOR
            pygame.draw.rect(self.screen, row_bg_color,
                             (matrix_start_x, start_y + (i + 1) * self.cell_height,
                              self.cell_width, self.cell_height))
            for j, action_name_2 in enumerate(self.game["action_names"]):
                rect = pygame.Rect(matrix_start_x + (j + 1) * self.cell_width,
                                   start_y + (i + 1) * self.cell_height,
                                   self.cell_width, self.cell_height)

                # Explicitly fill each cell with the MATRIX_BACKGROUND_COLOR
                pygame.draw.rect(self.screen, Styling.MATRIX_BACKGROUND_COLOR, rect)

                # Highlighting the chosen cell based on both agents' actions
                if i == agent1_action and j == agent2_action:
                    pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect)

                pygame.draw.rect(self.screen, Styling.MATRIX_BORDER_COLOR, rect, Styling.MATRIX_CELL_BORDER_WIDTH)

                # Displaying the payoff values
                payoff = self.game_modes.get_payoff(i, j)
                payoff_text = f"{payoff[0]}, {payoff[1]}"
                text = self.matrix_font.render(payoff_text, True, Styling.FONT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

            # Displaying the row headers
            text = self.font.render(action_name_1, True, Styling.FONT_COLOR)
            text_rect = text.get_rect(
                center=(matrix_start_x + 0.5 * self.cell_width, start_y + (i + 1.5) * self.cell_height))
            self.screen.blit(text, text_rect.topleft)

    def display_scoreboard(self, agent1_action, agent2_action, reward_agent1, reward_agent2,
                           agent1_cumulative_reward, agent2_cumulative_reward, avg_reward_agent1, avg_reward_agent2,
                           agent1_total_cumulative_reward, agent2_total_cumulative_reward,
                           agent1_total_avg_reward, agent2_total_avg_reward, start_y):
        scoreboard_width = 3 * self.cell_width
        start_x = (self.screen.get_width() - scoreboard_width) // 2

        # Explicitly drawing the background for the scoreboard header
        header_rect = pygame.Rect(start_x, start_y, scoreboard_width, self.cell_height)
        pygame.draw.rect(self.screen, Styling.SCOREBOARD_HEADER_COLOR, header_rect)

        # Displaying the column headers
        headers = ["", "Agent 1", "Agent 2"]
        for i, header in enumerate(headers):
            color = Styling.FONT_COLOR
            if header == "Agent 1":
                color = Styling.AGENT1_ALT_COLOR
            elif header == "Agent 2":
                color = Styling.AGENT2_ALT_COLOR
            text = self.font_bold.render(header, True, color)
            # centering the text horizontally and vertically
            text_rect = text.get_rect(center=(start_x + (i + 0.5) * self.cell_width, start_y + 0.5 * self.cell_height))
            self.screen.blit(text, text_rect.topleft)

        # Row values
        rows = [
            ("Chosen Move", self.game["action_names"][agent1_action], self.game["action_names"][agent2_action]),
            ("Reward", reward_agent1, reward_agent2),
            ("", "", ""),
            ("Cum. Reward", agent1_cumulative_reward, agent2_cumulative_reward),
            ("Avg Reward", round(avg_reward_agent1, 1), round(avg_reward_agent2, 1)),
            ("", "", ""),
            ("Total Cum. Reward", agent1_total_cumulative_reward, agent2_total_cumulative_reward),
            ("Total Avg. Reward", round(agent1_total_avg_reward, 1), round(agent2_total_avg_reward, 1))
        ]

        # For rows, labels are rendered in bold, while values are rendered in normal font.
        for i, (label, value_agent1, value_agent2) in enumerate(rows):
            row_rect = pygame.Rect(start_x, start_y + (i + 1) * self.cell_height, scoreboard_width, self.cell_height)
            pygame.draw.rect(self.screen, Styling.SCOREBOARD_ROW_COLOR, row_rect)  # Explicitly setting row background
            pygame.draw.rect(self.screen, Styling.SCOREBOARD_BORDER_COLOR, row_rect,
                             Styling.SCOREBOARD_BORDER_WIDTH)  # Draw borders

            text = self.font_bold.render(label, True, Styling.FONT_COLOR)
            # centering the text horizontally and vertically
            text_rect = text.get_rect(center=(start_x + 0.5 * self.cell_width, row_rect.centery))
            self.screen.blit(text, text_rect.topleft)

            text_agent1 = self.font.render(str(value_agent1), True, Styling.FONT_COLOR)
            text_agent2 = self.font.render(str(value_agent2), True, Styling.FONT_COLOR)

            text_rect_agent1 = text_agent1.get_rect(center=(start_x + 1.5 * self.cell_width, row_rect.centery))
            text_rect_agent2 = text_agent2.get_rect(center=(start_x + 2.5 * self.cell_width, row_rect.centery))

            self.screen.blit(text_agent1, text_rect_agent1.topleft)
            self.screen.blit(text_agent2, text_rect_agent2.topleft)

        # Draw vertical grid lines for the scoreboard
        for i in range(1, 3):
            pygame.draw.line(self.screen, Styling.SCOREBOARD_BORDER_COLOR,
                             (start_x + i * self.cell_width, start_y),
                             (start_x + i * self.cell_width, start_y + 9 * self.cell_height))

    def close(self):
        pygame.quit()