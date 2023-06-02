import numpy as np
import random
import pygame
import sys
import math
import time
from functools import partial
from itertools import product
import textwrap


BOARD_ROWS = 6
BOARD_COLS = 7

#utility function to split multi-line text
def render_multiline_text_with_background(text, text_color, bg_color, font):
    lines = text.split('\n')
    surfaces = [font.render(line, True, text_color) for line in lines]
    max_width = max(surface.get_width() for surface in surfaces)
    total_height = sum(surface.get_height() for surface in surfaces)
    
    background = pygame.Surface((max_width, total_height), pygame.SRCALPHA)
    background.fill(bg_color)
    
    y_offset = 0
    for surface in surfaces:
        background.blit(surface, (0, y_offset))
        y_offset += surface.get_height()

    return background


#utility function to draw multi-line text
def draw_multiline_text(screen, surfaces, x, y):
    for surface in surfaces:
        screen.blit(surface, (x, y))
        y += surface.get_height()

# this is a dictionary of 16 colors accoring to RGB values 

COLORS = {
    'Sunshine Yellow': (255, 253, 55),
    'Canary Yellow': (255, 239, 0),
    'Goldenrod': (218, 165, 32),
    'Saffron': (244, 196, 48),
    'Citrine': (228, 208, 10),
    'Trombone': (210, 181, 91),
    'Light Yellow': (255, 255, 224),
    'Flax': (238, 220, 130),
    'Pale Goldenrod': (238, 232, 170),
    'Gold': (255, 215, 0),
    'Hunyadi Yellow': (232, 172, 65),
    'Bright Yellow': (255, 253, 1),
    'Greenish Yellow': (238, 234, 98),
    'Neon Yellow': (207, 255, 4),
    'Dark Yellow': (213, 182, 10),
    'Sand': (226, 202, 118)
}

#This function is used to display the color selection menu
def show_color_menu(screen, colors, font):
    color_buttons = []
    color_button_size = 50
    spacing = 20
    columns = 4
    rows = 4

    total_width = (columns * color_button_size) + ((columns - 1) * spacing)
    total_height = (rows * color_button_size) + ((rows - 1) * spacing)

    start_x = (screen.get_width() - total_width) // 2
    start_y = (screen.get_height() - total_height) // 2

    color_list = list(colors.items())

    for row in range(rows):
        for col in range(columns):
            idx = row * columns + col
            color_name, color = color_list[idx]
            button_rect = pygame.Rect(start_x + col * (color_button_size + spacing), start_y + row * (color_button_size + spacing), color_button_size, color_button_size)
            pygame.draw.rect(screen, color, button_rect)
            color_buttons.append((color_name, color, button_rect))

    message = "Click on a color to set the board color and start the game"
    message_surface = font.render(message, True, WHITE)
    message_rect = message_surface.get_rect(center=(screen.get_width() // 2, start_y - 50))
    screen.blit(message_surface, message_rect)

    return color_buttons





#GOLDENROD = (218, 165, 32)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER = 0
AI = 1
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4
# to initialize the board
def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board
# to put the piece at specified row, col
def drop_piece(board, row, col, piece):
	board[row][col] = piece
# to check if the selected location is free (valid)
def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r
# to change the orientation of the board to see it look like filled from the buttom to top
# where it is in fact filled from top to buttom.
def print_board(board):
	print(np.flip(board, 0))
    
def render_text_with_background(text, text_color, bg_color, font):
    text_surface = font.render(text, True, text_color)
    bg_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
    bg_surface.fill(bg_color)
    bg_surface.blit(text_surface, (0, 0))
    return bg_surface
# to get the user name
def get_user_input(screen, prompt, font, x, y, color):
    text = ""
    input_box = pygame.Rect(x, y, 200, 50)
    active = False
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = True
                else:
                    active = False
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        screen.fill((0, 0, 0))
        prompt_text = font.render(prompt, True, color)
        screen.blit(prompt_text, (x, y - 40))
        txt_surface = font.render(text, True, color)
        input_box.w = max(100, txt_surface.get_width() + 10)
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(screen, color, input_box, 2)
        pygame.display.flip()

    return text

#function to allow the user to select the first player

def choose_first_player(screen, player_name, agent_name, font, text_color):
    text1 = font.render(f"{player_name}", True, text_color)
    text2 = font.render(f"{agent_name}", True, text_color)
    instructions = "Select who is the first player: {} or {}. Please click on the name that you want to give the first turn.".format(player_name, agent_name)
    wrapped_instructions = textwrap.wrap(instructions, width=40)
    instruction_texts = [font.render(line, True, text_color) for line in wrapped_instructions]

    choice_made = False
    while not choice_made:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.fill(BLACK)
        for i, instruction_text in enumerate(instruction_texts):
            screen.blit(instruction_text, (width // 2 - instruction_text.get_width() // 2, (height // 4 - instruction_text.get_height() // 2) + i * instruction_text.get_height()))
        screen.blit(text1, (width // 4 - text1.get_width() // 2, height // 2 - text1.get_height() // 2))
        screen.blit(text2, (3 * width // 4 - text2.get_width() // 2, height // 2 - text2.get_height() // 2))
        pygame.display.update()

        pressed = pygame.mouse.get_pressed()
        if pressed[0]:  # Left mouse button clicked
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if width // 4 - text1.get_width() // 2 <= mouse_x <= width // 4 + text1.get_width() // 2 and height // 2 - text1.get_height() // 2 <= mouse_y <= height // 2 + text1.get_height() // 2:
                return PLAYER  # Return PLAYER instead of AI
            elif 3 * width // 4 - text2.get_width() // 2 <= mouse_x <= 3 * width // 4 + text2.get_width() // 2 and height // 2 - text2.get_height() // 2 <= mouse_y <= height // 2 + text2.get_height() // 2:
                return AI  # Return AI instead of PLAYER


# this function displays the options for the search depth and returns the chosen depth via buttons for each search depth option [1–5]            
def select_search_depth(screen, font, text_color):
    text = font.render("Select search depth for Minimax [1-5]:", True, text_color)
    buttons = [font.render(str(i), True, text_color) for i in range(1, 6)]

    choice_made = False
    while not choice_made:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.fill(BLACK)
        screen.blit(text, (width // 2 - text.get_width() // 2, height // 4 - text.get_height() // 2))

        for i, button in enumerate(buttons):
            screen.blit(button, (width // 6 * (i + 1) - button.get_width() // 2, height // 2 - button.get_height() // 2))

        pygame.display.update()

        pressed = pygame.mouse.get_pressed()
        if pressed[0]:  # Left mouse button clicked
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for i, button in enumerate(buttons):
                if (width // 6 * (i + 1) - button.get_width() // 2 <= mouse_x <= width // 6 * (i + 1) + button.get_width() // 2) and (height // 2 - button.get_height() // 2 <= mouse_y <= height // 2 + button.get_height() // 2):
                    return i + 1


# this function check for the modified game win rule as specified (4 discs of the same color connected in a square) 
def winning_move(board, piece):
    # Check horizontal and vertical square formation
    for c in range(COLUMN_COUNT-1):
        for r in range(ROW_COUNT-1):
            if (board[r][c] == piece and board[r][c+1] == piece and
                board[r+1][c] == piece and board[r+1][c+1] == piece):
                return True

    return False

def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opp_piece = AI_PIECE

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4

	return score

def score_position(board, piece):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score Vertical
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score posiive sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	return score

def is_terminal_node(board):
	return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, AI_PIECE):
				return (None, 100000000000000)
			elif winning_move(board, PLAYER_PIECE):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, AI_PIECE))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, AI_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value

	else: # Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value
# to see all the locations on board which free to select
def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMN_COUNT):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations

def pick_best_move(board, piece):

	valid_locations = get_valid_locations(board)
	best_score = -10000
	best_col = random.choice(valid_locations)
	for col in valid_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		drop_piece(temp_board, row, col, piece)
		score = score_position(temp_board, piece)
		if score > best_score:
			best_score = score
			best_col = col

	return best_col

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, board_color, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # empt circle color is grey
            pygame.draw.circle(screen, GREY, (c * SQUARESIZE + SQUARESIZE // 2, r * SQUARESIZE + SQUARESIZE + SQUARESIZE // 2), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                # Player 1 color is white
                pygame.draw.circle(screen, WHITE, (c * SQUARESIZE + SQUARESIZE // 2, height - (r * SQUARESIZE + SQUARESIZE // 2)), RADIUS)
                # AI is Black
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, BLACK, (c * SQUARESIZE + SQUARESIZE // 2, height - (r * SQUARESIZE + SQUARESIZE // 2)), RADIUS)
    pygame.display.update()


# initialize the board before start the game
board = create_board()
print_board(board)
#change to trun when some player get a formal row wich mean won the game
game_over = False 

pygame.init()
name_font = pygame.font.SysFont("monospace", 20)
SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)

selected_color = None
while selected_color is None:
    color_buttons = show_color_menu(screen, COLORS, name_font)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            for color_name, color, button_rect in color_buttons:
                if button_rect.collidepoint(pos):
                    selected_color = color
                    break

    pygame.display.update()
    pygame.time.delay(100)  # Add this line


board_color = selected_color

player_name = get_user_input(screen, "Enter the player's name:", name_font, width // 2 - 100, height // 2 - 50, WHITE)
agent_name = "AI Agent"
#to select the first turn 
turn = choose_first_player(screen, player_name, agent_name, name_font, WHITE)
#to select search depth
search_depth = select_search_depth(screen, name_font, WHITE)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)
# this variabe to calculate Player 1 + Player 2 moves 
total_moves = 0

# store start time in start_time
start_time = time.time()
while not game_over:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

		if event.type == pygame.MOUSEMOTION:
			pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
			posx = event.pos[0]
			if turn == PLAYER:
				pygame.draw.circle(screen, WHITE, (posx, int(SQUARESIZE/2)), RADIUS)

		pygame.display.update()

		if event.type == pygame.MOUSEBUTTONDOWN:
			pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
			#print(event.pos)
			# Ask for Player 1 Input
			if turn == PLAYER:
				posx = event.pos[0]
				col = int(math.floor(posx/SQUARESIZE))

				if is_valid_location(board, col):
					row = get_next_open_row(board, col)
					drop_piece(board, row, col, PLAYER_PIECE)
					total_moves += 1 

					if winning_move(board, PLAYER_PIECE):
						#label = render_text_with_background(f"{player_name} wins!!", WHITE, (0, 0, 0, 128), myfont)
						game_over = True
						# store end time in end_time if player 1 won
						end_time = time.time()
						duration = end_time - start_time
						message = f"{player_name} wins!!\nGame lasted: {duration:.2f} seconds |  Total moves: {total_moves}"
						multiline_label = render_multiline_text_with_background(message, WHITE, BLACK, name_font)
						screen.blit(multiline_label, (40, 10))
						pygame.display.update()
					# to alternate between player 1 and AI
					turn += 1
					turn = turn % 2

					print_board(board)
					draw_board(board)


	# # Ask for Player 2 Input
	if turn == AI and not game_over:				

        #call minimax with search depth
		col, _ = minimax(board, search_depth, -math.inf, math.inf, True)

		if is_valid_location(board, col):
			row = get_next_open_row(board, col)
			drop_piece(board, row, col, AI_PIECE)
			total_moves += 1
			if winning_move(board, AI_PIECE):
				#label = render_text_with_background("Player 2 wins!!", BLACK, (255, 255, 255, 128), myfont)
				game_over = True
				# store end time in end_time if player 2 (AI) won
				end_time = time.time()
				duration = end_time - start_time
				message = f"AI wins!! Game lasted: {duration:.2f} seconds | Total moves: {total_moves}"
				multiline_label = render_multiline_text_with_background(message, WHITE, BLACK, name_font)
				screen.blit(multiline_label, (40, 10))
				pygame.display.update()

			print_board(board)
			draw_board(board)

			turn += 1
			turn = turn % 2

	if game_over:
		pygame.time.wait(10000)
