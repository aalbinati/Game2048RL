from env import Game2048
import logic


def render(matrix):
    for row in matrix:
        print(row)
    print(f'Score: {game.score} Steps: {game.steps} Reward: {game.rew}')


game = Game2048(env_info={"random_movements": False, "human": True})

while not game.is_done():
    for row in game.matrix:
        print(row)
    if not game.random:
        game.step(input("\nMake a move: Up: 0, Down: 1, Left: 2, Right: 3\n"))
    else:
        game.step(action=0)

if logic.game_state(game.matrix) == 'win':
    print("You win!")
else:
    print("You lose :(")
print(f'Score: {game.score} Steps: {game.steps} Reward: {game.rew}')
