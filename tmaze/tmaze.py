import random
import time
import tkinter as tk


# This file implements the T-maze environment. This environment consists of
# a long corridor ending with a treasure placed either in an 'up' cell or
# in a 'down' cell (see figure).
# The agent starts at the beginning of the corridor and receives a state
# telling wether the reward is up or down. Then once it moves in the corridor,
# it receives always the same state (no matter where the treasure is located)
# telling that it is in the corridor. Once it reaches the end, it receives a
# different state, and it must make a choice: either go up or go down. The
# agent must thus retain the information of where is located the reward. If it
# finds the treasure, it receives a big positive reward. But if it chooses the
# bad side, it receives a bad reward. In both cases a terminal state is reached.
#
#                                     _
#               _____________________|R|
#              |A                      |
#               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾| |
#                                     ‾
#


class States:
    """Define all the possible states of the T-maze environment."""
    UpInitial = [0, 1, 1]
    DownInitial = [1, 1, 0]
    Corridor = [1, 0, 1]
    TJunction = [0, 1, 0]
    End = [0, 0, 0]


class Rewards:
    """Define the rewards of the T-maze environment"""
    Action = -0.1
    # The 'End' reward's value is changed to the length of the T-maze when it
    # is initialized
    End = 10.0


class Actions:
    """Define the actions of the T-maze environment"""
    Up = 0
    Down = 1
    Right = 2
    Left = 3


class TMaze:
    """Implements the T-maze environment."""
    def __init__(self, length=10, render=False, init=2, print_end=False):
        """Creates a new environment."""
        if length < 3:
            raise ValueError("'length' must be greater than 3.")

        self.length = length
        self.render = render
        self.num_states = 3
        self.num_actions = 4
        self.init = init
        self.print_end = print_end
        Rewards.End = length

    def reset(self):
        """Reset the environment and return the initial state."""
        if self.init == 2:
            self.reward_is_up = bool(random.randint(0, 1))

        else:
            self.reward_is_up = bool(self.init)

        self.pos = 0
        self.done = False

        self.current_state = States.UpInitial if self.reward_is_up else States.DownInitial

        if self.render is True:
            self.gui = TMazeGui(self.length, self.reward_is_up)

        return self.current_state

    def step(self, action):
        """
        Make a transition in the environment given an action. Return the
        new state of the environment, the reward and a boolean telling whether
        or not a terminal state has been reached.
        """
        if self.done is True:
            return self.current_state, 0.0, True

        # In initial state
        if self.current_state in [States.UpInitial, States.DownInitial]:
            if action == Actions.Right:
                self.current_state = States.Corridor
                self.pos = 1

                if self.render is True:
                    self.gui.go_right()

            return self.current_state, Rewards.Action, False

        # In corridor
        elif self.current_state == States.Corridor:
            if action == Actions.Right:
                self.pos += 1
                self.current_state = States.TJunction if self.pos == self.length - 1 \
                                        else States.Corridor

                if self.render is True:
                    self.gui.go_right()

            elif action == Actions.Left and self.pos != 0:
                self.pos -= 1

                if self.render is True:
                    self.gui.go_left()

            return self.current_state, Rewards.Action, False

        # At junction
        if action == Actions.Up:
            self.done = True
            self.current_state = States.End

            if self.render is True:
                self.gui.go_up()

            if self.reward_is_up is True:
                if self.print_end is True:
                    print('Won !')

                return self.current_state, Rewards.End, True

            if self.print_end is True:
                print('Lost !')

            return self.current_state, Rewards.Action, True

        if action == Actions.Down:
            self.done = True
            self.current_state = States.End

            if self.render is True:
                self.gui.go_down()

            if self.reward_is_up is False:
                if self.print_end is True:
                    print('Won !')

                return self.current_state, Rewards.End, True

            if self.print_end is True:
                print('Lost !')

            return self.current_state, Rewards.Action, True

        if action == Actions.Left:
            self.pos -= 1
            self.current_state = States.Corridor
            if self.render is True:
                self.gui.go_left()

        return self.current_state, Rewards.Action, False

    def render_now(self):
        """Render the environment."""
        self.gui.update()


class Colors:
    """Colors used when rendering the environment."""
    Background = '#f2f2f2'
    Agent = '#292929'
    Win = '#2ebd19'
    Lose = '#db2616'


class Config:
    """Some parameters used when rendering the environment"""
    RenderTime = 0.1
    CellSize = 50


class TMazeGui():
    """Handles the rendering of the T-maze."""
    def __init__(self, length, reward_is_up):
        self.length = length
        self.reward_is_up = reward_is_up
        self.width = (length + 4) * Config.CellSize
        self.height = 5 * Config.CellSize
        self.finished = False

        self.generate()

    def generate(self):
        self.root = tk.Tk()
        self.root.title('T-maze Playground')

        self.frame = tk.Frame(self.root)
        self.canvas = tk.Canvas(self.frame, width=self.width,
                                height=self.height, bg=Colors.Background)

        self.cells = []
        self.pos = 0
        x = Config.CellSize * 2
        y = Config.CellSize * 2

        self.cells.append(self.canvas.create_rectangle(
            x, y, x + Config.CellSize, y + Config.CellSize, fill=Colors.Agent
        ))

        for i in range(1, self.length):
            x += Config.CellSize
            self.cells.append(self.canvas.create_rectangle(
                x, y, x + Config.CellSize, y + Config.CellSize, fill=Colors.Background
            ))

        if self.reward_is_up:
            up_color = Colors.Win
            down_color = Colors.Lose

        else:
            up_color = Colors.Lose
            down_color = Colors.Win

        self.cells.append(self.canvas.create_rectangle(
            x, y - Config.CellSize, x + Config.CellSize, y, fill=up_color
        ))

        self.cells.append(self.canvas.create_rectangle(
            x, y + Config.CellSize, x + Config.CellSize, y + 2 * Config.CellSize, fill=down_color
        ))

        self.canvas.pack()
        self.frame.pack()
        self.update()

    def update(self):
        time.sleep(Config.RenderTime/2)
        self.root.update()
        time.sleep(Config.RenderTime/2)
        self.root.update()

        if self.finished is True:
            time.sleep(1)

    def go_right(self):
        if self.pos >= self.length - 1:
            return

        self.pos += 1
        self.canvas.itemconfigure(self.cells[self.pos], fill=Colors.Agent)
        self.canvas.itemconfigure(self.cells[self.pos - 1], fill=Colors.Background)

    def go_left(self):
        if self.pos == 0 or self.pos >= self.length:
            return

        self.pos -= 1
        self.canvas.itemconfigure(self.cells[self.pos], fill=Colors.Agent)
        self.canvas.itemconfigure(self.cells[self.pos + 1], fill=Colors.Background)

    def go_up(self):
        if self.pos != self.length - 1:
            return

        self.pos += 1
        self.canvas.itemconfigure(self.cells[self.pos], fill=Colors.Agent)
        self.canvas.itemconfigure(self.cells[self.pos - 1], fill=Colors.Background)

        self.canvas.create_text(self.width/2, Config.CellSize/2, fill="darkblue",
                                 font="Times 20 italic bold",
                                 text="Won !" if self.reward_is_up else "Lost...")

        self.finished = True

    def go_down(self):
        if self.pos != self.length - 1:
            return

        self.pos += 2
        self.canvas.itemconfigure(self.cells[self.pos], fill=Colors.Agent)
        self.canvas.itemconfigure(self.cells[self.pos - 2], fill=Colors.Background)

        self.canvas.create_text(self.width/2, Config.CellSize/2, fill="darkblue",
                                 font="Times 20 italic bold",
                                 text="Won !" if not self.reward_is_up else "Lost...")

        self.finished = True

if __name__ == '__main__':
    # Small script to test the T-maze.

    length = random.randint(3, 20)
    env = TMaze(length, render=True)

    initial_state = env.reset()
    print('initial_state: {} length: {:>2}'.format(initial_state, length))

    state = initial_state
    action = Actions.Right
    for i in range(length - 1):
        state, reward, done = env.step(action)
        print('step: {:>2} state: {} reward: {} done: {}'.format(i, state, reward, done))
        env.render_now()

    if initial_state == States.UpInitial:
        state, reward, done = env.step(Actions.Up)
        env.render_now()

    else:
        state, reward, done = env.step(Actions.Down)
        env.render_now()

    print('step: {:>2} state: {} reward: {} done: {}'.format(i, state, reward, done))
