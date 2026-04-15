import numpy as np

_FLOOR = True
_WALL = False

class Maze:
    """A maze represented by a grid of floors and walls."""

    def __init__(self, size=8):
        """Initialize a new, random maze"""
        self.shape = (2*size + 1,)*2 # Twice as large as the actual maze (includes space for walls)
        self.start_position = (0,)*2 # Upper left corner
        self.goal_position = (size - 1,)*2 # Lower right corner
        self.maze = np.zeros(shape=self.shape, dtype=bool) # Start with all walls

        # Randomly initialize the maze with a depth-first search
        visited = np.zeros(shape=(size,size), dtype=bool)
        def generate_maze_from_cell(position):
            # Local function to allow recursion, but prevent calling outside of the initializer
            x, y = position # unpack position
            visited[x, y] = True # Mark the current position as visited
            self.maze[2*x+1, 2*y+1] = _FLOOR # Mark the position in the maze as floor

            # Make sure we've visited each neighbor
            neighbor_coords = (x+1, y), (x, y+1), (x-1, y), (x, y-1)
            for neighbor in np.random.permutation(neighbor_coords):
                neighbor_x, neighbor_y = neighbor

                # First make sure the coordinates are in bounds
                if not (0 <= neighbor_x < size) or not (0 <= neighbor_y < size):
                    continue # invalid coords, skip

                # Next check whether this neighbor has been visited already. If not, recurse
                if visited[neighbor_x, neighbor_y]:
                    continue # previously visited, skip
                else:
                    # Mark the connecting piece of the maze as floor
                    self.maze[x+neighbor_x+1, y+neighbor_y+1] = _FLOOR
                    generate_maze_from_cell(neighbor)
        generate_maze_from_cell(self.goal_position)
        self.maze[1, 1] = _FLOOR # Ensure start position is accessible

    def display(self):
        """Display the maze, using block characters for floors and empty spaces for walls"""
        str_maze = np.where(self.maze, '██', '  ')
        print('START')
        for row in str_maze:
            print(''.join([c for c in row]))
        print('  '*(self.shape[0]-3), 'FINISH')



if __name__ == "__main__":
    m = Maze(16)
    m.display()
