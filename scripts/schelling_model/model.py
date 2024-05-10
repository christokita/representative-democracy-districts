import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import random
from scipy import signal
from typing import List


MODEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
# MODEL_COLORS = ["#420a68", "#fca50a", "#932667", "#dd513a", "#000004"]


class SchellingModel:
    """
    Class for our Schelling model of social segregation.
    """

    def __init__(self, dimensions: List[int], proportions: List[float], vacancy: float, satisfaction_threshold: float) -> None:
        """
        Parameters
            dimensions:                 desired dimensions of grid for population size List[0] x List[1].
            proportions:                desired proportions for different identities in population.
            vacancy:                    % of spots on grid vacant.
            satisfaction_threshold:     minimum desired % of neighbors that are the same identity.

        Returns:
            None
        """
        # Model parameters
        self.dimensions = dimensions
        self.population_proportions = proportions
        self.vacancy = vacancy
        self.threshold = satisfaction_threshold

        # Set up model grid
        self.grid = self.populate_grid(dimensions=dimensions, proportions=np.array(proportions), vacancy=vacancy)
        self.grid_initial = None

        # Simulation variables for tracking purposes
        self.t_end = None #what timestep does simulation end on

        # Neighborhood kernel for each individual on the grid
        self.kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

    def populate_grid(self, dimensions: List[int], proportions: np.array, vacancy: float) -> np.ndarray:
        """
        Generate the desired population mix on the desired grid size.

        Parameters:
            dimensions:     desired dimensions of grid for population size List[0] x List[1].
            proportions:    desired proportions for different identities in population.
            vacancy:        % of spots on grid vacant.

        Returns:

        """
        # Make sure population proportions sum to one
        proportions = proportions / proportions.sum()

        # Calculate number of spots to be filled
        total_spots = dimensions[0] * dimensions[1]
        vacant_spots = int(total_spots * vacancy)
        occupied_spots = int(total_spots - vacant_spots)
        remaining_spots = occupied_spots
        
        # Create vacancies
        spot_values = np.repeat(-1, vacant_spots)

        # Assign the minimum number of spots based on the population proportions
        for type, prop in enumerate(proportions):
            n_spots = int(prop * occupied_spots)
            spot_values = np.append(spot_values, np.repeat(int(type), n_spots))
            remaining_spots -= n_spots

        # Distribute the remaining spots 
        # Needed in case proportion * total_spots results in fractions
        proportions = np.append(proportions, vacancy) # include vacancies in the remaining spots distribution
        while remaining_spots > 0:
            max_proportion_index = proportions.argmax()
            if max_proportion_index == len(proportions) - 1:
                spot_values = np.append(spot_values, -1) #vacancy
            else:
                spot_values = np.append(spot_values, max_proportion_index) #occupied
            remaining_spots -= 1
            proportions[max_proportion_index] = 0

        # Shuffle and shape population into grid
        random.shuffle(spot_values)
        return np.reshape(spot_values, newshape=tuple(dimensions))
    
    def evolve(self, max_steps: int = 1000, boundary: str = "wrap", verbose: bool = True) -> None:
        """
        Simulate Schelling model on the population grid.

        Parameters:
            max_steps:  maximum number of steps to run model.
            boundary:   how boundaries are dealt with on the grid.
            verbose:    whether to display evolution progress messages.

        Returns:
            None (this updates self.grid)
        """
        self.grid_initial = self.grid.copy()

        # Grab unique types in the grid
        types = range(len(self.population_proportions))

        # Iterate until all individuals are satsfied or we run out of steps
        step = 0
        while step < max_steps:

            # Calculate how many non-vacant neighbors each individual has
            total_neighbors = signal.convolve2d(
                self.grid != -1, 
                self.kernel,
                mode="same",
                boundary=boundary
            )

            # Count up how many in-group neighbors each individual has
            type_neighbors = {}
            for t in types:
                type_neighbors[t] = signal.convolve2d(
                    self.grid == t, 
                    self.kernel,
                    mode="same",
                    boundary=boundary
                ) 

            # Find dissatisfied individuals
            dissatisfied = np.zeros_like(self.grid, dtype=bool)
            type_dissatisfied = {}
            for t in types:
                type_dissatisfied[t] = (type_neighbors[t]/total_neighbors < self.threshold) & (self.grid == t)
                dissatisfied |= type_dissatisfied[t]

            if not dissatisfied.any():
                break #stop iterating if all individuals are satisfied

            # Allow all dissatisfied individuals to move randomly
            vacant_spots = np.sum(self.grid == -1)
            self.grid[dissatisfied] = -1
            
            filling = -np.ones(vacant_spots)
            for t in types:
                n_movers = type_dissatisfied[t].sum()
                filling = np.append(filling, np.repeat(t, n_movers))

            np.random.shuffle(filling)
            self.grid[self.grid == -1] = filling

            # Update step
            step += 1
        
        if verbose:
            print(f"Model simulation done at t = {step:,}")
        self.t_end = step
        return None
    

    def _plot_grid(
            self, stage: str = "current", vacant_color: str = "white", colors: List[str] = MODEL_COLORS, ax: plt.axes = None
        ) -> None:
        """
        Visualize populated grid.

        Parameters:
            stage:          "current" or "initial" grid
            vacant_color:   color for empty spots.
            colors:         list of colors for occupied spots.
            ax:             matplotlib axes object to plot on (optional).

        Returns:
            None
        """
        # Generate color map
        if len(self.population_proportions) > len(colors):
            raise("Not enough colors to plot.")
        elif self.vacancy != 0:
            colors = colors[0:len(self.population_proportions)]
            colors.insert(0, vacant_color)
        else:
            colors = colors[0:len(self.population_proportions)]

        # Select grid to display
        if stage == "current":
            grid = self.grid
        elif stage == "initial":
            grid = self.grid_initial

        # Plot grid
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(
            grid,
            cmap=ListedColormap(colors)
        )

        # Format grid
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        return ax
    
    def show_grid(
            self, stage: str = "current", vacant_color: str = "white", colors: List[str] = MODEL_COLORS
        ) -> None:
        """
        Visualize a stage of the populated grid.

        Parameters:
            stage:          "current" or "initial" grid
            vacant_color:   color for empty spots.
            colors:         list of colors for occupied spots.

        Returns:
            None
        """
        self._plot_grid(stage=stage, vacant_color=vacant_color, colors=colors)
        plt.tight_layout()
        plt.show()

    def show_grid_evolution(
            self, vacant_color: str = "white", colors: List[str] = MODEL_COLORS
        ) -> None:
        """
        Visualize grid before and after simulation.

        Parameters:
            vacant_color:   color for empty spots.
            colors:         list of colors for occupied spots.

        Returns:
            None
        """
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        self._plot_grid(stage="initial", vacant_color=vacant_color, colors=colors, ax=ax[0])
        self._plot_grid(stage="current", vacant_color=vacant_color, colors=colors, ax=ax[1])
        ax[0].title.set_text("Initial")
        ax[1].title.set_text(f"End (t = {self.t_end})")
        plt.tight_layout()
        plt.show()

    def grid_to_long_form(self, stage: str = "current") -> np.array:
        """
        Turn 2D population grid into long form: location (x, y) and value.

        Parameters:
            stage:  "current" or "initial" grid.

        Return:
            long_grid:  flattened population grid.
        """
        if stage == "current":
            grid = self.grid
        elif stage == "initial":
            grid = self.grid_initial
        x, y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
        x_flat = x.ravel()
        y_flat = y.ravel()
        value_flat = grid.ravel()
        long_grid = np.column_stack((x_flat, y_flat, value_flat))
        return pd.DataFrame(long_grid, columns=["grid_x", "grid_y", "identity"])

