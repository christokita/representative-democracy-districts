import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
import numpy as np
import random
import scipy as sp
from typing import List


class SchellingModel:
    """
    Class for our Schelling model of social segregation.
    """

    def __init__(self, dimensions: List[int], proportions: List[float], vacancy: float) -> None:
        """
        Parameters
            dimensions:     desired dimensions of grid for population size List[0] x List[1].
            proportions:    desired proportions for different identities in population.
            vacancy:        % of spots on grid vacant.

        Returns:
            None
        """
        self.dimensions = dimensions
        self.population_proportions = proportions
        self.grid = self.populate_grid(dimensions=dimensions, proportions=np.array(proportions), vacancy=vacancy)

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
    
    def show_grid(self, vacant_color: str = "white", colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]) -> None:
        """
        Visualize populated grid.

        Parameters:
            vacant_color:   color for empty spots.
            colors:         list of colors for occupied spots.

        Returns:
            None
        """
        # Generate color map
        if len(self.population_proportions) > len(colors):
            raise("Not enough colors to plot.")
        else:
            colors = colors[0:len(self.population_proportions)]
            colors.insert(0, vacant_color)

        # Display grid
        plt.imshow(
            self.grid,
            cmap=ListedColormap(colors)
        )
        plt.show()


