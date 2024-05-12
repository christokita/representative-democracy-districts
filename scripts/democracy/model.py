import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

class LegislativeDistricts:
    """
    Class that takes results of Schelling model and constructs legislative districts for election.
    """

    def __init__(
            self, voters: pd.DataFrame, x_column: str = "grid_x", y_column: str = "grid_y", identity_column: str = "identity"
        ) -> None:
        """
        Parameters:
            voters:             long-form of Schelling model population grid with three colums: x coordinate, y coordinate, and voter identity.
            x_column:           name of column containing voter's X coordinate in grid.
            x_column:           name of column containing voter's Y coordinate in grid.
            identity_column:    name of column containing voter's identty, which dictates voting behavior.

        Returns:
            None
        """
        self.voters = voters
        self.x_col = x_column
        self.y_col = y_column
        self.identity_col = identity_column

    def create_districts(self, district_width: int, district_height: int, verbose: bool = True) -> None:
        """
        Assign each voter to a legislative district.
        In order to make sure districts are evenly sized,
        choose appropriate district width and height.

        Parameters:
            district_width:     desired width of each district.
            district_height:    desired height of each district.
            verbose:            whether to display message upon district creation.

        Returns:
            None. Adds `district` column to self.voters dataframe.
        """
        # Get dimensions of population grid
        grid_width = len( self.voters[self.x_col].unique() )
        grid_height = len( self.voters[self.y_col].unique() )
        
        # Assign user to district
        district_x = self.voters[self.x_col] // district_width
        district_y = self.voters[self.y_col] // district_height
        
        num_districts_per_row = grid_width // district_width
        num_districts_per_col = grid_height // district_height
        self.voters["district"] = district_x + 1 + district_y * num_districts_per_row
        
        # Print message
        n_districts = num_districts_per_row*num_districts_per_col
        if verbose:
            print(f"Generated {n_districts:,} voting districts, each with {int(self.voters.shape[0]/n_districts):,} voters.")
        return None 

    def conduct_election(self, members_per_district: int = 1) -> Tuple[pd.DataFrame]:
        """
        Simulate election of N-member districts in our electorate.
        With multi-member districts, in the event there is only one type of voter there,
        two representatives of the same identity will be elected.

        Parameters:
            members_per_district:   number of representatives elected per district (1 = single-member, 2+ = multi-member)
            
        Returns:
            district_votes:     breakdown of votes for each candidate.
            elected:            list of who won in each district.
        """
        district_votes = (
            self.voters
                .groupby(["district", self.identity_col])
                .size()
                .reset_index(name="votes")
                .sort_values(by=["district", "votes"], ascending=[True,False])
                .reset_index(drop=True)
        )
        elected = (
            district_votes
                .groupby("district")
                .head(members_per_district)
                .set_index("district")
                .groupby(level=0)
                .apply(lambda x: x.reindex(x.index.repeat(members_per_district))[:members_per_district] if len(x) < members_per_district else x)
                .reset_index(level=1)
                .reset_index(drop=True)
                .rename(columns={self.identity_col: "representive_identity"})
                .assign(order_elected=lambda x: x.groupby("district").cumcount())
                .drop(columns=["votes"])
        )
        return district_votes, elected
    
    def calculate_segregation(self) -> pd.DataFrame:
        """
        Calculate the social segregation within each district and across the entire population.
        """
        # Calculate identity breakdown by district
        district_demographics = (
            self.voters
            .groupby(["district", "identity"])
            .size()
            .reset_index(name="voters")
            .pivot(index="district", columns="identity", values="voters")
            .fillna(0)
        )
        district_demographics = district_demographics.drop(columns=[-1]) #drop vancancies

        # Calculate overall entropy
        E = stats.entropy( district_demographics.sum(axis=0) )
        T = district_demographics.values.sum()

        # Calcualte segregation by district
        E_i = district_demographics.apply(stats.entropy, axis=1)
        T_i = district_demographics.sum(axis=1)

        district_segregation = pd.DataFrame({
            "district": district_demographics.index,
            "n_voters": T_i,
            "diversity": E_i,
            "segregation": (E - E_i) / E
        })
        district_segregation = district_segregation.reset_index(drop=True)

        # Calculate Thiel's Index
        H = sum(district_segregation.n_voters/T * district_segregation.segregation)
        return district_segregation, H
    
    def calculate_representation(self, elected: pd.DataFrame) -> pd.DataFrame:
        """
        Determine the number and percent of voters who now have an elected official that is of the same identity.

        Parameters:
            elected:    dataframe of representatives elected in each district.

        Return:
            representation:     
        """
        representation = pd.DataFrame({
            "district": self.voters.district.unique(), 
            "n_voters": np.nan,
            "n_represented": np.nan, 
            "prop_represented": np.nan
        })
        for _, row in representation.iterrows():
            district_voters = self.voters[self.voters.district == row.district]
            district_voters = district_voters[district_voters.identity != -1]
            district_representatives = elected[elected.district == row.district]
            representation.loc[representation.district == row.district, "n_voters"] = district_voters.shape[0]
            representation.loc[representation.district == row.district, "n_represented"] = sum(district_voters.identity.isin(district_representatives.representive_identity))
            representation.loc[representation.district == row.district, "prop_represented"] = sum(district_voters.identity.isin(district_representatives.representive_identity)) / district_voters.shape[0]
        return representation
