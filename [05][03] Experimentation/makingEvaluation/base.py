"""
The code provided in this file is based on the dysts Python library, which can be found at the following 
GitHub repository: https://github.com/williamgilpin/dysts/

In this modified version, dynamical systems can be computed for multivariate time series, in Python.

It's important to note that we are not affiliated with the creators of nolds. We have included this file solely 
to maintain the reproducibility of our benchmark experiments. Therefore, if you intend to use or modify this code, 
please ensure that you cite the original creators of nolds and comply with their licensing terms.

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

"""


from dataclasses import dataclass, field, asdict
import warnings
import json
import collections

import os
import sys
import gzip

curr_path = sys.path[0]

import pkg_resources
# conda install setuptools

data_path_continuous = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/moreSystems/chaotic_attractors.json"

import numpy as np

import importlib

try:
    from numba import jit, njit

    #     from jax import jit
    #     njit = jit

    has_jit = True
except ModuleNotFoundError:
    import numpy as np

    has_jit = False
    # Define placeholder functions
    def jit(func):
        return func

    njit = jit

staticjit = lambda func: staticmethod(
    njit(func)
)  # Compose staticmethod and jit decorators

data_default = {'bifurcation_parameter': None,
                'citation': None,
                 'correlation_dimension': None,
                 'delay': False,
                 'description': None,
                 'dt': 0.001,
                 'embedding_dimension': 3,
                 'hamiltonian': False,
                 'initial_conditions': [0.1, 0.1, 0.1],
                 'kaplan_yorke_dimension': None,
                 'lyapunov_spectrum_estimated': None,
                 'maximum_lyapunov_estimated': None,
                 'multiscale_entropy': None,
                 'nonautonomous': False,
                 'parameters': {},
                 'period': 10,
                 'pesin_entropy': None,
                 'unbounded_indices': []
               }

@dataclass(init=False)


class BaseDyn:
    """A base class for dynamical systems
    
    Attributes:
        name (str): The name of the system
        params (dict): The parameters of the system.
        random_state (int): The seed for the random number generator. Defaults to None
        
    Development:
        Add a function to look up additional metadata, if requested
    """

    name: str = None
    params: dict = field(default_factory=dict)
    random_state: int = None

    def __init__(self, **entries):
        self.name = self.__class__.__name__
        self._load_data()

        self.params = self._load_data()["parameters"]
        self.params.update(entries)
        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)

        ic_val = self._load_data()["initial_conditions"]
        if not np.isscalar(ic_val):
            ic_val = np.array(ic_val)
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self._load_data().keys():
            setattr(self, key, self._load_data()[key])
    
    def update_params(self):
        """
        Update all instance attributes to match the values stored in the 
        `params` field
        """
        for key in self.params.keys():
            setattr(self, key, self.params[key])
    
    def get_param_names(self):
        return sorted(self.params.keys())

    def _load_data(self):
        """Load data from a JSON file"""
        # with open(os.path.join(curr_path, "chaotic_attractors.json"), "r") as read_file:
        #     data = json.load(read_file)
        with open(self.data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            #return {"parameters": None}
            return data_default

    @staticmethod
    def _rhs(X, t):
        """The right-hand side of the dynamical system"""
        return X

    @staticmethod
    def bound_trajectory(traj):
        """Bound a trajectory within a periodic domain"""
        return np.mod(traj, 2 * np.pi)
    
    def load_trajectory(
        self,
        subsets="train", 
        granularity="fine", 
        return_times=False,
        standardize=False,
        noise=False
    ):
        """
        Load a precomputed trajectory for the dynamical system
        
        Args:
            subsets ("train" |  "test"): Which dataset (initial conditions) to load
            granularity ("course" | "fine"): Whether to load fine or coarsely-spaced samples
            noise (bool): Whether to include stochastic forcing
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
                
        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory
        
        """
        period = 12
        granval = {"coarse": 15, "fine": 100}[granularity]
        dataset_name = subsets.split("_")[0]
        data_path = f"{dataset_name}_multivariate__pts_per_period_{granval}__periods_{period}.json.gz"
        if noise:
            name_parts = list(os.path.splitext(data_path))
            data_path = "".join(name_parts[:-1] + ["_noise"] + [name_parts[-1]])
            
        cwd = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(cwd, "data", data_path)
        # with open(data_path, "r") as file:
        #     dataset = json.load(file)
        with gzip.open(data_path, 'rt', encoding="utf-8") as file:
            dataset = json.load(file)
            
        tpts, sol = np.array(dataset[self.name]['time']), np.array(dataset[self.name]['values'])
        
        if standardize:
            """standardize the time series after the computation"""
            sol = standardize_ts(sol)

        if return_times:
            return tpts, sol
        else:
            return sol

    def make_trajectory(self, *args, **kwargs):
        """Make a trajectory for the dynamical system"""
        raise NotImplementedError

    def sample(self, *args,  **kwargs):
        """Sample a trajectory for the dynamical system via numerical integration"""
        return self.make_trajectory(*args, **kwargs)
    
        
class DynSys(BaseDyn):
    """
    A continuous dynamical system base class, which loads and assigns parameter
    values from a file

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the base dynamical
            model class
    """

    def __init__(self, **kwargs):
        self.data_path = data_path_continuous
        super().__init__(**kwargs)
        self.dt = self._load_data()["dt"]
        self.period = self._load_data()["period"]

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        param_list = [
            getattr(self, param_name) for param_name in self.get_param_names()
        ]
        out = self._rhs(*X.T, t, *param_list)
        return out

    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)
    

