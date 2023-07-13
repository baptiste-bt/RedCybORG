# Red CybORG

This repository is a fork from the Cyber Operations Research Gym (CybORG). It is aimed at adding several features to CybORG for training Red Team (offensive agents). It has not been updated to use the last Scenario "Drone Swarm Scenario", it is mainly focused on Scenario1b and Scenario2 from the previous editions of the CAGE Challenge.

All main features of the original version of CybORG have been preserved, you can refer to the Tutorials to learn the basics to run CybORG to train your RL agents.

## Installation

Clone the repository with the following

```
git clone https://github.com/baptiste-bt/RedCybORG.git
```

Change your active directory to RedCybORG and install RedCybORG locally using pip from the main directory that contains this readme

```
cd RedCybORG
pip install -e .
```

## Creating the environment

Create a CybORG environment with the DroneSwarm Scenario that is used for CAGE Challenge 3:

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator
import inspect


scenario = 'Scenario2' # 'Scenario1b' is also possible 
path = str(inspect.getfile(CybORG))
path = path[:-7] + f'/Simulator/Scenarios/scenario_files/{scenario}.yaml'
sg = FileReaderScenarioGenerator(path)
cyborg = CybORG(sg, 'sim')

```

## Wrappers


To alter the interface with CybORG, [wrappers](CybORG/Agents/Wrappers) are available. The usual way of wrapping RedCybORG is using the following:

* [CybORG env](CybORG/env.py) - the default unwrapped environment. It returns dict observations.
* [RedTableWrapper](CybORG/Agents/Wrappers/RedTableWrapper.py) - wraps the environment with a simple wrapper, returning summarized observations, with only the needed information for the red agent. If you want full observations (with all information on processes, OS, files...) you can use [FixedFlatWrapper](CybORG/Agents/Wrappers/FixedFlatWrapper.py) instead. Be careful with this wrapper which generates a huge number of features for each host which are not necessarily well encoded and may hurt training performances.
* [OpenAIGymWrapper](CybORG/Agents/Wrappers/OpenAIGymWrapper.py) - alters the interface to conform to the OpenAI Gym specification. Requires the observation to be changed into a fixed size array.

## How to Use

### OpenAI Gym Wrapper

The OpenAI Gym Wrapper allows interaction with a single external agent. The name of that external agent must be specified at the creation of the OpenAI Gym Wrapper.

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper, RedTableWrapper
import inspect

scenario = 'Scenario2' 
path = str(inspect.getfile(CybORG))
path = path[:-7] + f'/Simulator/Scenarios/scenario_files/{scenario}.yaml'
sg = FileReaderScenarioGenerator(path)
cyborg = CybORG(sg, 'sim')
wrapped_cyborg = OpenAIGymWrapper(agent_name='Red', env=RedTableWrapper(cyborg))
obs = wrapped_cyborg.reset()
print(obs)

```