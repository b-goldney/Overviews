# %%
# this udpated code usess a dictionary to map the functions to a string. This way,
# we can avoid using if else statements to match the function

from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import Optional, Callable

# create internal states
class JobType(Enum):
    momentum = 'momentum'
    volatility = 'volatility'
    spread = 'spread'
    reversion = 'reversion'

# create functions for each job type
def momentum():
    return "momentum"

def volatility():
    return "volatility"

def spread():
    return "spread"

def reversion():
    return "reversion"

# map each function into the dict to avoid if/else statements 
job_types = {
    JobType.momentum : momentum,
    JobType.volatility : volatility,
    JobType.spread : spread,
    JobType.reversion : reversion 
}

# create worker class
class JobProcessor():
	def __init__(self):
		self.state: Optional[Callable[ [], str ]] = None

	def getState(self):
		return self.state

	def setState(self, state_fn: Callable[[], str] ):
		self.state: Callable[[],str] = state_fn

	def changeState(self):
		self.state = self.state()

# run Script
job_processor = JobProcessor()
print(f'The JobProcessors internal state is currently: {job_processor.getState()} \n')

# update state to momentum
target_state = JobType.momentum 
state_fn = job_types[target_state]
job_processor.setState(state_fn)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()}  \n')

# update state to volatility
target_state = JobType.volatility 
state_fn = job_types[target_state]
job_processor.setState(state_fn)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()}  \n')


# %%



