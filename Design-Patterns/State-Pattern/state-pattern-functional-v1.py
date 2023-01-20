# %%
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import Optional, Callable

# create functions for each state 
def momentum():
    return "momentum"

def volatility():
    return "volatility"

def spread():
    return "spread"

def reversion():
    return "reversion"


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

# Run Script
job_processor = JobProcessor()
print(f'The JobProcessors internal state is currently: {job_processor.getState()} \n')

print("changing state to momentum")
job_processor.setState(momentum)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()}  \n')

print("changing state to volatility")
job_processor.setState(volatility)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()} \n')

print("changing state to spread")
job_processor.setState(spread)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()} \n')

print("changing state to reversion")
job_processor.setState(reversion)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()} \n')

# %%



