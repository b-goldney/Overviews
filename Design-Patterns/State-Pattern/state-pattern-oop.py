# %%
from abc import abstractmethod, ABCMeta
from typing import Optional, Callable

class InternalState(metaclass = ABCMeta):
	@abstractmethod
	def changeState(self):
		pass

class Momentum(InternalState):
	def changeState(self):
		print("Executing momentum trade")
		return "momentum trade"

class Volatility(InternalState):
	def changeState(self):
		print("Executing volatility trade")
		return "volatility trade"


class Spread(InternalState):
	def changeState(self):
		print("Executing spread trade")
		return "spread trade"

class Reversion(InternalState):
	def changeState(self):
		print("Executing reversion trade")
		return "reversion trade"

class JobProcessor(InternalState):
	def __init__(self):
		self.state: Optional[InternalState] = None

	def getState(self):
		return self.state

	def setState(self, state: InternalState):
		self.state = state 

	def changeState(self):
		self.state = self.state.changeState()

job_processor = JobProcessor()
print(f'The JobProcessors internal state is currently: {job_processor.getState()} \n')

momentum = Momentum()
volatility = Volatility()

print("changing state to momentum")
job_processor.setState(momentum)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()} \n \n')

print("changing state to volatility")
job_processor.setState(volatility)
job_processor.changeState()
print(f'The job_processors internal state is currently: {job_processor.getState()}')


# %%



