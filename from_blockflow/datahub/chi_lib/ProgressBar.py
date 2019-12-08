import sys
import timeit
import datetime
from chi_lib.library import logTracker, getTerminalSize
from math import floor


class ProgressBar:
	def __init__(self, name, maxValue, scale=5, width=100, progressType='bar'):
		self.__name = str(name)
		self.__maxValue = maxValue
		self.__curValue = 0
		self.__preProgress = None
		self.__startTime = timeit.default_timer()
		self.__terminalSize = getTerminalSize()[0]
		self.__totalTime = 0
		self.__preMess = ''
		self.__scale = scale
		self.__width = width * self.__scale
		if not progressType in ['bar', 'lineByLine']:
			logTracker.logException('Invalid progressType: ' + str(progressType))
		self.__progressType = progressType
		self.update(0)

	def __write(self, mess):
		n = len(mess)
		nPre = len(self.__preMess)
		if n < nPre:
			mess += (nPre - n) * ' '
		self.__preMess = mess
		sys.stdout.write(mess)

		if self.__progressType == 'bar':
			sys.stdout.flush()
			# return to start of line, after '['
			sys.stdout.write("\b" * len(mess))
		elif self.__progressType == 'lineByLine':
			sys.stdout.write('\n')
		else:
			logTracker.logException('Invalid progressType: ' + str(progressType))

	def __getTimeString(self, estTime):
		# time.strftime('%Hh %Mm %Ss', time.gmtime(estTime))
		#return "{:0>8}".format(datetime.timedelta(seconds=estTime))
		return str(datetime.timedelta(seconds=estTime))

	def update(self, curValue):
		if self.__maxValue == 0:
			percentage = 1
		else:
			percentage = float(curValue) / float(self.__maxValue)

		# --------------------------------------
		# Progress name
		namePart = '[' + self.__name + '] '

		# Remaining time
		timeBar = 'RTime: '
		if percentage == 0:
			timeBar += 'N/A'
		else:
			self.__totalTime = timeit.default_timer() - self.__startTime
			estTime = int(self.__totalTime * float(1.0 - percentage) / percentage)
			timeBar += self.__getTimeString(estTime)

		timeBar = '(' + timeBar + ')'

		#print curTime, self.__startTime, curTime - self.__startTime, self.__totalTime
		# Current percentage
		percentagePart = str(round(percentage * 100, 4)) + '% '

		# Calculate the remaining slots for progress bar (2 is the number of slots for [ and ] of progress bar)
		remainingWidth = self.__terminalSize - 3 - len(namePart) - len(percentagePart) - len(timeBar)
		self.__width = min(self.__width, remainingWidth) * self.__scale
		# --------------------------------------

		curProgress = int(floor(percentage * self.__width))
		if curProgress > self.__width or curProgress < 0:
			logTracker.logException('Invalid update value: ' + str(curValue) + ' > ' + str(self.__width))

		if self.__preProgress == None or curProgress > self.__preProgress:
			self.__preProgress = curProgress
			# Progress bar
			temp = int(curProgress / self.__scale)
			curProgressBar = '#' * temp
			# width is divisible by scale because it was scaled by scale
			remainingProgressBar = '.' * int(self.__width / self.__scale - temp)
			if self.__progressType == 'bar':
				progressBar = "[%s%s] " % (curProgressBar, remainingProgressBar)	
			elif self.__progressType == 'lineByLine':
				#progressBar = '[' + self.__name + "] " % (percentageStr)
				progressBar = ''
			else:
				logTracker.logException('Invalid progressType: ' + str(progressType))
			
			self.__write(namePart + progressBar + percentagePart + timeBar)

	def increase(self):
		self.__curValue += 1
		self.update(self.__curValue)
		
	def done(self):
		curProgressBar = '#' * int(self.__width / self.__scale)
		remainingProgressBar = ''
		percentageStr = '100%'
		totalTimeStr = 'Runtime: ' + self.__getTimeString(int(self.__totalTime))

		if self.__progressType == 'bar':
			mess = '[' + self.__name + "] [%s%s] %s (%s)" % (curProgressBar, remainingProgressBar, percentageStr, totalTimeStr)
		elif self.__progressType == 'lineByLine':
			mess = '[' + self.__name + "] %s (%s)" % (percentageStr, totalTimeStr)
		else:
			logTracker.logException('Invalid progressType: ' + str(progressType))

		self.__write(mess)
		sys.stdout.write("\n")
		self.__preProgress = 0