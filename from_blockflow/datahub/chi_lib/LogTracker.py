import os
from datetime import datetime as dt

class LogTracker:
	def __init__(self, logDir, isPrintLog, isWriteLog):
		self.__logDir = logDir
		if not os.path.isdir(logDir):
			os.makedirs(logDir)
		fName = dt.now().strftime('%Y%m%d-%H%M%S')
		self.__logPath = os.path.join(logDir, fName) + '.txt'
		#os.system('touch ' + self.__logPath)
		self.__isPrintLog = isPrintLog
		self.__prePrintLog = []
		self.__isWritetLog = isWriteLog
		self.__preWriteLog = []

	def log(self, message):
		if self.__isPrintLog == True:
			print(message)

		if self.__isWritetLog == True:	
			with open(self.__logPath, 'a', encoding='utf8') as f:
				prefix = dt.now().strftime('[%Y%m%d-%H:%M:%S:%f] ')
				f.write(prefix + message + '\n')

	def logException(self, message):
		self.log(message)
		raise Exception(message)

	def holdPrintLog(self, isPrintLog):
		self.__prePrintLog.append(self.__isPrintLog)
		self.__isPrintLog = isPrintLog

	def releasePrintLog(self):
		if len(self.__prePrintLog) == 0:
			raise Exception('Nothing to release')
		self.__isPrintLog = self.__prePrintLog.pop()

	def holdWriteLog(self, isWriteLog):
		self.__preWriteLog.append(self.__isWritetLog)
		self.__isWritetLog = isWriteLog

	def releaseWriteLog(self):
		if len(self.__preWriteLog) == 0:
			raise Exception('Nothing to release')
		self.__isWritetLog = self.__preWriteLog.pop()


