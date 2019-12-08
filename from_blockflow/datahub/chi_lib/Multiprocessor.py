import os
import sys
import time
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import traceback
from chi_lib.ProgressBar import ProgressBar


class Multiprocessor:
    def __init__(self, nProcesses=None, chunksize=1, progressType='bar'):
        cpuCount = mp.cpu_count()
        if nProcesses == None:
            nProcesses = cpuCount
        self.__nProcesses = nProcesses
        self.__chunksize = chunksize
        self.__progressType = progressType

    def run(self, targetFunction, inputs, title, isThread=False):
        n = len(inputs)
        progress = ProgressBar(title + ' (' + str(self.__nProcesses) + '/' + str(mp.cpu_count()) + ')', n, progressType=self.__progressType)
        if isThread == True:
            pool = ThreadPool(processes=self.__nProcesses)
        else:
            # Using tempStdout to avoid error from RedirectText of Update.py (we change the default stdout at here)
            tempStdout = sys.stdout 
            # Set to the default stdout
            sys.stdout = sys.__stdout__
            # The sys.stdout.flush() of the default stdout will be called inside the mp.Pool() constructor
            # The RedirectText outstream has no flush() method
            pool = mp.Pool(processes=self.__nProcesses)
            # Then set back to the current stdout
            sys.stdout = tempStdout
        result = pool.map_async(targetFunction, inputs, chunksize=self.__chunksize)
        while not result.ready():
            progress.update(n - result._number_left)
            time.sleep(1)
        progress.done()
        if not result.successful():
            mess = 'Nothing'
            try:
                mess = result.get()
            except:
                mess = str(traceback.format_exc())
            print('Incompleted result: ' + mess)

        # no more tasks
        #print result.get()
        pool.close() 
        # wrap up current tasks
        pool.join()  
        return result.get()

    @staticmethod
    def cpuCount():
        return mp.cpu_count()
