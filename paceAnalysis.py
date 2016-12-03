import json
import math
import matplotlib.pyplot as plt

class paceAnalyser:
	def __init__(self, filename):
		self.filename = filename
		self.amplitude = []
		self.time_series = []
		self.data = ''

	def readData(self):
		with open(self.filename, 'r') as old:
			filedata = old.read()

		if filedata[0] != '[':
			filedata = '[' + filedata
		filedata = filedata.replace('}\n', '},\n')
		if filedata[-2] == ',':
			filedata = filedata[:-2] + ']'
		
		with open(self.filename + '_json', 'w') as new:
			new.write(filedata)

		with open(self.filename + '_json') as jsonfile:
			data = json.load(jsonfile)
		
		self.data = data

	def readPace(self):
		self.amplitude = []
		self.time_series = []
		for record in self.data:
			if 'accelerometer_values' not in record or 'gyroscope_values' not in record or 'magnetometer_values' not in record:
				continue
			accer_x = record['accelerometer_values']['x']
			accer_y = record['accelerometer_values']['y']
			accer_z = record['accelerometer_values']['z']
			amp = math.sqrt(accer_x ** 2 + accer_y ** 2 + accer_z ** 2)
			timestamp = record['timestamp']
			self.amplitude.append(amp)
			self.time_series.append(timestamp)
		plt.plot(self.time_series, self.amplitude)
		plt.savefig('accer_plot.png')
		plt.show()

if __name__ == '__main__':
	pa = paceAnalyser('12rounds.data')
	pa.readData()
	pa.readPace()







