import matplotlib.pyplot as plt
import json
import math
from scipy.stats import norm
import pdb

class feature_extractor(object):
	def __init__(self, fn):
		self.filename = fn
		self.timestamp = []
		self.data = ''
		with open(fn, 'r') as f:
			data = json.load(f)
			self.data = data
			for record in data:
				self.timestamp.append(record['timestamp'])

	def smooth(self, int_list, timestamps, factor, number = 9):	# number should be odd number
		if len(int_list) < 2 * number:
			return [False, False]
		print int_list
		norm_filter = [norm.pdf(factor * (k - (number - 1) / 2)) for k in range(number)]
		print int_list
		print norm_filter
		result = []
		ts = []
		for ind in range(len(int_list)):
			if ind < (number - 1) / 2 or ind > len(int_list) - (number + 1) / 2:
				continue
			temp = [int_list[ind + i - (number - 1) / 2] * norm_filter[i] for i in range(number)]
			avg = sum(temp) / (sum(norm_filter))
			result.append(avg)
			ts.append(timestamps[ind])
		print result
		print '**********************'
		return [result, ts]

	def rssi_analyzer(self):
		thresh = -60
		rssi_dict = {}
		mags = []
		timestamps = []
		accels = []
		gyros = []
		for record in self.data:
			points = record['access_points']
			mag = math.sqrt(record['magnetometer_values']['x'] ** 2 + record['magnetometer_values']['y'] ** 2 + record['magnetometer_values']['z'] ** 2)
			accel = math.sqrt(record['accelerometer_values']['x'] ** 2 + record['accelerometer_values']['y'] ** 2 + record['accelerometer_values']['z'] ** 2)
			gyro = math.sqrt(record['gyroscope_values']['x'] ** 2 + record['gyroscope_values']['y'] ** 2 + record['gyroscope_values']['z'] ** 2)
			ts = record['timestamp']
			mags.append(mag)
			timestamps.append(ts)
			accels.append(accel)
			gyros.append(gyro)
			for point in points:
				rssi = point['RSSI']
				bssid = point['BSSID']
				if bssid in rssi_dict:
					rssi_dict[bssid]['RSSI'].append(rssi)
					rssi_dict[bssid]['timestamp'].append(ts)
				else:
					rssi_dict[bssid] = {'RSSI': [rssi], 'timestamp': [ts]}
		for k, v in rssi_dict.items():
			print k
			lb = k
			rssi = v['RSSI']
			ts = v['timestamp']
			[rssi, ts] = self.smooth(rssi, ts, 0.5)
			if rssi == False:
				continue
			if max(rssi) > thresh:
				plt.plot(ts, rssi, label = str(lb))
		plt.legend()
		plt.savefig('rssi.png')
		plt.show()

		[smt_mags, smt_ts] = self.smooth(mags, timestamps, 0.5)
		plt.plot(smt_ts, smt_mags)
		plt.legend()
		plt.savefig('magnet.png')
		plt.show()

		[smt_accels, smt_ts] = self.smooth(accels, timestamps, 0.5)
		plt.plot(smt_ts, smt_accels)
		plt.legend()
		plt.savefig('accelerate.png')
		plt.show()

		[smt_gyros, smt_ts] = self.smooth(gyros, timestamps, 0.5)
		plt.plot(smt_ts, smt_gyros)
		plt.legend()
		plt.savefig('gyroscope.png')
		plt.show()



if __name__ == '__main__':
	fe = feature_extractor('Oct_24.data_json')
	fe.rssi_analyzer()


