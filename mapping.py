import matplotlib.pyplot as plt
import json
import math
from scipy.stats import norm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import pdb

class mapping(object):
	def __init__(self, fn, params):
		self.filename = fn
		self.timestamp = []
		self.directions = {}
		self.data = ''
		self.params = params
		self.wifimarks = []
		self.gyros = []
		self.gyros_y = []
		self.clockwises = []
		self.gyro_ts = []
		self.mags = []
		self.amplitude = []
		self.time_series = []
		self.sorted_ts = []
		self.corr_cluster = []
		self.num_clusters = 0
		self.max_occurence = 0
		self.turning_dict = {}
		self.sign = True
		with open(fn, 'r') as f:
			data = json.load(f)
			self.data = data
			for record in data:
				self.timestamp.append(record['timestamp'])
		for record in self.data:
			if 'gyroscope_values' not in record or 'magnetometer_values' not in record:
				continue
			self.gyros.append(math.sqrt(record['gyroscope_values']['x'] ** 2 + record['gyroscope_values']['y'] ** 2 + record['gyroscope_values']['z'] ** 2))
			self.gyros_y.append(record['gyroscope_values']['y'])
			clockwise = 1 if record['gyroscope_values']['y'] > 0 else -1
			self.clockwises.append(clockwise)
			self.gyro_ts.append(record['timestamp'])
			self.mags.append(math.sqrt(record['magnetometer_values']['x'] ** 2 + record['magnetometer_values']['y'] ** 2 + record['magnetometer_values']['z'] ** 2))
			accer_x = record['accelerometer_values']['x']
			accer_y = record['accelerometer_values']['y']
			accer_z = record['accelerometer_values']['z']
			amp = math.sqrt(accer_x ** 2 + accer_y ** 2 + accer_z ** 2)
			timestamp = record['timestamp']
			self.amplitude.append(amp)
			self.time_series.append(timestamp)
		

	def smooth(self, int_list, timestamps, factor, number = 9):	# number should be odd number
		if len(int_list) < 2 * number:
			return [False, False]
		norm_filter = [norm.pdf(factor * (k - (number - 1) / 2)) for k in range(number)]
		result = []
		ts = []
		for ind in range(len(int_list)):
			if ind < (number - 1) / 2 or ind > len(int_list) - (number + 1) / 2:
				continue
			temp = [int_list[ind + i - (number - 1) / 2 - 1] * norm_filter[i] for i in range(number)]
			avg = sum(temp) / (sum(norm_filter))
			result.append(avg)
			ts.append(timestamps[ind])
		# print result
		# print '**********************'
		return [result, ts]

	def turn_dict(self):
		[smt_gyros, smt_ts] = self.smooth(self.gyros, self.gyro_ts, self.params['gyro_factor'], self.params['gyro_number'])
		turn_point_ts = []
		turn_point_value = []
		for i in range(1, len(smt_gyros) - 3):
			if smt_gyros[i] > smt_gyros[i - 1] and smt_gyros[i] > smt_gyros[i + 1]:
				if smt_gyros[i] > self.params['uturn_thresh1'] - 0.1 and (len(turn_point_ts) == 0 or smt_ts[i] - turn_point_ts[-1] > 1000):
					turn_point_ts.append(smt_ts[i])
					turn_point_value.append(smt_gyros[i])

		half_u_turn = True
		for i in range(len(turn_point_ts)):
			if turn_point_value[i] < self.params['turn_type_thresh']:
				label = 'normal_turn'
			elif half_u_turn == True:
				label = 'half_u_turn'
				half_u_turn = not half_u_turn
			else:
				label = 'period_u_turn'
				half_u_turn = not half_u_turn
			ind = smt_ts.index(turn_point_ts[i])
			k = 0
			low_stop_sign = 0
			high_stop_sign = 0
			low_bound = turn_point_ts[i]
			high_bound = turn_point_ts[i]
			while low_stop_sign * high_stop_sign == 0:
				k += 1
				if low_stop_sign == 0 and smt_gyros[ind - k] > self.params['uturn_thresh2']:
					low_bound = smt_ts[ind - k]
				else:
					low_stop_sign = 1
				if high_stop_sign == 0 and smt_gyros[ind + k] > self.params['uturn_thresh2']:
					high_bound = smt_ts[ind + k]
				else:
					high_stop_sign = 1
			self.turning_dict[(low_bound, high_bound)] = label

	def turn_type(self, timestamp):
		# turning_time = 2200
		# ind = self.gyro_ts.index(timestamp)
		# ts = self.gyro_ts
		# k = ind + 1
		# endpoint = timestamp
		# sign = 0
		# angle = 0
		# if k >= len(self.gyros) - 1:
		# 	return False
		# while(ts[k] - timestamp < turning_time):
		# 	interval = (ts[k] - ts[k - 1]) / 1000
		# 	angle += interval * self.gyros[k - 1] * self.clockwises[k - 1]
		# 	k += 1
		# 	if self.gyros[k] > self.params['uturn_thresh2'] and sign == 0:
		# 		endpoint = self.gyro_ts[k]
		# 	else:
		# 		sign = 1
		# 	if k >= len(self.gyros) - 1:
		# 		break
		# if abs(angle) > self.params['uturn_angle']:
		# 	return endpoint
		# else:
		# 	return False
		for k, v in self.turning_dict.items():
			low = k[0]
			high = k[1]
			if timestamp > low and timestamp < high:
				return v
		return 'no_turn'

	def get_direction(self):
		origin = self.params['origin_direction']
		last_ts = min(self.timestamp)
		last_direct = origin
		last_gyro = 0
		last_status = 'no_turn'
		before_turn = 0

		for record in self.data:
			current_status = self.turn_type(record['timestamp'])
			if last_status == 'no_turn' and (current_status == 'half_u_turn' or current_status == 'period_u_turn'):
				before_turn = last_direct	
			if (last_status == 'half_u_turn' or last_status == 'period_u_turn') and current_status == 'no_turn':
				current_direct = self.params['back_direct'] if last_status == 'half_u_turn' else origin
			else:
				gyro = last_gyro if 'gyroscope_values' not in  record else record['gyroscope_values']['y']
				interval = 1.0 * (record['timestamp'] - last_ts) / 1000
				current_direct = (last_direct + interval * gyro) % (2 * math.pi)
			self.directions[record['timestamp']] = current_direct
			last_ts = record['timestamp']
			last_direct = current_direct
			last_gyro = gyro
			last_status = current_status

		if self.params['vertical_turn']:
			for record in self.data:
				current_status = self.turn_type(record['timestamp'])
				if last_status == 'no_turn' and (current_status == 'normal_turn' or current_status == 'half_u_turn' or current_status == 'period_u_turn'):
					before_turn = last_direct	
				if (last_status == 'normal_turn' or last_status == 'half_u_turn' or last_status == 'period_u_turn') and current_status == 'no_turn':
					# current_direct = self.params['back_direct'] if last_status == 'half_u_turn' else origin
					if last_status == 'half_u_turn':
						current_direct = self.params['back_direct']
					elif last_status == 'period_u_turn':
						current_direct = origin
					else:
						gyro_ind = self.gyro_ts.index(record['timestamp'])
						direct_sum = sum([self.gyros_y[gyro_ind - x] for x in range(0,12)])
						turn_direct = 1 if direct_sum > 0 else -1
						current_direct = (before_turn + turn_direct * math.pi / 2) % (2 * math.pi)
				else:
					gyro = last_gyro if 'gyroscope_values' not in  record else record['gyroscope_values']['y']
					interval = 1.0 * (record['timestamp'] - last_ts) / 1000
					current_direct = (last_direct + interval * 0) % (2 * math.pi)
				self.directions[record['timestamp']] = current_direct
				last_ts = record['timestamp']
				last_direct = current_direct
				last_gyro = gyro
				last_status = current_status
		else:
			for record in self.data:
				current_status = self.turn_type(record['timestamp'])
				if last_status == 'no_turn' and (current_status == 'half_u_turn' or current_status == 'period_u_turn'):
					before_turn = last_direct	
				if (last_status == 'half_u_turn' or last_status == 'period_u_turn') and current_status == 'no_turn':
					current_direct = self.params['back_direct'] if last_status == 'half_u_turn' else origin
				else:

					gyro = 0 if 'gyroscope_values' not in  record  or abs(record['gyroscope_values']['y']) < self.params['uturn_thresh2'] else record['gyroscope_values']['y']
					interval = 1.0 * (record['timestamp'] - last_ts) / 1000
					current_direct = (last_direct + interval * gyro) % (2 * math.pi)
				self.directions[record['timestamp']] = current_direct
				last_ts = record['timestamp']
				last_direct = current_direct
				last_gyro = gyro
				last_status = current_status


		# origin = self.params['origin_direction']
		# last_ts = min(self.timestamp)
		# last_direct = origin
		# last_clock = 1
		# last_gyro = 0
		# turning = 'no'
		# full = False
		# endpoint = 0
		# for record in self.data:
		# 	if 'gyroscope_values' not in  record:
		# 		interval = 1.0 * (record['timestamp'] - last_ts) / 1000
		# 		clockwise = last_clock
		# 		gyro = last_gyro
		# 		current_direct = (last_direct + interval * gyro * clockwise) % (2 * math.pi)
		# 		self.directions[record['timestamp']] = current_direct
		# 		last_ts = record['timestamp']
		# 		last_direct = current_direct
		# 		last_clock = clockwise
		# 		last_gyro = gyro
		# 		continue
		# 	if record['timestamp'] > endpoint:
		# 		turning = 'no'
		# 	else:
		# 		interval = 1.0 * (record['timestamp'] - last_ts) / 1000
		# 		clockwise = 1 if record['gyroscope_values']['y'] > 0 else -1
		# 		gyro = math.sqrt(record['gyroscope_values']['x'] ** 2 + record['gyroscope_values']['y'] ** 2 + record['gyroscope_values']['z'] ** 2)
		# 		current_direct = (last_direct + interval * gyro * clockwise) % (2 * math.pi)
		# 		self.directions[record['timestamp']] = current_direct
		# 		last_ts = record['timestamp']
		# 		last_direct = current_direct
		# 		last_clock = clockwise
		# 		last_gyro = gyro
		# 		continue
		# 	uturn = self.is_U_turn(record['timestamp'])
		# 	if not uturn == False:
		# 		turning = 'yes'
		# 		endpoint = uturn
		# 		clockwise = 1 if record['gyroscope_values']['y'] > 0 else -1
		# 		gyro = math.sqrt(record['gyroscope_values']['x'] ** 2 + record['gyroscope_values']['y'] ** 2 + record['gyroscope_values']['z'] ** 2)
		# 		current_direct = (last_direct + math.pi) % (2 * math.pi) if not full else origin
		# 		full = not full
		# 		self.directions[record['timestamp']] = current_direct
		# 		last_direct = current_direct
		# 		last_ts = record['timestamp']
		# 		last_clock = clockwise
		# 		last_gyro = gyro
		# 		continue
			
		# 	interval = 1.0 * (record['timestamp'] - last_ts) / 1000
		# 	clockwise = 1 if record['gyroscope_values']['y'] > 0 else -1
		# 	gyro = math.sqrt(record['gyroscope_values']['x'] ** 2 + record['gyroscope_values']['y'] ** 2 + record['gyroscope_values']['z'] ** 2)
		# 	current_direct = (last_direct + interval * gyro * clockwise) % (2 * math.pi)
		# 	self.directions[record['timestamp']] = current_direct
		# 	last_ts = record['timestamp']
		# 	last_direct = current_direct
		# 	last_clock = clockwise
		# 	last_gyro = gyro

	def displacement(self, ts1, ts2):
		ind1 = self.time_series.index(ts1)
		ind2 = self.time_series.index(ts2)
		displace = [0, 0]
		step_len = self.params['step_length']
		last_ts = 0
		# print self.turn_type(ts1)
		# print self.turn_type(ts2)
		if self.turn_type(ts1) == 'normal_turn':
			self.sign = False
		for i in range(ind1 + 1, ind2 - 1):
			# sign = 0
			# ts = self.time_series[i]
			# if ts not in self.gyro_ts:
			# 	sign = 1
			# else:
			# 	ind = self.gyro_ts.index(ts)
			# 	if abs(self.gyros[ind]) < self.params['uturn_thresh1']:
			# 		sign = 1
			ts = self.time_series[i]
			if self.turn_type(ts) != 'no_turn':
				continue
			if self.amplitude[i] > self.amplitude[i - 1] and self.amplitude[i] > self.amplitude[i + 1] and ts - last_ts > 300:
				direct = self.directions[ts]
				displace_x = step_len * math.cos(direct)
				displace_y = step_len * math.sin(direct)
				displace[0] += displace_x
				displace[1] += displace_y
				last_ts = ts
				# print 'ssssssssssssssssss ' + str(direct)
				# print displace
			# if self.sign:
			# 	print ts1
			# 	print ts2
			# 	print displace
			# 	print '############'
		# print displace
		# print '888888888888'
		print str(ts1) + ', ' + str(ts2) + ': ' + str(ts2 - ts1)
		print displace
		return displace

	def find_8common_bssid(self, rssi_dict):
		lengths = [0, 0, 0, 0, 0, 0, 0, 0]
		corr_bssid = [0, 0, 0, 0, 0, 0, 0, 0]
		for k, v in rssi_dict.items():
			if len(v) > min(lengths) and max(v['RSSI']) > self.params['thresh']:
				ind = lengths.index(min(lengths))
				lengths[ind] = len(v)
				corr_bssid[ind] = k
		if 0 in corr_bssid:
			corr_bssid.remove(0)
		return corr_bssid

	def find_marks(self, rssi_dict):

		def is_tipping_point(l1, ind):
			ref = l1[ind] - l1[ind - 1]
			sampling = [1,3,5,9,15]
			for s in sampling:
				if ind - s in range(len(l1)):
					if ref * (l1[ind] - l1[ind - s]) < 0:
						return False
				if ind + s in range(len(l1)):
					if ref * (l1[ind] - l1[ind + s]) < 0:
						return False
			return True

		ts_shrink = 1000000
		common_bssid = self.find_8common_bssid(rssi_dict)
		mark_dist = {}
		for k, v in rssi_dict.items():
			last_ts = 0
			rssi = v['RSSI']
			ts = v['timestamp']
			[rssi, ts] = self.smooth(rssi, ts, self.params['factor'], self.params['number'])
			if rssi == False:
				continue
			if max(rssi) < self.params['thresh']:
				continue
			for i in range(1, len(rssi) - 1):
				timestamp = ts[i]
				if is_tipping_point(rssi, i):
					last_ts = timestamp
					mark = []
					# mark.append(rssi[i])            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
					for n in range(len(common_bssid)):
						thisId = common_bssid[n]
						id_rssi = rssi_dict[thisId]
						if timestamp in id_rssi['timestamp']:
							ind = id_rssi['timestamp'].index(timestamp)
							mark.append(id_rssi['RSSI'][ind])
						else:
							mark.append(-95)
						
					ind = self.time_series.index(timestamp)
					mark.append(self.mags[ind] * 1.5)
					mark.append(float(timestamp) / ts_shrink)
					self.wifimarks.append(mark)
		[smt_gyros, smt_ts] = self.smooth(self.gyros, self.gyro_ts, self.params['gyro_factor'], self.params['gyro_number'])
		last_status = 'no_turn'
		turn_index = 0
		index_change = 1
		last_tsts = 0

		test_runs = self.params['run_times']
		run_time_count = 0
		for n in range(1, len(smt_ts) - 1):
			current_status = self.turn_type(smt_ts[n])
			if last_status == 'no_turn' and current_status == 'period_u_turn':
				run_time_count += 1
			if run_time_count > test_runs:
				break
			if last_status == 'period_u_turn' and current_status == 'no_turn':
				print smt_ts[n] - last_tsts
				print '***********'
				last_tsts = smt_ts[n]
				turn_index = 0
				index_change = 1
			if last_status == 'half_u_turn' and current_status == 'no_turn':
				turn_index += 1
				index_change = -1
			if (last_status == 'no_turn' and current_status == 'normal_turn') or (last_status == 'period_u_turn' and current_status == 'no_turn') or (last_status == 'half_u_turn' and current_status == 'no_turn'): # need revision
				turn_index += index_change
				if last_status == 'period_u_turn' and current_status == 'no_turn':
					turn_index = 0
				if last_status == 'half_u_turn' and current_status == 'no_turn':
					turn_index = 5
				print turn_index
				mark = []
				clock = self.clockwises[n]
				# mark.append(clock * 100)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(100 * turn_index)
				mark.append(float(smt_ts[n]) / ts_shrink)
				self.wifimarks.append(mark)
			last_status = current_status
		return

	def cluster(self):
		maxd = self.params['max_d']
		# print self.wifimarks
		z = linkage(self.wifimarks, 'average', 'euclidean')
		plt.figure(figsize = (25, 10))
		plt.title('Hierarchical Clustering Dendrogram')
		plt.xlabel('wifimarks')
		plt.ylabel('distance')
		dendrogram(
		    z,
		    leaf_rotation = 90.,  # rotates the x axis labels
		    leaf_font_size = 3.,  # font size for the x axis labels
		)
		plt.savefig('dendrogram.png')
		plt.clf()
		# plt.show()
		clusters = fcluster(z, maxd, criterion='distance')
		clusters = clusters.tolist()
		marks = self.wifimarks
		self.num_clusters = max(clusters)
		for i in range(1, max(clusters) + 1):
			if i >= len(clusters):
				break
			count = clusters.count(i)
			if count > self.max_occurence:
				self.max_occurence = count
			if count < self.params['minimum_rounds']:
				self.num_clusters -= 1
				while i in clusters:
					ind = clusters.index(i)
					del clusters[ind]
					del marks[ind]
		for m in marks:
			ts = int(m[-1] * 1000000)
			self.sorted_ts.append(ts)
		self.sorted_ts.sort()
		for t in self.sorted_ts:
			for m in marks:
				ts = int(m[-1] * 1000000)
				if ts == t:
					ind = marks.index(m)
					self.corr_cluster.append(clusters[ind])
					break
		for c in range(1, max(clusters) + 1):
			if c not in self.corr_cluster:
				continue
			ts_ind = [i for i in range(len(self.corr_cluster)) if self.corr_cluster[i] == c]
			ts_seq = [self.sorted_ts[i] for i in ts_ind]
			corr_x = [i for i in range(len(ts_ind))]
			plt.scatter(ts_seq, corr_x)
			plt.legend()
			plt.savefig('cluster_' + str(c)  + '.png')
			plt.clf()
			# plt.show()


		# print self.sorted_ts
		# print self.corr_cluster

	def Arturia(self):
		def list_op(l1, l2, method):
			res = [0, 0]
			if method == 'minus':
				for i in range(len(l1)):
					res[i] = l1[i] - l2[i]
				return res
			elif method == 'plus':
				for i in range(len(l1)):
					res[i] = l1[i] + l2[i]
				return res
			else:
				for i in range(len(l1)):
					res[i] = l1[i] * l2
				return res

		def energy(ppij, rrij, ffill_ij, height, width):
			penergy = 0
			for i in range(height):
				for j in range(width):
					if ffill_ij[i, j] > 0:
						average = [0, 0]
						nums = ffill_ij[i, j]
						for n in range(int(nums)):
							# print rij[i, j, n]
							average = list_op(average, rrij[i, j, n], 'plus')
						average = list_op(average, 1.0 / nums, 'multiply')
						# print average
						# print 'bbbbbbbbbbbbbbbbbbb'
						dij = list_op(ppij[j], ppij[i], 'minus')
						penergy += math.sqrt((average[0] - dij[0]) ** 2 + (average[1] - dij[1]) ** 2)
			return penergy

		n_clusters = self.num_clusters
		max_cluster_ind = max(self.corr_cluster)
		pij = np.zeros(shape = (max_cluster_ind + 1, 2))
		rij = np.zeros(shape = (max_cluster_ind + 1, max_cluster_ind + 1, self.max_occurence ** 2, 2))
		fill_ij = np.zeros(shape = (max_cluster_ind + 1, max_cluster_ind + 1))
		for i in range(len(self.sorted_ts) - 1):
			current_cluster = self.corr_cluster[i]
			k = i + 1
			if k >= len(self.corr_cluster) - 1:
				break
			while self.corr_cluster[k] == current_cluster:
				k += 1
				if k >= len(self.corr_cluster) - 1:
					break
				continue
			if k >= len(self.corr_cluster) - 1:
					break
			next_cluster = self.corr_cluster[k]
			while self.corr_cluster[k] == next_cluster:
				displace = self.displacement(self.sorted_ts[i], self.sorted_ts[k])
				rij[current_cluster, next_cluster, int(fill_ij[current_cluster, next_cluster])] = displace
				fill_ij[current_cluster, next_cluster] += 1
				k += 1
				if k >= len(self.corr_cluster) - 1:
					break
		k = 0
		while(k < 30):
			for i in range(max_cluster_ind + 1):
				for j in range(max_cluster_ind + 1):
					if fill_ij[i, j] > 0:
						average = [0, 0]
						nums = fill_ij[i, j]
						for n in range(int(nums)):
							# print rij[i, j, n]
							average = list_op(average, rij[i, j, n], 'plus')
						average = list_op(average, 1.0 / nums, 'multiply')
						# print average
						# print 'bbbbbbbbbbbbbbbbbbb'
						dij = list_op(pij[j], pij[i], 'minus')
						modify = list_op(average, dij, 'minus')
						adjustment = list_op(modify, self.params['spring_step'], 'multiply')
						pij[j] = list_op(pij[j], adjustment, 'plus')
				# current_cluster = self.corr_cluster[i]
				# n = i + 1
				# if n >= len(self.corr_cluster) - 1:
				# 	break
				# while self.corr_cluster[n] == current_cluster:
				# 	n += 1
				# 	if n >= len(self.corr_cluster) - 1:
				# 		break
				# 	continue
				# if n >= len(self.corr_cluster) - 1:
				# 		break
				# next_cluster = self.corr_cluster[n]
				# displace = self.displacement(self.sorted_ts[i], self.sorted_ts[n])
				# dij = list_op(pij[next_cluster], pij[current_cluster], 'minus')
				# modify = list_op(displace, dij, 'minus')
				# adjustment = list_op(modify, self.params['spring_step'], 'multiply')
				# pij[next_cluster] = list_op(pij[next_cluster], adjustment, 'plus')
			p_energy = energy(pij, rij, fill_ij, max_cluster_ind + 1, max_cluster_ind + 1)
			# print p_energy
			k += 1
		# print pij
		x = [pij[k, 0] for k in range(len(pij))]
		y = [pij[k, 1] for k in range(len(pij))]
		while 0 in x:
			x.remove(0)
			y.remove(0)
		plt.scatter(x, y)
		plt.legend()
		plt.savefig('dot_map_' + str(self.params['run_times']) + 'rounds_' + 'vertical_' + str(self.params['vertical_turn']) + '.png')
		plt.clf()
		# plt.show()


			# rij[current_cluster, next_cluster, fill_ij[current_cluster, next_cluster]] = displace
			# fill_ij[current_cluster, next_cluster] += 1


		# print fill_ij
		# print rij


	def rssi_analyzer(self):
		thresh = self.params['thresh']
		rssi_dict = {}
		mags = []
		timestamps = []
		accels = []
		gyros = []
		self.turn_dict()
		for record in self.data:
			if 'access_points' not in record or 'accelerometer_values' not in record or 'gyroscope_values' not in record or 'magnetometer_values' not in record:
				continue
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

		self.get_direction()
		self.find_marks(rssi_dict)
		# print self.wifimarks
		self.cluster()
		kk = []
		vv = []
		for k, v in self.directions.items():
			kk.append(k)
			vv.append(v)
		plt.scatter(kk, vv)
		plt.legend()
		plt.savefig('direction.png')
		plt.clf()
		# plt.show()
		self.Arturia()

		for k, v in rssi_dict.items():
			lb = k
			rssi = v['RSSI']
			ts = v['timestamp']
			[rssi, ts] = self.smooth(rssi, ts, self.params['factor'], self.params['number'])

			if rssi == False:
				continue
			if max(rssi) > thresh:
				plt.plot(ts, rssi, label = str(lb))
		plt.legend()
		plt.savefig('rssi.png')
		plt.clf()
		# plt.show()


		[smt_mags, smt_ts] = self.smooth(mags, timestamps, 0.05, 50)
		plt.plot(smt_ts, smt_mags)
		plt.legend()
		plt.savefig('magnet.png')
		plt.clf()
		# plt.show()

		# [smt_accels, smt_ts] = self.smooth(accels, timestamps, 0.05, 50)
		plt.plot(timestamps, accels)
		plt.legend()
		plt.savefig('accelerate.png')
		plt.clf()
		# plt.show()

		[smt_gyros, smt_ts] = self.smooth(self.gyros, self.gyro_ts, 0.3, 20)
		plt.plot(smt_ts, smt_gyros)
		plt.legend()
		plt.savefig('gyroscope.png')
		plt.clf()
		# plt.show()

	

if __name__ == '__main__':
	params = {'run_times': 10, 'vertical_turn': True, 'back_direct': 0, 'turn_type_thresh': 1.08, 'gyro_y_only': True, 'spring_step': 0.4, 'gyro_thresh': 0.75, 'gyro_factor': 0.3, 'gyro_number':20, 'minimum_rounds': 8, 'max_d': 14, 'step_length': 2, 'thresh': -62, 'factor': 0.03, 'number': 200, 'gyro_factor': 0.2, 'gyro_number': 30, 'origin_direction': 0, 'uturn_thresh1': 0.6, 'uturn_thresh2': 0.4, 'uturn_angle': 2.4}
	for i in range(10):
		params['vertical_turn'] = True
		params['run_times'] = i + 1
		params['minimum_rounds'] = i - 1
		fe = mapping('complex_path.data_json', params)
	# fe.get_direction()
		fe.rssi_analyzer()

		params['vertical_turn'] = False
		fe = mapping('complex_path.data_json', params)
	# fe.get_direction()
		fe.rssi_analyzer()

	l1 = 58
	l2 = 13
	l3 = 46


