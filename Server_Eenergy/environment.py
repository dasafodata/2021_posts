# Import libraries
import numpy as np

#Create the class and its methods
class Environment(object):
	def __init__(self, optimal_temp = (15.0, 25.0), initial_month = 0, initial_number_users = 8, initial_rate_data = 80):
		self.monthly_atmospheric_temp = [1.8, 2.5, 6, 9.5, 13.6, 16.7, 18.3, 18.1, 15.1, 10.5, 6.1, 2.9]
		self.initial_month = initial_month
		self.atmospheric_temp = self.monthly_atmospheric_temp[initial_month]
		self.optimal_temp = optimal_temp
		self.min_temp = -25
		self.max_temp = 80
		self.min_number_users = 8
		self.max_number_users = 120
		self.max_update_users = 6
		self.min_rate_data = 20
		self.max_rate_data = 400
		self.max_update_data = 10
		self.initial_number_users = initial_number_users
		self.current_number_users = initial_number_users
		self.initial_rate_data = initial_rate_data
		self.current_rate_data = initial_rate_data
		self.intrinsec_temp = self.atmospheric_temp + 1.25*self.current_number_users+1.25*self.current_rate_data
		self.temp_ai = self.intrinsec_temp
		self.temp_noai = (self.optimal_temp[0]+self.optimal_temp[1])/2.0
		self.total_energy_ai = 0.0
		self.total_energy_noai = 0.0
		self.reward = 0.0
		self.game_over = 0
		self.train = 1
		
	def update_env(self, direction, energy_ai, month):
		# GETTING THE REWARD

		# Energy spended by cooling system server (no-IA)
		energy_noai = 0
		if(self.temp_noai  < self.optimal_temp[0]):
		    energy_noai = self.optimal_temp[0] - self.temp_noai
		    self.temp_noai = self.optimal_temp[0]
		elif(self.temp_noai > self.optimal_temp[1]):
		    energy_noai = self.temp_noai - self.optimal_temp[1]
		    self.temp_noai = self.optimal_temp[1]

		# The Reward
		self.reward = energy_noai - energy_ai
		# Scaled the reward
		self.reward = 1e-3*self.reward

		# GETTING THE NEXT STATE

		# Updating the atmospheric temp
		self.atmospheric_temp = self.monthly_atmospheric_temp[month]
		# Updating the number of users
		self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
		if(self.current_number_users < self.min_number_users):
		    self.current_number_users = self.min_number_users
		elif(self.current_number_users > self.max_number_users):
		    self.current_number_users = self.max_number_users
		# Updating the current rate data
		self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
		if(self.current_rate_data < self.min_rate_data):
		    self.current_rate_data = self.min_rate_data
		elif(self.current_rate_data > self.max_rate_data):
		    self.current_rate_data = self.max_rate_data
		# Intrinsic temperature variation
		past_intrinsic_temp =  self.intrinsec_temp #previous temperature 
		self.intrinsec_temp = self.atmospheric_temp + 1.25*self.current_number_users+1.25*self.current_rate_data
		delta_intrinsec_temp = self.intrinsec_temp - past_intrinsic_temp
		# Temperature variation caused by IA
		if(direction==-1): #if temperature down 
		    delta_temp_ai = -energy_ai
		elif(direction == 1): #if temperature up
		    delta_temp_ai = energy_ai
		# New server temperature when IA is connected
		self.temp_ai += delta_intrinsec_temp + delta_temp_ai
		# New server temperature when IA is disabled
		self.temp_noai += delta_intrinsec_temp

		# GETTING THE GAME OVER
		if(self.temp_ai < self.min_temp):
		    if(self.train == 1):
		        self.game_over = 1
		    else:
		        self.total_energy_ai += self.optimal_temp[0] - self.temp_ai
		        self.temp_ai = self.optimal_temp[0]
		if(self.temp_ai > self.max_temp):
		    if(self.train == 1):
		        self.game_over = 1
		    else:
		        self.total_energy_ai += self.temp_ai - self.optimal_temp[1]
		        self.temp_ai = self.optimal_temp[1]

		# UPDATING THE SCORES

		# Total Energy spends by IA
		self.total_energy_ai += energy_ai
		# Total Energy spends by no-IA (without IA)
		self.total_energy_noai += energy_noai


		# SCALING NEXT STATE
		scaled_temp_ai = (self.temp_ai - self.min_temp) / (self.max_temp - self.min_temp)
		scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
		scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
		next_state = np.matrix([scaled_temp_ai, scaled_number_users, scaled_rate_data])

		# RETURN NEXT STATE, REWARD AND GAME OVER
		return next_state, self.reward, self.game_over
		
	def reset_env(self, new_month):
		self.atmospheric_temp = self.monthly_atmospheric_temp[new_month]
		self.initial_month = new_month
		self.current_number_users = self.initial_number_users
		self.current_rate_data = self.initial_rate_data
		self.intrinsec_temp = self.atmospheric_temp + 1.25*self.current_number_users+1.25*self.current_rate_data
		self.temp_ai = self.intrinsec_temp
		self.temp_noai = (self.optimal_temp[0]+self.optimal_temp[1])/2.0
		self.total_energy_ai = 0.0
		self.total_energy_noai = 0.0
		self.reward = 0.0
		self.game_over = 0
		self.train = 1

	def give_env(self):
		scaled_temp_ai = (self.temp_ai - self.min_temp)/(self.max_temp - self.min_temp)
		scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
		scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
		current_state = np.matrix([scaled_temp_ai, scaled_number_users, scaled_rate_data])

		return current_state, self.reward, self.game_over

