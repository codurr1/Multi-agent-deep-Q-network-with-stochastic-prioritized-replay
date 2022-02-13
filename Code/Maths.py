import math
class Maths(object):
    
    def __init__(self,args):
        
        self.num_agents = args['agent_number']
        self.num_uav = args['uav_number']
        self.grid_width = args['grid_width']
        self.uav_height = args['uav_height']
        self.uav_range = args['uav_range']
        self.local_compute = args['local_compute']
        self.uav_compute = args['uav_compute']
        self.cloud_compute = args['cloud_compute']
        self.reference_distance = args['reference_distance']
        self.los_channel_power = args['los_channel_power']
        self.uav_bandwidth = args['uav_bandwidth']
        self.cloud_bandwidth = args['cloud_bandwidth']
        self.uav_power = args['uav_power']
        self.cloud_power = args['cloud_power']
        self.noise_power = args['noise_power']
        self.propagation_time_factor = args['propagation_time_factor']
        self.local_energy_consumption_factor = args['local_energy_consumption_factor']
        self.task_size = args['task_size']
        self.cpu_cycle = args['cpu_cycle']
        self.tolerant_delay = args['tolerant_delay']
        self.punishment_factor = args['punishment_factor']
        
        

        # value not given in paper
        # -----------------------------------------------------------------
        self.a = 0.2   
        self.b = 0.3   
        self.attenuation_coefficient = 0.2
        self.nue_los = 0.3
        self.nue_nlos = 0.3
        self.cloud_channel_gain = 0.3   # H(t) used in equation 6     
        #-------------------------------------------------------------------
        
        
        
            
    def uav_channel_gain(self, uav_pos, iiot_pos):  # takes position of a uav and iiot_device.    
        
        #P_los = 1/(1+(self.a * math.exp(-self.b * (math.atan(self.uav_height/uav_pos[0]))) - self.a))      # [equation -1]
        P_los = 0.63
        P_nlos = 1 - P_los            
        
        a, b = uav_pos, iiot_pos
        distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)
        PL_los = self.los_channel_power * ((math.sqrt(self.uav_height**2 + (distance)**2)) **(-self.attenuation_coefficient)) * self.nue_los      # [equation-2]                                      
        
        PL_nlos = self.los_channel_power * ((math.sqrt(self.uav_height**2 + (distance)**2)) **(-self.attenuation_coefficient)) * self.nue_nlos      # [equation-3]
    
        h_channel_condition = (P_los * PL_los) + (P_nlos * PL_nlos)        # [equation - 4]
            
        return h_channel_condition
                   
    
    def uav_energy_consumption(self, uav_pos, iiot_pos):
        self.h_channel_condition = self.uav_channel_gain(uav_pos, iiot_pos)
        
    
        v1 = 1 + ((self.uav_power * self.h_channel_condition) / (self.noise_power) )    # value for log 
        uplink_transmission_rate = self.uav_bandwidth * (math.log(v1, 2))               # [equation - 5]  
        
        transmission_time = self.task_size / uplink_transmission_rate     # [equation -7]
          
        computation_time = self.cpu_cycle / self.uav_compute     #[equation -11]
                   
                   
        uav_energy = self.uav_power * (transmission_time + computation_time)   #[equation -12]
        
        return uav_energy
                   
                   
                   
    def cloud_energy_consumption(self):
         
        v1 = 1 + ((self.cloud_power * self.cloud_channel_gain) / (self.noise_power) ) 
        uplink_transmission_rate = self.cloud_bandwidth * (math.log(v1, 2))   #[equation -6]
        
        transmission_time = (self.task_size / uplink_transmission_rate) + self.propagation_time_factor     # [equation -8]
                   
        cloud_energy = self.cloud_power * (transmission_time + self.propagation_time_factor)   #[equation - 13]        
                   
        return cloud_energy
                   
                   
    
    def local_energy_consumption(self):
        return self.local_energy_consumption_factor * (self.cpu_cycle ** 2)      #[equation - 10]
    
                   
                   
        
