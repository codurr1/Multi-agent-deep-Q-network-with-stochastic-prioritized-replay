

class Environment(object):
    
    def __init__(self,agrs ):
        self.num_agents = args['agent_number']
        self.num_uav = args['uav_number']
        self.grid_width = args['grid_width']
        self.uav_height = args['uav_height']
        self.uav_range = args['uav_range']
        self.local_compute = args['local_compute']
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
        self.uav_compute = args["uav_compute"]
        
        # -----------------------
        # value not given in paper
        self.a = 0.2   
        self.b = 0.3   
        self.attenuation_coefficient = 0.2
        self.nue_los = 0.3
        self.nue_nlos = 0.3
        self.cloud_channel_gain = 0.3   # H(t) used in equation 6     
        # ------------------------
        
        
        self.action_space = np.arange(-1, self.num_uav+1)   # action_space = {-1, 0, 1, ...., N}
        self.users_space = np.zeros([self.num_agents], np.int32)
        self.users_observation = np.zeros([self.num_agents], np.int32)
        self.state_size = 3*(self.num_uav) + 6
        self.action_size = args['uav_number']+2    
        self.cloud_channel_gain = 0.3   # H(t) used in equation 6 
       
        self.UAVs_pos = self.UAVs_Position()
        self.iiot_pos=self.IIots_Position() 
        self.terminal = False
        
        self.Maths = Maths(args)
        
    
    def UAVs_Position(self): # choose random (x,y) position of UAV on grid with constant ht=uav_height 
        UAVs_pos = {}
        x = random.sample(range(self.grid_width),self.num_uav)
        y = random.sample(range(self.grid_width),self.num_uav)
        
        for i in range(1,self.num_uav+1):
            point = (x[i-1],y[i-1],self.uav_height)
            UAVs_pos[i] = point
        return  UAVs_pos    # list of uav_positions
    
     
        
    def IIots_Position(self): # choose random (x,y) position of iiot on grid
        iiot_pos = {}
        for i in range(self.num_agents):
            x = random.randint(1, self.grid_width)
            y = random.randint(1, self.grid_width)
            point = (x,y)
            iiot_pos[i] = point   # list of iiot_positions 
        return iiot_pos 
    
    
    def state(self):     # for each iiot_device state vector is different because it takes the current position of iiot_device
        uav_pos = [self.UAVs_pos[i] for i in range(1, self.num_uav+1)]
        l = []
        for i in uav_pos:
            l.append(i[0])
            l.append(i[1])
            
        all_states = []

        
        for i in range(self.num_agents):

            state = [self.task_size, self.cpu_cycle, self.tolerant_delay, self.iiot_pos[i][0], self.iiot_pos[i][1]]
            state.extend(l) # l =  uav_pos
            state.append(self.cloud_channel_gain) 
            h_channel_condition = [self.Maths.uav_channel_gain(j, self.iiot_pos[i]) for j in uav_pos] 
            state.extend(h_channel_condition)
            
            state = np.array(state)
            state = state.reshape(1,(args['uav_number']*3)+6)
            #state = tf.convert_to_tensor(state)
            all_states.append(state)
        # print(all_states)    
        return  all_states    #@@@@@@@@@@@@@@@@@@@@@@ is one_hot encoding required? 
         
            
            
            
    def reset(self):
        self.UAV_Pos()
        self.IIOT_Pos()
        
    
    def step(self, actions):  # it takes a list of actions 
        rewards = []
        for i,j in enumerate(actions): 
                if j > 0:               # uav task
                    r = self.reward(j, self.UAVs_pos[j], self.iiot_pos[i])
                
                elif j == 0:
                    r = self.reward(j)
                
                else : 
                    r = self.reward(j)
                
                rewards.append(r)
        
                
        if sum(rewards):      #@@@@@@@@@@@@@@@@@@@@@@  any specific condition? 
            self.terminal = [False for i in range(len(actions))]
        else: 
            self.terminal = True
            
        next_state = self.state()     #@@@@@@@@@@@@@@@@@@ how to get next_state?
        
        return next_state, rewards, self.terminal
                    
                    
     
    
    
    def reward_calculate(self, energy_consumption):
        
        if energy_consumption <= self.tolerant_delay: 
            return 1/energy_consumption
        else:
            return (1/energy_consumption)- self.punishment_factor 
        
        
    def reward(self, agent_action, uav_pos = None, iiot_pos = None):  ## Maths.uav_energy_consumption takes 2 parameters
        
        if agent_action == -1:   # offload to cloud
            energy_consumption = Maths.cloud_energy_consumption(self)   
            return self.reward_calculate(energy_consumption)
        
        elif agent_action == 0:  # compute locally
            energy_consumption = Maths.local_energy_consumption(self)
            return self.reward_calculate(energy_consumption)
        
        else: # offload to UAV
            obj = Maths(args)
            energy_consumption = obj.uav_energy_consumption(uav_pos, iiot_pos)
            return self.reward_calculate(energy_consumption)
        
    def sample(self):
        x =  np.random.choice(self.action_space,size=self.num_agents)
        return x   
        
