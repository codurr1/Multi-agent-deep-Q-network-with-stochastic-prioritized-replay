# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class Agent(object):
    
    def __init__(self, state_size, action_size, agent_idx, arguments):
        
        self.state_size = state_size
        self.action_size = action_size
        self.agent_idx = agent_idx
        
        self.learning_rate = arguments['learning_rate']
        self.update_target_frequency = arguments['target_frequency']
        self.batch_size = arguments["batch_size"]
        
    
        self.gamma = arguments["gamma"]
        self.epsilon = arguments['epsilon']
        self.min_epsilon = arguments['min_epsilon']
        self.epsilon_decay = arguments['epsilon_decay']
        self.beta = arguments['min_beta']
        self.beta_max = arguments['beta_max']
        
        self.step = 0
        
        self.dqn_model = NeuralNetwork(self.state_size, self.action_size, arguments)
        self.memory = ReplayMemory(arguments['memory_capacity'], arguments['priority_scale'])
     
    
    def decay_epsilon(self):
        self.step +=1
        
        if (self.epsilon > self.min_epsilon):
            self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * math.exp(-1. * self.step * self.epsilon_decay)
            return self.epsilon
        else : 
            return self.min_epsilon
        
        if self.beta < self.beta_max:
            self.beta+=0.01
            #self.beta = self.beta + (self.beta - self.beta_max) * math.exp(-1. * self.step * 0.01)  #@@@@@@@@@@@ incrementing beta by 0.01
    
    def choose_action(self, state):
        exploration_rate = self.decay_epsilon()
        
        if exploration_rate > random.random():
            return random.randrange(-1,args['uav_number']+1)  #explore
        else : 
            return np.argmax(self.dqn_model.predict_one_sample(state))
        
                         
    def batch_error(self, batch):   # batch = [[priority, sample]] = [[0, (states, actions, rewards, next_states, done)]]
        
        
        batch_len = len(batch)
       
        states = np.array([batch[i][1][0] for i in range(batch_len)])
        states_ = np.array([batch[i][1][3] for i in range(batch_len)])
        
        predict = []
        predict_ =[]
        predict_target = []
        for i in range(batch_len):
           
            predict.append(self.dqn_model.predict(states[i]))
            predict_.append(self.dqn_model.predict(states_[i]))
            predict_target.append(self.dqn_model.predict(states_[i], target=True))
            
        x = np.zeros((batch_len, self.state_size))
        
        y = np.zeros ((batch_len, args['uav_number']+2))
       
        
        errors = np.zeros(batch_len)
        
        for i in range(batch_len):
            
            o = batch[i][1]                   # tuple = (state, action, reward, next_state, done)
           
            state = o[0]                     #state array 
            
            action = o[1][self.agent_idx] 
            
            reward = o[2]      # reward value 
            
            
            next_state = o[3]    # next_state
           
            done = o[4]
             
            q_value = predict[i]
            next_q_value = predict_target[i]
            
            if done: 
                target_q_value = reward
            else: 
                target_q_value  = reward + self.gamma * np.amax(next_q_value)
            
            
            x[i] = state
            y[i] = action 
            
            errors[i] = np.abs(target_q_value- np.amax(q_value))
        
     
        return [x, y, errors]
    
    
    def observe(self, sample):
        
        _, _, errors = self.batch_error([(0, sample)])
        
        self.memory.remember(sample, errors)
        
        
  
    
    def replay(self):
        
        [batch, batch_idx, batch_priorities] = self.memory.sample(self.batch_size)
        
        x, y, errors = self.batch_error(batch)
        
        
        normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
        
        # b_values = importance sampling weights 
        b_values = [(self.batch_size * i) ** (-1 * self.beta) for i in normalized_batch_priorities]
        
        normalized_b_values = [float(i) / max(b_values) for i in range(len(b_values))]
        
        sample_weights = [errors[i] * normalized_b_values[i] for i in range(len(errors))]
        
        self.dqn_model.train(x, y, np.array(sample_weights))
        
        self.memory.update(batch_idx, errors)
        
        
        
    def update_target_model(self):
        if self.step % self.update_target_frequency == 0 : 
            self.dqn_model.update_target_model()
            
            
