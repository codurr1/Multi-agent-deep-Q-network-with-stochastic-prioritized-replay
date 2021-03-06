
class ENV(object):
    def __init__(self, args):
        self.action_space = np.arange(-1, args['uav_number']+1)   # action_space = {-1, 0, 1, ...., N}
        self.users_space = np.zeros([args['agent_number']], np.int32)
        self.users_observation = np.zeros([args['agent_number']], np.int32)
        self.state_size = 3*(args['uav_number']) + 6
        self.action_size = args['uav_number']+2
        self.env = Environment(args)
        self.dqn_model = NeuralNetwork(self.state_size, self.action_size, args)
        self.step_b_updates = 1
        
    def main(self, agents): # MDSPR Algorithm
        total_step = 0
        rewards_list = []

        for episode in range(args['episodes']): 

            state = self.env.state()  # list of initial state vector for all iiot_devices.
            done = False
            reward_all = []
            time_step = 0
            
           
            while not done and time_step < args['max_timesteps']:
                
                actions = []   
                    
                for i in range(args['agent_number']): # using epsilon greedy to choose the action
                    actions.append(agents[i].choose_action(state[i])) # appending action of each agent into actions list. 
                
                
                next_state, reward, done = self.env.step(actions)
        
                for i in range(len(agents)): # experience replay
            
                    agents[i].observe((state[i], actions, reward[i], next_state[i], done[i]))

                    agents[i].decay_epsilon()

                   
                    
                    
                    if time_step % self.step_b_updates  and (time_step!=0)== 0 : 
                        
                        agents[i].replay()

                    if time_step % args['target_frequency'] and (time_step!=0) == 0:
                       
                        agents[i].update_target_model()
                        
                #print("total_step : ", total_step,"\n\n")
                time_step +=1
                total_step +=1
                state = next_state
                reward_all.append(reward)

            rewards_list.append(reward_all)
            print(f"episode {episode} -----reward : {reward_all} --------Final step : {time_step}, --------done : {done}\n\n\n\n\n")
