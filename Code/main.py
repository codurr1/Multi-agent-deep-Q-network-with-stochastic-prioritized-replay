if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("-f")
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float, help="learning rate")
    parser.add_argument("-tf", "--target_frequency", default=2, type=int, help="target weights replace steps")
    parser.add_argument("-bs", "--batch_size", default=50, type=int, help="batch size")
    parser.add_argument("-ga", "--gamma", default=0.7, type=float, help="reward decay rate")
    parser.add_argument("-e", "--epsilon", default=0.9, type=float, help="exploration rate")
    parser.add_argument("-c", "--memory_capacity", default=10000, type=int, help="replay memory capacity")
    parser.add_argument("-nn", "--number_nodes", default=100, type=int, help="number of nodes in each layer of neural network")
    parser.add_argument("-m", "--agent_number", default=3, type=int, help="total number of iiot devices")
    parser.add_argument("-uav", "--uav_number", default=5, type=int, help="total number of UAVs")
    parser.add_argument("-g", "--grid_width", default=800, type=int, help="size of fixed area under consideration")
    parser.add_argument("-H", "--uav_height" , default=100, type=int, help="flying height of UAV")
    parser.add_argument("-r", "--uav_range", default=300, type=int, help="communication range of UAV")
    parser.add_argument("-cl", "--local_compute", default=500, type=int, help="local computation capacity 'MHz'")
    parser.add_argument("-cu", "--uav_compute", default=2, type=int, help="UAV compution capacity 'GHz'")
    parser.add_argument("-cc", "--cloud_compute", default=100, type=int, help="cloud computation capacity 'GHz'")
    parser.add_argument("-rd", "--reference_distance", default=1, type=float, help="channel gain reference distance 'meters'")
    parser.add_argument("-lcp", "--los_channel_power", default=1.42e-4, type=float, help="channel gain at the reference")
    parser.add_argument("-ub", "--uav_bandwidth", default=15, type=int, help="bandwidth allocated for UAV uplin transmission rate 'MHz'")
    parser.add_argument("-cb", "--cloud_bandwidth", default=10, type=int, help="bandwidth allocated for cloud uplink transmission 'MHz'")
    parser.add_argument("-up","--uav_power", default=0.01, type=float, help="uplink transmission power for UAV offloading  'W'")
    parser.add_argument("-cp", "--cloud_power", default=0.015, type=float, help="uplink transmission power for cloud offloading  'W'")
    parser.add_argument("-n", "--noise_power", default=-90, type=float, help="background noise power  'dBm/Hz'")
    parser.add_argument("-ptf", "--propagation_time_factor", default=4e-9, type=float, help="uplink propogation delay factor  's/bit'")
    parser.add_argument("-lec", "--local_energy_consumption_factor", default=1e-23, type=float, help="local energy consumption factor 'theta' J/cycle")
    
    
    # values not given in paper
    #------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("-pf", "--punishment_factor", default=-2, type = float, help="if tolerant delay < energy consumption")  
    parser.add_argument("-p", "--priority_scale", default=0.4, type=float, help="scale for prioritization")  
    parser.add_argument("-m_e", "--min_epsilon", default=0.02, type=float, help="minimum value of exploration rate")
    parser.add_argument("-e_d", "--epsilon_decay", default=0.0001, type=float, help="exploration decay rate")
    parser.add_argument('-m_b', "--min_beta", default=0.4, type=float, help="minimum value of importance sampling")
    parser.add_argument("-b_max", "--beta_max", default=1.0, type=float, help="incrementing value of importance sampling beta")
    parser.add_argument("-ts", "--max_timesteps", default=20, type=int, help="maximum timesteps in each epsisode")  ## value not given in paper
    parser.add_argument("-ed", "--episodes", default=10 , type=int, help="total number of episodes")    ## value not given in paper 
    #-------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    # values range is given in paper
    # ----------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("-t", "--task_size", default=random.uniform(100, 800000), type=float, help="offloading task size  'Kb'")
    parser.add_argument("-cpu", "--cpu_cycle", default=random.uniform(5e+5, 5e+9), type=float, help="cpu_cycle required by task ")
    parser.add_argument("-d", "--tolerant_delay", default=random.uniform(0.01, 1), type=float, help="task tolerant value 'seconds'")
    #------------------------------------------------------------------------------------------------------------------------------------------
    np.set_printoptions(precision=2)

    args = vars(parser.parse_args())
    
    
    env = ENV(args)
   
    
    all_agents = []
    for agent_idx in range(args['agent_number']):
        all_agents.append(Agent(env.state_size, env.action_size, agent_idx, args))
    
    env.main(all_agents)
    
   
