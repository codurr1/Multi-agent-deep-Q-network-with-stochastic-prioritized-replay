import random  

class ReplayMemory(object):
    e = 0.05    # to avoid 0 probability of experiences 
    
    def __init__(self, capacity, priority_scale):
        self.capacity = capacity 
        self.priority_scale = priority_scale      # a in formula, used for balance b/w high priority and random sampling
        self.max_priority = 0 
        
        self.memory = SumTree(self.capacity)    
       
    def get_priority(self, TDerror):
        return (TDerror + self.e) ** self.priority_scale
    
    def remember(self, sample, TDerror):
        priority = self.get_priority(TDerror)
        self_max = max(self.max_priority, priority)
        self.memory.add(self_max, sample)
        
        
    def sample(self, batch_size):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        
        num_segments = self.memory.total() / batch_size
        
        for i in range(batch_size):
            left = num_segments * i 
            right = num_segments * (i + 1)
            
            s = random.uniform(left, right)
            idx, priority, data = self.memory.get(s)
            
            sample_batch.append((idx,data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(priority)
            
        return [sample_batch, sample_batch_indices, sample_batch_priorities]
    
    
    def update(self, batch_indices, errors):
        
        for i in range(len(batch_indices)):
            priority = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], priority)
        
        
        
        
