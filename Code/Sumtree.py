import numpy as np 

class SumTree(object):
    
    def __init__(self, capacity):
        self.write = 0 
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        
        # overwrite to the first index if the memory capacity is completed
        if self.write >= self.capacity:
            self.write = 0
            
    def total(self):
        return self.tree[0]
            
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        
        self.tree[idx] = priority
        
        # propagate the change through tree
        while idx !=0:
            idx = (idx -1) // 2
            self.tree[idx] +=change
     
    def retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s<= self.tree[left]:
            return self.retrieve(left, s)
        else : 
            return self.retrieve(right, s - self.tree[left])
        
            
            
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[dataIdx]   
        # here returning leaf index, priority value and experience of that leaf index
        
        
        