# Bloom Filter
# Run in terminal: ./bloom_filter.py 5 input.txt test.txt
import numpy as np
import sys

n=int(sys.argv[1])
input_text = sys.argv[2]
test_text = sys.argv[3]

def h1(s, n):
    hash = 5381
    for c in s :
        hash =((hash << 5) + hash) + ord(c)
    return hash % n

def h2(s, n):
    hash = 0
    for c in s:
        hash += ord(c)
    return hash % n

class BloomFilter:
    def __init__(self,n, hash_functions):
        self.B = np.zeros(n)
        self.hash_functions = hash_functions
        
    def add(self,element):
        pos_to_one = self.pos_for_string(element)
        self.B[pos_to_one] = 1
    
    def pos_for_string(self, element):
        return [f(element) for f in self.hash_functions]
        
    def contains(self, element):
        pos_to_check = self.pos_for_string(element)
        return np.all(self.B[pos_to_check]==1)
    
def read_strings(filename):
    f = open(filename, 'r')
    str_list = []
    for line in f:
        str_list.append(line.strip())
    return str_list

def print_result(pos, isin):
    rs = ''
    for i in pos:
        rs+=str(i)+','
    if isin:
        rs+='T'
    else:
        rs+='F'
    print(rs)
    
def run_bloom_filter(n, input_text, test_text):
    input_list = read_strings(input_text)
    test_list = read_strings(test_text)
    h1_n = lambda s: h1(s, n)
    h2_n = lambda s: h2(s, n)
    BFilter = BloomFilter(n,[h1_n,h2_n])
    for s in input_list:
        BFilter.add(s)
    for s in test_list:
        pos = BFilter.pos_for_string(s)
        isin = BFilter.contains(s)
        print_result(pos,isin)

run_bloom_filter(n, input_text, test_text)

