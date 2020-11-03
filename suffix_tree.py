# Suffix Tree
# Run in terminal: suffix_tree.py "babacacb$"
import sys
s = sys.argv[1] # string for the suffix tree 
# From the dot file a pdf can be constructed with the following call (in linux terminal):
# dot -Tps2 output.dot -o output.ps | ps2pdf output.ps

class SuffixTree:
    def __init__(self):
        self.text = ""
        root_node = Node(0,None,None)
        self.nodes = [root_node]
        self.num_nodes = 1
    
    def move_down(self,active_pos,c):
        """
        There are multiple cases:
        a) if the depth is 0, we look at the first letters on the children, if there is no 'c' among them, return None
        b) if the next letter on the edge is not 'c', return none
        c) if we found 'c' and we do not reach a new node, then return the same node_id with the new char and depth on the edge
        d) if we found 'c' and reach a new node, return the new node with depth 0 and char ''.
        """
        node_id, letter, depth = active_pos
        node = self.nodes[node_id]
        if depth == 0: #a)
            if c in node.children:
                next_node = node.children[c]
            else:
                return None
        else:
            next_node = node.children[letter]
        if self.text[next_node.start+depth] != c: #b)
            return None
        
        elif next_node.start+depth+1 == next_node.end: #d)
            return (next_node.idx, '',0)
        else:
            return (node_id, self.text[next_node.start], depth+1) #c)
    
    def new_node(self,active_pos, i, j):
        #update the nodes accordingly if we need new internal nodes or not.
        node_id, letter, depth = active_pos
        if depth != 0:
            next_node = self.nodes[node_id].children[letter]
            inner_node = Node(self.num_nodes,next_node.start, next_node.start+depth, parent = node_id)
            self.nodes[node_id].children[letter] = inner_node
            next_node.start = next_node.start+depth
            next_node.parent = inner_node.idx
            inner_node.children[self.text[next_node.start]] = next_node
            self.num_nodes += 1
            self.nodes.append(inner_node)
            node_id = inner_node.idx
        n_node = Node(self.num_nodes, i, j, parent = node_id)
        self.nodes[node_id].children[self.text[i]] = n_node
        self.nodes.append(n_node)
        self.num_nodes += 1
        return node_id
    
    
    def add_suffix_link(self,node_i, node_j):
        self.nodes[node_i].suffix_link = node_j
    
    def use_suffix_link(self,active_pos):
        node_id, letter, depth = active_pos
        node = self.nodes[node_id]
        if node.suffix_link == None:
            # if the root node is the parent and depth is not zero, we have to find the suffix manually,
            #since we have no suffix link
            if depth > 0:
                next_node = self.nodes[0]     
                letter = self.text[node.children[letter].start+1]
                depth -=1
            else:
                return None
        else:
            next_node = self.nodes[node.suffix_link]
        while True:
            # when the string on the edge is less than ours, we jump to the next node
            if next_node.children[letter].end != None and next_node.children[letter].start+depth >= next_node.children[letter].end:
                next_node = next_node.children[letter]
                depth -= next_node.end - next_node.start
                letter = self.text[next_node.children[letter].start+depth]
            else:
                return (next_node.idx, letter, depth)
        
    def print_tree(self,i):
        out ='digraph phase%i {\n'%i
        for node in self.nodes:
            if len(node.children) == 0:
                out+= ' %i [shape=box]\n'%node.idx
            else:
                for letter in node.children:
                    next_node = node.children[letter]
                    out+= ' %i -> %i [label ="%s"]\n'%(node.idx, next_node.idx,self.text[next_node.start:next_node.end])
                if node.suffix_link != None:
                    out+= ' %i -> %i [style = dotted]\n'%(node.idx, node.suffix_link)
        out+= '}\n'
        return out
            

class Node:
    def __init__(self, idx, start, end, parent = None, suffix_link = None):
        self.idx = idx
        self.start = start
        self.end = end
        self.parent = parent
        self.children = dict()
        self.suffix_link = suffix_link





# code from class
tree = SuffixTree() # an empty tree
active_pos = (0 , '' , 0) # ( node , letter , depth )
tree_dots = ''
for i , c in enumerate(s):
    tree.text += c
    visited_nodes = []
    while True:
        new_pos = tree.move_down(active_pos, c)
        if new_pos is None:
            link_target = tree.use_suffix_link(active_pos)
            parent_id = tree.new_node(active_pos, i, None)
            visited_nodes.append(parent_id)
            if link_target is None:
                break # we have reached the root
            active_pos = link_target
        else:
            visited_nodes.append(active_pos[0])
            active_pos = new_pos
            break
    for j in range(len(visited_nodes)-1):
        tree.add_suffix_link(visited_nodes[j] ,visited_nodes[j+1])
    out = tree.print_tree(i)
    print(out)
    tree_dots +=out

f = open('output.dot', 'w')
f.write(tree_dots)
f.close()


