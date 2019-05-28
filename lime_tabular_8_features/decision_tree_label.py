class Node:
    
    def __init__(self, index, value, left = None, right = None, label = None):
            
        self.index = index
        self.value = value
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(object):

    def __init__(self, decision_rules):
        
        # the decision_rules are in pre-order, since the binary decision tree is complete
        self.curr_index = 0
        self.root = self.construct_tree(decision_rules)

    def construct_tree(self, rules):

        if rules[self.curr_index] == None:
            return None
        else:
            
            print(self.curr_index, rules[self.curr_index])
            if isinstance(rules[self.curr_index], tuple):
                index = rules[self.curr_index][0]
                value = rules[self.curr_index][1]
            else:
                label = rules[self.curr_index]
            
            self.curr_index += 1
            left_subtree = self.construct_tree(rules)
            self.curr_index += 1
            right_subtree = self.construct_tree(rules)

            if left_subtree == None and right_subtree == None:
                curr_node = Node(None, None, left_subtree, right_subtree, label)
            else:
                curr_node = Node(index, value, left_subtree, right_subtree, None)
            
            return curr_node
    
    def get_label(self, data):
        
        return self.get_label_helper(data, self.root)

    def get_label_helper(self, data, curr_node):

        if curr_node.label != None:
            return curr_node.label
        else:
            if data[curr_node.index] < curr_node.value:
                return self.get_label_helper(data, curr_node.left)
            else:
                return self.get_label_helper(data, curr_node.right)

if __name__ == "__main__":

    dt = DecisionTree([(0, 0.5), (1, 0.5), 0, None, None, (0, 0), 1, None, None, 2, None, None, (2, -0.5), (0, 0.8), 3, None, None, 4, None, None, 5, None, None])

    print(dt.get_label([0.2, 0.2, -0.5, 0.9, 0.1]))
    print(dt.get_label([0.8, -0.3, 0.9]))
    print(dt.get_label([0.3, 0.8, -0.3]))
    print(dt.get_label([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]))