"""
Tree node class for constructing the tree.
"""
import random


class Node:
    def __init__(self, action, parent=None, is_root=False):
        self.is_root = is_root
        self.action = action
        self.parent = parent
        self.children = []
        self.vote = 1
        self.executed = False

    def add_child_by_ao(self, action):
        """Create a new child node with the given value and add it to the children list."""
        new_child = Node(action, self)
        if self == new_child:
            return self
        if new_child not in self.children:
            self.children.append(new_child)
            return new_child
        else:
            n = self.get_child_by_node(new_child)
            n.vote += 1
            return n

    def add_child_by_node(self, new_child):
        if self == new_child:
            return self
        if new_child not in self.children:
            self.children.append(new_child)
            return new_child
        else:
            return self.get_child_by_node(new_child)

    def get_child_by_ao(self, action):
        n = Node(action)
        for node in self.children:
            if node == n:
                return node
        return None

    def get_child_by_node(self, n):
        for node in self.children:
            if node == n:
                return node
        return None

    def get_child_by_index(self, idx):
        return self.children[idx]

    def get_child_highest_vote(self):
        temp = None
        for n in self.children:
            if temp is None or n.vote > temp.vote:
                temp = n
        return temp

    def get_parent(self):
        return self.parent

    def print_tree(self):
        if self.is_root:
            print("root")
        else:
            print("this is not root of a tree!")

    def __str__(self):
        res = f"[{self.action}] --> "
        for node in self.children:
            res += f"[{node.action}], "
        if len(self.children) == 0:
            return res + "NoChild"
        return res[0:-2]

    def __eq__(self, other):
        """Override the default equality method. Nodes are considered equal if their values and parents are the same."""
        if isinstance(other, Node):
            return self.action == other.action
        return False

    def __hash__(self):
        """Override the default hash function. Hash based on the node's value and parent's identity (memory location)."""
        return hash(self.action)


class CommandIssuer:
    def __init__(self, root_node):
        self.root_node = root_node
        self.curr_node = root_node
        self.returned_node = None

    def get_next_step(self):
        if not self.curr_node.is_root and len(self.returned_node.children) == 0:
            return None
        while True:
            if self.curr_node is None:
                return None
            unexec_children = [node for node in self.curr_node.children if not node.executed]
            if len(unexec_children) == 0:
                if self.curr_node.is_root:
                    return None
                else:
                    self.curr_node = self.curr_node.parent
                    print("correction")
            else:
                break
        sorted_children = sorted(unexec_children, key=lambda x: x.vote, reverse=True)
        self.returned_node = sorted_children[0]
        return self.returned_node.action

    def execution_result(self, result):
        self.returned_node.executed = True
        if result:
            self.curr_node = self.returned_node

    def get_next_highest_step(self):
        # get highest voted node
        if len(self.curr_node.children) == 0:
            return None
        sorted_children = sorted(self.curr_node.children, key=lambda x: x.vote, reverse=True)
        self.curr_node = sorted_children[0]
        return self.curr_node.action

        # ablation study: randomly choose one
        # if len(self.curr_node.children) == 0:
        #     return None
        # n = len(self.curr_node.children)
        # random_num = random.randint(0, n - 1)
        # self.curr_node = self.curr_node.children[random_num]
        # return self.curr_node.action
