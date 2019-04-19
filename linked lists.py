class Node(object):
 
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
 
 
class LinkedList(object):
    def __init__(self, head=None, tail=None):
        self.head = None
        self.tail = None
 
    def print_list(self):
        print("List Values:")
        # Start at the head.
        current_node = self.head
        # Iterate as long as current node is not None.
        while current_node != None:
            # Print the data of the node.
            print(current_node.data)
            # Move to next element.
            current_node = current_node.next
        print(None)
 
    def append(self, data):
        node = Node(data, None)
        # Handle the empty case.
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            # Otherwise set a new next for the tail and set a new tail.
            self.tail.next = node
        self.tail = node
        
    def remove(self, node_value):
        # We're going to want to track a current and previous node.
        current_node = self.head
        previous_node = None
        # Iterate through the list to find the value to remove.
        while current_node != None:
            if current_node.data == node_value:
                if previous_node is not None:
                    previous_node.next = current_node.next
                else:
                    # Handle edge case.
                    self.head = current_node.next
 
            # Keep track of previous nodes to repair list after removal.
            previous_node = current_node
            current_node = current_node.next
    
    # Note that insert is a permutation of remove. 
    def insert(self, value, at):
        current_node = self.head
        new_node = Node(value)
        # Iterate to find our value to insert after.
        while current_node != None:
            if current_node.data == at:
                if current_node is not None:
                    new_node.next = current_node.next
                    current_node.next = new_node 
                else:
                    # Handle edge case.
                    self.tail = current_node.next
 
            # Move to the next one.
            current_node = current_node.next
 # Run these tests, then try play with the LinkedList class and try your own.
s = LinkedList()
s.append(1)
s.append(2)
s.append(3)
s.append(4)
s.print_list()

s.remove(3)
s.remove(2)
s.insert(100, at=1) 

s.print_list()
