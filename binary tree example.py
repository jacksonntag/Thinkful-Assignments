# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:43:36 2019

@author: Jack
"""

class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.value = val

########### CREATE TREE ###########
class Tree:
    def __init__(self):
        self.root = None
    # END _init_

########### DELETE TREE ###########
    def DeleteTree(self):
        self.root = None
    # END DeleteTree.

########### ISEMPTY TREE ###########    
    def IsEmpty(self):
        return self.root == None
    # END IsEmpty.

########### DISPLAY TREE ###########
    def PrintTree(self):
        if(self.root != None):
            self._PrintTree(self.root)
    # END PrintTree.

    def _PrintTree(self, node):
        if(node != None):
            self._PrintTree(node.left)
            print(str(node.value) + ' ')
            self._PrintTree(node.right)
    # END _PrintTree.

########### FIND NODE ###########
    def find(self, val):
        if(self.root != None):
        # THEN
            return self.findx(val, self.root)
        else:
            return None
        # ENDIF;
    # END find.

    def findx(self, val, node):
        if(val == node.value):
        # THEN
            print(node.value, "is found")
            return node.value
        elif(val < node.value):
        # THEN
            if (node.left != None):
                return self.findx(val, node.left)
            else:
                print(val, "is not in tree")
        elif(val > node.value):
        # THEN
            if (node.right != None):
                return self.findx(val, node.right)
            else:
                print(val, "is not in tree")
        # ENDIF;
    # END _find.

########### INSERT NODE ###########
    def add(self, val):
        if(self.root == None):
        # THEN
            self.root = Node(val)
        else:
            self._add(val, self.root)
        # ENDIF;
    # END add.

    def _add(self, val, node):
        if(val < node.value):
        # THEN
            if(node.left != None):
            # THEN
                self._add(val, node.left)
            else:
                node.left = Node(val)
            # ENDIF;
        else:
            if(node.right != None):
            # THEN
                self._add(val, node.right)
            else:
                node.right = Node(val)
            # ENDIF;
        # ENDIF;
    # END _add;

########### GET ROOT ###########
    def getRoot(self):
        return self.root
    # END getRoot.

########### MAIN PROGRAM ###########

#     3
# 0     4
#   2      8
tree = Tree()
i=0
while i < 16:
    i+=1
    tree.add(i)

print("@@@@@@@@@@@@")
tree.PrintTree()
print("@@@@@@@@@@@@")
print((tree.find(3)))
print(tree.find(0))
print(tree.find(16))
print(tree.find(-1))
tree.DeleteTree()
tree.PrintTree()