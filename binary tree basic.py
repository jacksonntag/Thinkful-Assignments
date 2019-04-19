# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:53:57 2019

 try out coding a binary decision tree

@author: Jack
"""

class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.value = val

class Tree:

    def __init__(self):
        self.root = None

    def delTree(self):
        self.root = None
        
    def add(self, val):
         if(self.root == None):
             self.root = Node(val)
         else:
              self._add(val, self.root)
        
    def IsEmpty(self):
        return self.root == None
    
    def PrintTree(self):
        if(self.root != None):
            self._PrintTree(self.root)
            
    def _PrintTree(self,node):
        if(node != None):
            self._PrintTree(node.left)
            print(str(node.value) + ' ')
            self._PrintTree(node.right)
            
    def find(self, val):
        if(self.root != None):
            return self._find(val, self.root)
        else:
            return None
        
    def _find(self, val, node):
        if(val==node.value):
            return node
        elif(val < node.value and node.left != None):
            self._find(val, node.left)
        elif(val > node.value and node.right != None):
            self._find(val, node.right)
            

    
    def _add(self, val, node):
        if(val < node.value):
            if(node.left != None):
                self._add(val, node.left)
            else:
                node.left = Node(val)
        else:
            if(node.right != None):
                self._add(val, node.right)
            else:
                node.right = Node(val)
            
i=0
tree = Tree()
while i < 15:
    i=i+1
    tree.add(i*11)
    
tree.PrintTree()
            
       