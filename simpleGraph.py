from __future__ import print_function
import numpy as np
from abc import ABCMeta

class Node(object):
	__metaclass__ = ABCMeta
	def forward_prop(self):
		pass

class Matmul(Node):
	def __init__(self, parent, child):
		self.parent = parent
		self.child = child
		self.op = 'matmul'
	def forward_prop(self, a, b):
		output = np.dot(a, b)
		return output
	
class Add(Node):
	def __init__(self, parent = None, child = None):
		self.parent = parent
		self.child = child
		self.op = 'add'
	def forward_prop(self, a, b):
		output = np.sum(a, b)
		return output


def lastNode(start):
	node = start
	prev = node
	while node != None:
		prev = node
		node = node.child
	return prev

def newNode(graph, op):
	last = lastNode(graph)
	if op == 'matmul':
		if last != None:
			last.child = Matmul(last,None)
		else:
			return  Matmul(None,None)
	elif op == 'add':
		if last != None:
			last.child = Add(last,None)
		else:
			return Add(None, None)

def printGraph(root):
	node = root
	while node != None:
		if node.child != None:
			print(node.op+" -> ", end='')
		else:
			print(node.op)
		node = node.child


def main():
	graph = newNode(None, 'matmul') # initialize
	newNode(graph,'matmul')
	newNode(graph,'add')
	newNode(graph,'add')
	newNode(graph,'matmul')

	graph2 = newNode(None, 'matmul') # initialize
	newNode(graph2,'matmul')
	newNode(graph2,'add')
	newNode(graph2,'add')
	newNode(graph2,'matmul')
	newNode(graph2,'add')
	newNode(graph2,'matmul')
	
	printGraph(graph)
	printGraph(graph2)

if __name__ == '__main__':
	main()
