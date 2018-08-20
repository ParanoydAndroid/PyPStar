import heapq
#
# Uses a heap to back a list representing a priority queue of arbitrary objects
#
import itertools


class PriorityQueue:

    # In line with heapq, this PriorityQueue uses a min heap, so ensure priorities are highest at lower numbers
    REMOVED = '<removed task>'

    def __init__(self):
        self.pq = []
        self.pqMap = {}
        self.counter = itertools.count()
        self.count = 0

    def push(self, id, priority=0):
        # because we can push an existing task with a new priority, we need to account for duplication
        if id in self.pqMap:
            self.remove(id)

        self.count = next(self.counter)

        # Do not reorder the list elements.  The tuple comparison moves L -> R and count tiebreaks priority
        node = [priority, self.count, id]
        self.pqMap[id] = node

        heapq.heappush(self.pq, node)

    def pop(self):

        while self.pq:
            priority, count, node_id = heapq.heappop(self.pq)

            if node_id is not self.REMOVED:
                del self.pqMap[node_id]
                return node_id

    def remove(self, id):

        # note we keep the heap property by not actually deleting the node at this point.
        node = self.pqMap.pop(id)

        # I'm not sure why I have to use a self. here, since I'm intentionally accessing a static variable.  ??
        node[1] = self.REMOVED

    def empty(self):
        return len(self.pq) == 0
