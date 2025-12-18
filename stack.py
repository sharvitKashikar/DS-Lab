class Stack:
    def __init__(self):
        self.stack = []

    # Push operation
    def push(self, item):
        self.stack.append(item)
        print(f"Pushed {item}")

    # Pop operation
    def pop(self):
        if self.is_empty():
            print("Stack Underflow")
            return None
        return self.stack.pop()

    # Peek operation
    def peek(self):
        if self.is_empty():
            print("Stack is empty")
            return None
        return self.stack[-1]

    # Check if stack is empty
    def is_empty(self):
        return len(self.stack) == 0

    # Display stack
    def display(self):
        print("Stack elements:", self.stack)


# Driver code
s = Stack()
s.push(10)
s.push(20)
s.push(30)

s.display()

print("Top element:", s.peek())
print("Popped element:", s.pop())

s.display()
