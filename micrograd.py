class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0
        self._prev = set(children)
        self._op = op
        self._backward = lambda : None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad +=  other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def backward(self):
        order = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
            
                for value in v._prev:
                    build(value)
                order.append(v)
            
        build(self)
        self.grad = 1
        for value in reversed(order):
            value._backward()

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
e = d * a
e.backward()
print(a.grad, b.grad, c.grad)