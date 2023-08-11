import math

class Var:
    def __init__(self, v):
        self.v = v
        self.d = lambda var : 1.0 if (var is self) else 0.0
    
class plus:
    def __init__(self, a, b):
        self.v = a.v + b.v
        # y' = a' + b'
        self.d = lambda var : a.d(var) + b.d(var)

class mul:
    def __init__(self, a, b):
        self.v = a.v * b.v
        # y' = ab' + ba'
        self.d = lambda var : a.v * b.d(var) + a.d(var) * b.v

class pow:
    def __init__(self, a, b):
        self.v = a.v**b.v
        # y' = a^b * (ln(a)*b)'
        self.d = lambda var : (a.v**b.v) * mul(ln(a), b).d(var)

class ln:
    def __init__(self, a):
        self.v = math.log(a.v)
        # y' = (1/a) * a'
        self.d = lambda var : pow(a, Var(-1)).v * a.d(var)

class sin:
    def __init__(self, a):
        self.v = math.sin(a.v)
        # y' = cos(a)*a'
        self.d = lambda var : math.cos(a.v) * a.d(var)

class cos:
    def __init__(self, a):
        self.v = math.cos(a.v)
        # y' = cos(b)*b'
        self.d = lambda var : -math.sin(a.v) * a.d(var)
