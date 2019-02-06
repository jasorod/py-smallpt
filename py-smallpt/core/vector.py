from math import isnan, sqrt, pow, floor, ceil, trunc, pi, sin, cos
from math_tools import clamp
from copy import copy

class Vector3(object):

    def __init__(self, x = 0.0, y = 0.0, z = 0.0):
        self.raw = [float(x), float(y), float(z)]

    def x(self):
        return self.raw[0]

    def y(self):
        return self.raw[1]

    def z(self):
        return self.raw[2]

    def has_NaNs(self):
        return isnan(self.raw[0]) or isnan(self.raw[1]) or isnan(self.raw[2])

    def __copy__(self):
        return self.__deepcopy__();

    def __deepcopy__(self):
        return Vector3(self.raw[0], self.raw[1], self.raw[2])

    def __getitem__(self, i):
        return self.raw[i]

    def __neg__(self):
        return Vector3(-self.raw[0], -self.raw[1], -self.raw[2]) 

    def __add__(self, x):
        if isinstance(x, Vector3):
            return Vector3(self.raw[0] + x[0], self.raw[1] + x[1], self.raw[2] + x[2])
        else:
            return Vector3(self.raw[0] + x, self.raw[1] + x, self.raw[2] + x)

    def __radd__(self, x):
        return Vector3(x + self.raw[0], x + self.raw[1], x + self.raw[2])
    
    def __sub__(self, x):
        if isinstance(x, Vector3):
            return Vector3(self.raw[0] - x[0], self.raw[1] - x[1], self.raw[2] - x[2])
        else:
            return Vector3(self.raw[0] - x, self.raw[1] - x, self.raw[2] - x)

    def __rsub__(self, x):
        return Vector3(x - self.raw[0], x - self.raw[1], x - self.raw[2])

    def __mul__(self, x):
        if isinstance(x, Vector3):
            return Vector3(self.raw[0] * x[0], self.raw[1] * x[1], self.raw[2] * x[2])
        else:
            return Vector3(self.raw[0] * x, self.raw[1] * x, self.raw[2] * x)

    def __rmul__(self, x):
        return Vector3(x * self.raw[0], x * self.raw[1], x * self.raw[2])
        
    def __div__(self, x):
        if isinstance(x, Vector3):
            return Vector3(self.raw[0] / x[0], self.raw[1] / x[1], self.raw[2] / x[2])
        else:
            inv_x = 1.0 / x
            return Vector3(self.raw[0] * inv_x, self.raw[1] * inv_x, self.raw[2] * inv_x)

    def __rdiv__(self, x):
        return Vector3(x / self.raw[0], x / self.raw[1], x / self.raw[2])

    def __truediv__(self, x):
        if isinstance(x, Vector3):
            return Vector3(self.raw[0] / x[0], self.raw[1] / x[1], self.raw[2] / x[2])
        else:
            inv_x = 1.0 / x
            return Vector3(self.raw[0] * inv_x, self.raw[1] * inv_x, self.raw[2] * inv_x)

    def __rtruediv__(self, x):
        return Vector3(x / self.raw[0], x / self.raw[1], x / self.raw[2])

    def dot(self, v):
        return self.raw[0] * v[0] + self.raw[1] * v[1] + self.raw[2] * v[2]

    def cross(self, v):
        return Vector3(self.raw[1] * v[2] - self.raw[2] * v[1], self.raw[2] * v[0] - self.raw[0] * v[2], self.raw[0] * v[1] - self.raw[1] * v[0]);

    def __eq__(self, v):
        return self.raw[0] == v[0] and self.raw[1] == v[1] and self.raw[2] == v[2]
    
    def __ne__(self, v):
        return self.raw[0] != v[0] or self.raw[1] != v[1] or self.raw[2] != v[2]
    
    def __lt__(self, v):
        return self.raw[0] < v[0] and self.raw[1] < v[1] and self.raw[2] < v[2]
    
    def __le__(self, v):
        return self.raw[0] <= v[0] and self.raw[1] <= v[1] and self.raw[2] <= v[2]
    
    def __gt__(self, v):
        return self.raw[0] > v[0] and self.raw[1] > v[1] and self.raw[2] > v[2]
   
    def __ge__(self, v):
        return self.raw[0] >= v[0] and self.raw[1] >= v[1] and self.raw[2] >= v[2]

    def min_dimension(self):
        if self.raw[0] < self.raw[1] and self.raw[0] < self.raw[2]:
            return 0
        if self.raw[1] < self.raw[2]:
            return 1
        return 2

    def max_dimension(self):
        if self.raw[0] > self.raw[1] and self.raw[0] > self.raw[2]:
            return 0
        if self.raw[1] > self.raw[2]:
            return 1
        return 2

    def min_value(self):
        if self.raw[0] < self.raw[1] and self.raw[0] < self.raw[2]:
            return self.raw[0]
        if self.raw[1] < self.raw[2]:
            return self.raw[1]
        return self.raw[2]

    def max_value(self):
        if self.raw[0] > self.raw[1] and self.raw[0] > self.raw[2]:
            return self.raw[0]
        if self.raw[1] > self.raw[2]:
            return self.raw[1]
        return self.raw[2]

    def norm2_squared(self):
        return self.raw[0] * self.raw[0] + self.raw[1] * self.raw[1] + self.raw[2] * self.raw[2]

    def norm2(self):
        return sqrt(self.norm2_squared())

    def normalize(self):
        a = 1.0 / self.norm2()
        self.raw[0] *= a
        self.raw[1] *= a
        self.raw[2] *= a
        return self

    def __str__(self):
        return '[' + str(self.raw[0]) + ' ' + str(self.raw[1]) + ' ' + str(self.raw[2]) + ']'

    @classmethod
    def apply_unary(cls, f, v, *args, **kwargs):
        return Vector3(f(v.raw[0], *args, **kwargs), f(v.raw[1], *args, **kwargs), f(v.raw[2], *args, **kwargs))

    @classmethod
    def apply_binary(cls, f, v1, v2, *args, **kwargs):
        return Vector3(f(v1.raw[0], v2.raw[0], *args, **kwargs), f(v1.raw[1], v2.raw[1], *args, **kwargs), f(v1.raw[2], v2.raw[2], *args, **kwargs))

    @classmethod
    def sqrt(cls, v):
        return cls.apply_unary(sqrt, v)

    @classmethod
    def pow(cls, v, a):
        return cls.apply_unary(pow, v, a)

    @classmethod
    def abs(cls, v):
        return cls.apply_unary(abs, v)

    @classmethod
    def min(cls, v1, v2):
        return cls.apply_binary(min, v1, v2)

    @classmethod
    def max(cls, v1, v2):
        return cls.apply_binary(max, v1, v2)

    @classmethod
    def round(cls, v):
        return cls.apply_unary(round, v)

    @classmethod
    def floor(cls, v):
        return cls.apply_unary(floor, v)

    @classmethod
    def ceil(cls, v):
        return cls.apply_unary(ceil, v)

    @classmethod
    def trunc(cls, v):
        return cls.apply_unary(trunc, v)

    @classmethod
    def clamp(cls, v, low = 0.0, high = 1.0):
        return cls.apply_unary(clamp, v, low=low, high=high)

    @classmethod
    def lerp(cls, a, v1, v2):
        return v1 + a * (v2 - v1)

    @classmethod
    def permute(cls, v):
        return cls(v.raw[1], v.raw[2], v.raw[0])
    
