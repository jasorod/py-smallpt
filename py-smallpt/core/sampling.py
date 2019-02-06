from math import sqrt, cos, sin
from math_tools import M_PI
from vector import Vector3
from copy import copy

def uniform_sample_on_hemisphere(u1, u2):
    sin_theta = sqrt(max(0.0, 1.0 - u1 * u1))
    phi = 2.0 * M_PI * u2
    return Vector3(cos(phi) * sin_theta, sin(phi) * sin_theta, u1)
	
def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = sqrt(1.0 - u1)
    sin_theta = sqrt(u1)
    phi = 2.0 * M_PI * u2
    return Vector3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta)

def unit_hemisphere_vector(normal, phi, theta):
    #find a new unit vector that is offset from the current vector by the spherical coordinates phi and theta
    direction = Vector3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi))

    #create a unit basis for the light direction vector
    w = copy(normal).normalize()
    u = Vector3(1, 0, -1 * (normal.x() / (normal.z() + 1e-7))).normalize()
    v = w.cross(u).normalize()

    return direction.x() * u + direction.y() * v + direction.z() * w