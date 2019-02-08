#!/usr/bin/python

from image_io import write_ppm
from math import *
from copy import copy
from ray import Ray
from rng import RNG
from sampling import cosine_weighted_sample_on_hemisphere, unit_hemisphere_vector
from sphere import Sphere
from specular import ideal_specular_reflect, ideal_specular_transmit
from vector import Vector3

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5

spheres = [
    Sphere("red wall", 1e5,  Vector3(-1e5, 40.8, 81.6),   f=Vector3(0.75,0.25,0.25)),
    Sphere("blue wall", 1e5,  Vector3(1e5 + 99, 40.8, 81.6), f=Vector3(0.25,0.25,0.75)),
    Sphere("gray wall", 1e5,  Vector3(50, 40.8, -1e5),         f=Vector3(0.75, 0.75, 0.75)),
    Sphere("env", 1e5,  Vector3(50, 40.8, 1e5 + 170)),
    Sphere("celing", 1e5,  Vector3(50, 1e5 + 81.6, 81.6),         f=Vector3(0.75, 0.75, 0.75)),
    Sphere("floor", 1e5,  Vector3(50, -1e5, 81.6), f=Vector3(0.75, 0.75, 0.75)),
    Sphere("reflective sphere", 16.5, Vector3(27, 16.5, 47),          f=Vector3(0.999, 0.999, 0.999), reflection_t=Sphere.Reflection_t.SPECULAR),
    Sphere("refractive sphere", 16.5, Vector3(73, 16.5, 78),          f=Vector3(0.999, 0.999, 0.999), reflection_t=Sphere.Reflection_t.SPECULAR),
    Sphere("light", 10,  Vector3(50, 70, 81.6), e=Vector3(2000, 2000, 2000))
]

def intersect(ray):
    id = None
    hit = False
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            hit = True
            id = i
    return hit, id

def intersectP(ray):
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            return True
    return False

def lights():
    light_list = []
    for i in range(len(spheres)):
        if spheres[i].name == "light":
            light_list.append(spheres[i])

    return light_list

def indirect_light_pass(ray, rng):
    r = ray
    L = Vector3()

    while (True):
        hit, id = intersect(r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        n = (p - shape.p).normalize()

        # Next path segment
        if shape.reflection_t == Sphere.Reflection_t.SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = Ray(p, d, tmin=Sphere.EPSILON, depth=r.depth)
            continue
        elif shape.reflection_t == Sphere.Reflection_t.REFRACTIVE:
            d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, rng)
            r = Ray(p, d, tmin=Sphere.EPSILON, depth=r.depth)
            continue
        else:
            w = n if n.dot(r.d) < 0 else -n
            u = (Vector3(0.0, 1.0, 0.0) if abs(w[0]) > 0.1 else Vector3(1.0, 0.0, 0.0)).cross(w).normalize()
            v = w.cross(u)

            sample_d = cosine_weighted_sample_on_hemisphere(rng.uniform_float(), rng.uniform_float())
            d = (sample_d[0] * u + sample_d[1] * v + sample_d[2] * w).normalize()
            r = Ray(p, d, tmin=Sphere.EPSILON, depth=r.depth + 1)
            break

    return shadow_ray_pass(r, rng)


def shadow_ray_pass(ray, rng):
    r = ray
    L = Vector3()
    F = Vector3(1.0, 1.0, 1.0)

    while True:
        hit, id = intersect(r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        n = (p - shape.p).normalize()

        F *= shape.f

        if shape.name == "light":
            return shape.e

        # Bounce another ray if the surface is reflective or refractive
        if shape.reflection_t == Sphere.Reflection_t.SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = Ray(p, d, tmin=Sphere.EPSILON, depth=r.depth + 1)
            continue
        elif shape.reflection_t == Sphere.Reflection_t.REFRACTIVE:
            d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, rng)
            F *= pr
            r = Ray(p, d, tmin=Sphere.EPSILON, depth=r.depth + 1)
            continue

        light_list = lights()
        for light in light_list:
            #get the spherical angle of the light projected from the point on the surface
            light_vec = light.p - p
            cos_projected_angle = sqrt(1 - (pow(light.r, 2) / light_vec.norm2_squared()))
            projected_angle = acos(cos_projected_angle)
            shadow_ray_dir = unit_hemisphere_vector(light_vec, rng.uniform_float() * projected_angle, rng.uniform_range_float(0, 2 * pi))
            shadow_ray_vec = Ray(p, shadow_ray_dir, tmin=Sphere.EPSILON, depth = r.depth + 1)

            hit, id = intersect(shadow_ray_vec)
            if spheres[id].name != "light":
                continue
            else:
                cos_incident_angle = shadow_ray_vec.d.dot(n)
                L += F * ((spheres[id].e / pow(shadow_ray_vec.tmax, 2)) * cos_incident_angle)

        return L 

def albedo_pass(ray, rng):
    r = ray
    L = Vector3()
    F = Vector3(1.0, 1.0, 1.0)

    hit, id = intersect(r)
    if (not hit):
        return L

    shape = spheres[id]
    L = shape.f
    return L

def normal_pass(ray, rng):
    r = ray

    hit, id = intersect(r)
    if (not hit):
        return Vector3()

    shape = spheres[id]
    p = r(r.tmax)
    n = (p - shape.p).normalize()
    n += 1
    n *= 0.5
    return n

import sys
import argparse

parser = argparse.ArgumentParser(description= "Small python based path tracer")
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--passtype", type=str, choices=['albedo', 'normal', 'indirect', 'shadow'], default="shadow")
args = parser.parse_args()

if __name__ == "__main__":
    rng = RNG()
    nb_samples = args.samples

    w = 1024
    h = 768

    #setup orthographic camera
    eye = Vector3(50, 52, 295.6)
    gaze = Vector3(0, -0.042612, -1).normalize()
    fov = 0.5135
    cam_x = Vector3(w * fov / h, 0.0, 0.0)
    cam_y = cam_x.cross(gaze).normalize() * fov

    Ls = [None] * w * h
    for i in range(w * h):
        Ls[i] = Vector3()
    
    for y in range(h):
        # pixel row
        print('\rRendering ({0} spp) {1:0.2f}%'.format(nb_samples, 100.0 * y / (h - 1)))
        for x in range(w):
            # pixel column
            i = (h - 1 - y) * w + x
            for s in range (nb_samples):
                # samples per pixel
                dx = rng.uniform_float()
                dy = rng.uniform_float()

                #create camera vector multipliers that range between screen-space coordinates of -0.5 to 0.5
                cam_x_multiplier = (dx + x) / w - 0.5
                cam_y_multiplier = (dy + y) / h - 0.5

                directional_vec = cam_x * cam_x_multiplier + cam_y * cam_y_multiplier + gaze
                L = Vector3()

                if args.passtype == "shadow":
                    L = shadow_ray_pass(Ray(eye + directional_vec * 130, directional_vec.normalize(), tmin=Sphere.EPSILON), rng)
                elif args.passtype == "albedo":
                    L = albedo_pass(Ray(eye + directional_vec * 130, directional_vec.normalize(), tmin=Sphere.EPSILON), rng)
                elif args.passtype == "normal":
                    L = normal_pass(Ray(eye + directional_vec * 130, directional_vec.normalize(), tmin=Sphere.EPSILON), rng)
                elif args.passtype == "indirect":
                    L = indirect_light_pass(Ray(eye + directional_vec * 130, directional_vec.normalize(), tmin=Sphere.EPSILON), rng)

                Ls[i] +=  (1.0 / nb_samples) * Vector3.clamp(L)

    write_ppm(w, h, Ls, fname = args.passtype + "_pass_" + str(nb_samples) + "spp.ppm")
