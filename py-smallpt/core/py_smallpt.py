from image_io import write_ppm
from math import sqrt
from ray import Ray
from rng import RNG
from sampling import cosine_weighted_sample_on_hemisphere
from sphere import Sphere
from specular import ideal_specular_reflect, ideal_specular_transmit
from vector import Vector3

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5

spheres = [
        Sphere(1e5,  Vector3(1e5 + 1, 40.8, 81.6),   f=Vector3(0.75,0.25,0.25)),
	    Sphere(1e5,  Vector3(-1e5 + 99, 40.8, 81.6), f=Vector3(0.25,0.25,0.75)),
	    Sphere(1e5,  Vector3(50, 40.8, 1e5),         f=Vector3(0.75, 0.75, 0.75)),
	    Sphere(1e5,  Vector3(50, 40.8, -1e5 + 170)),
	    Sphere(1e5,  Vector3(50, 1e5, 81.6),         f=Vector3(0.75, 0.75, 0.75)),
	    Sphere(1e5,  Vector3(50, -1e5 + 81.6, 81.6), f=Vector3(0.75, 0.75, 0.75)),
	    Sphere(16.5, Vector3(27, 16.5, 47),          f=Vector3(0.999, 0.999, 0.999), reflection_t=Sphere.Reflection_t.SPECULAR),
	    Sphere(16.5, Vector3(73, 16.5, 78),          f=Vector3(0.999, 0.999, 0.999), reflection_t=Sphere.Reflection_t.REFRACTIVE),
	    Sphere(600,  Vector3(50, 681.6 - .27, 81.6), e=Vector3(12, 12, 12))
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

def radiance(ray, rng):
    r = ray
    L = Vector3()
    F = Vector3(1.0, 1.0, 1.0)

    while (True):
        hit, id = intersect(r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        n = (p - shape.p).normalize()

        L += F * shape.e
        F *= shape.f
        
	    # Russian roulette
        if r.depth > 4:
            continue_probability = shape.f.max_value()
            if rng.uniform_float() >= continue_probability:
                return L
            F /= continue_probability

        # Next path segment
        if shape.reflection_t == Sphere.Reflection_t.SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        elif shape.reflection_t == Sphere.Reflection_t.REFRACTIVE:
            d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, rng)
            F *= pr
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        else:
            w = n if n.dot(r.d) < 0 else -n
            u = (Vector3(0.0, 1.0, 0.0) if abs(w[0]) > 0.1 else Vector3(1.0, 0.0, 0.0)).cross(w).normalize()
            v = w.cross(u)

            sample_d = cosine_weighted_sample_on_hemisphere(rng.uniform_float(), rng.uniform_float())
            d = (sample_d[0] * u + sample_d[1] * v + sample_d[2] * w).normalize()
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue

import sys
import argparse

parser = argparse.ArgumentParser(description= "Small python based path tracer")
parser.add_argument("--samples", type="int", default=4)
args = parser.parse_args(sys.argv)

if __name__ == "__main__":
    rng = RNG()
    nb_samples = args.samples

    w = 1024
    h = 768

    eye = Vector3(50, 52, 295.6)
    gaze = Vector3(0, -0.042612, -1).normalize()
    fov = 0.5135
    cx = Vector3(w * fov / h, 0.0, 0.0)
    cy = cx.cross(gaze).normalize() * fov

    Ls = [None] * w * h
    for i in range(w * h):
        Ls[i] = Vector3()
    
    for y in range(h):
        # pixel row
        print('\rRendering ({0} spp) {1:0.2f}%'.format(nb_samples * 4, 100.0 * y / (h - 1)))
        for x in range(w):
            # pixel column
            for sy in range(2):
                i = (h - 1 - y) * w + x
                # 2 subpixel row
                for sx in range(2):
                    # 2 subpixel column
                    L = Vector3()
                    for s in range(nb_samples):
                        #  samples per subpixel
                        u1 = 2.0 * rng.uniform_float()
                        u2 = 2.0 * rng.uniform_float()
                        dx = sqrt(u1) - 1.0 if u1 < 1 else 1.0 - sqrt(2.0 - u1)
                        dy = sqrt(u2) - 1.0 if u2 < 1 else 1.0 - sqrt(2.0 - u2)
                        d = cx * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                            cy * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + gaze
                        L += radiance(Ray(eye + d * 130, d.normalize(), tmin=Sphere.EPSILON_SPHERE), rng) * (1.0 / nb_samples)
                    Ls[i] += 0.25 * Vector3.clamp(L)

    write_ppm(w, h, Ls)
