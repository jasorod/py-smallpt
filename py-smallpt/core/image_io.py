from math_tools import to_byte

def write_ppm(w, h, Ls, fname = "py-image.ppm"):
    with open(fname, 'w') as outfile:
        outfile.write('P3\n{0} {1}\n{2}\n'.format(w, h, 255))
        for L in Ls:
            outfile.write('{0} {1} {2} '.format(to_byte(L[0]), to_byte(L[1]), to_byte(L[2])))