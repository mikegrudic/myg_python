from GridDeposit import *

def CoordTransform(coords, plane):
    if plane != 'z':
        x, y, z = coords.T
        return {"x": np.c_[y,z,x], "y": np.c_[x,z,y]}[plane]
    else:
        return coords

def Density(snapdata, ptype, rmax, gridres, z=0.0, plane='z'):
    d = snapdata.field_data[ptype]
    x = CoordTransform(d["Coordinates"], plane)
    h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
    return GridDensity(d["Masses"], x, h, z, gridres, rmax)

def SurfaceDensity(snapdata, ptype, rmax, gridres, plane='z'):
    d = snapdata.field_data[ptype]
    x = CoordTransform(d["Coordinates"], plane)
    h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
    return GridSurfaceDensity(d["Masses"], x, h, gridres, rmax)

def TotalSurfaceDensity(snapdata, ptype, rmax, gridres, plane='z'):
    result = np.zeros((gridres,gridres))
    for d in snapdata.field_data:
        if d==1: continue
        if not "Masses" in d.keys(): continue
#d = snapdata.field_data[ptype]
        x = CoordTransform(d["Coordinates"], plane)
        h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
        result += GridSurfaceDensity(d["Masses"], x, h, gridres, rmax)
    return result
