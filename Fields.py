from GridDeposit import *

def ComputeTemperature(d):
    gamma = 5.0/3.0
    y_He = 0.24  / (4*(1-0.24))
    if 'ElectronAbundance' in d.keys():
        a_e = d['ElectronAbundance']
    else:
        a_e = 0.0
    mu = (1 + 4*y_He) / (1 + y_He + a_e)
#    mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
    d["Temperature"] = d["InternalEnergy"]*mu*(gamma-1)*121.128
#    print d["Temperature"].mean()

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

def Temperature(snapdata, ptype, rmax, gridres, z=0.0, plane='z'):
    d = snapdata.field_data[ptype]
    x = CoordTransform(d["Coordinates"], plane)
    h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
    #compute the temperature
    if not "Temperature" in d.keys():
        ComputeTemperature(d)
    rho = GridDensity(d["Masses"], x, h, z, gridres, rmax)
    rho[rho==0] = rho[rho>0].min()
    return  GridDensity(d["Masses"]*d["Temperature"], x, h, z, gridres, rmax) / rho
    
def AverageTemperature(snapdata, ptype, rmax, gridres, plane='z'):
    d = snapdata.field_data[ptype]
    x = CoordTransform(d["Coordinates"], plane)
    h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
    if not "Temperature" in d.keys():
        ComputeTemperature(d)
    if not "SurfaceDensity" in snapdata.grid_fields[ptype].keys():
        snapdata.grid_fields[ptype]["SurfaceDensity"] = SurfaceDensity(snapdata, ptype, rmax, gridres, plane)
    
    return GridSurfaceDensity(d["Masses"]*d["Temperature"], x, h, gridres, rmax) / snapdata.grid_fields[ptype]["SurfaceDensity"]

def VelocityDispersion(snapdata, ptype, rmax, gridres, plane='z'):
    d = snapdata.field_data[ptype]
    x = CoordTransform(d["Coordinates"], plane)
    v = CoordTransform(d["Velocities"], plane)
    h = np.clip(d["SmoothingLength"], 2*rmax/(gridres-1), 1e100)
    return GridVariance(d["Masses"], v[:,2], x, h, gridres, rmax)

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
