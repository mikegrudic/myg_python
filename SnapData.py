import h5py
import numpy as np
from SmoothingLength import SmoothingLength
from PlotSettings import *
import Fields

class SnapData:
    def __init__(self, name, center=np.array([0,0,0]), periodic=False, verbose=False, n_ngb=32):
        f = h5py.File(name, "r")
        header_toparse = f["Header"].attrs

        if periodic:
            box_size = header_toparse["BoxSize"]
        else:
            box_size = None

        self.time = header_toparse["Time"]
        
        particle_counts = header_toparse["NumPart_ThisFile"]
        
        self.field_data = [{}, {}, {}, {}, {}, {}]
        r = {}
        
        for i, n in enumerate(particle_counts):
            if n==0: continue
#            if len(fields_toplot[i]) == 0 : continue
            if i==5: continue

            pname = {0:"Gas", 1:"DM", 2:"Disk", 3:"Bulge", 5:"BH", 4:"Stars"}[i]
            
            ptype = f["PartType%d" % i]
            X = np.array(ptype["Coordinates"]) - center
            if periodic:
                X[X<-0.5*box_size] = X[X<-0.5*box_size] + box_size
                X[X>=0.5*box_size] = X[X>=0.5*box_size] - box_size
            r = np.sqrt(np.sum(X[:,:2]**2, axis=1))
#            filter = np.max(np.abs(X), axis=1) <= 1e100
            
            for key in ptype.keys():
                self.field_data[i][key] = np.array(ptype[key])

            self.field_data[i]["Coordinates"] = X
            if not "Masses" in ptype.keys():
                self.field_data[i]["Masses"] = f["Header"].attrs["MassTable"][i] * np.ones_like(self.field_data[i]["Coordinates"][:,0])
            if not "SmoothingLength" in ptype.keys():
                if "AGS-Softening" in ptype.keys() and i != 4 and False:
                    self.field_data[i]["SmoothingLength"] = np.array(ptype["AGS-Softening"])
                else:
                    if verbose: print("Computing smoothing length for %s..." % pname.lower())
                    self.field_data[i]["SmoothingLength"] = SmoothingLength(self.field_data[i]["Coordinates"], des_ngb=n_ngb, box_size=box_size)
                if verbose: print "done!"
            if not "Density" in ptype.keys():
                self.field_data[i]["Density"] = self.field_data[i]["Masses"]*32 / (4*np.pi * self.field_data[i]["SmoothingLength"]**3 / 3)
        if "PartType5" in f.keys():
            self.field_data[5]["Coordinates"] = np.array(f["PartType5"]["Coordinates"]) - center
        f.close()

        if verbose: print("Reticulating splines...")        

        try:
            self.num = int(name.split("_")[1].split('.')[0])
        except:
            self.num = 0

        self.grid_fields = [{},{},{},{},{},{}]


    def ComputeField(self, ptype, field, rmax, gridres=400, plane='z'):
        function = getattr(Fields, field)
        self.grid_fields[ptype][field] = function(self, ptype, rmax, gridres,plane=plane)
        return self.grid_fields[ptype][field]
