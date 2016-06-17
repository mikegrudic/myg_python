def MakePlot():
    plotname = "%s_%s_PartType%s_r%g_%s.png"%(field, data.num, type, rmax, plane)
    if field in proj_fields:
        if projdata==None: continue
        Z = projdata[field]
    else:
        if slicedata==None: continue
        Z = slicedata[field]
    print Z[np.logical_not(np.isnan(Z))].max()

    zlabel = field_labels[field]

    if verbose: print("Saving %s..."%plotname)

    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.set_aspect('equal')

    zmin, zmax = field_limits[field]
    if imshow:
        if  len(Z)>1:
            if not np.prod(Z<=0):
                Z[Z==0] = Z[Z>0].min()

        if zmin > 0 and not linscale:
            mpl.image.imsave(plotname, np.log10(np.abs(Z)), cmap=colormap, vmin=np.log10(field_limits[field][0]), vmax=np.log10(field_limits[field][1]))
        else:
            mpl.image.imsave(plotname, Z, cmap=colormap, vmin=field_limits[field][0], vmax=field_limits[field][1])

        if not notext:                
            F = Image.open(plotname)
            draw = ImageDraw.Draw(F)
            draw.line(((gridres/16, 7*gridres/8), (gridres*5/16, 7*gridres/8)), fill="#FFFFFF", width=6)
            draw.text((gridres/16, 7*gridres/8 + 5), "%gpc"%(rmax*500), font=font)
            if blackholes:
                for coords in data.field_data[5]["Coordinates"]:

                    if plane != 'z':
                        x, y, z = coords
                        coords = {"x": np.array([y,z,x]), "y": np.array([x,z,y])}[plane]
                    coords = np.int_((coords+rmax)*gridres/(2*rmax))
                    coords[1] = gridres-coords[1]

                    bbox = (coords[0]-gridres/75, coords[1]-gridres/75, coords[0]+gridres/75, coords[1]+gridres/75)
                    draw.ellipse(bbox, fill="#000000")

            F.save(plotname)
            F.close()                    


    else:
        if zmin > 0 and np.log10(np.abs(zmax)/np.abs(zmin)) >= 2 and zmin != 0 and zmax != 0:
            print X.shape, Y.shape, Z.shape
            print Z.max()
            plot = ax.pcolormesh(X, Y, Z, norm=LogNorm(field_limits[field][0],field_limits[field][1]), antialiased=AA, cmap=colormap)
        else:
            plot = ax.pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, antialiased=AA, cmap="RdBu")
        bar = plt.colorbar(plot, fraction=0.046, pad=0.04)
        bar.set_label(zlabel)


        if plot_clusters:
            clump_data = np.loadtxt("clumps_%d.dat"%data.num)
            ax.scatter(clump_data[:,1]-center[0], clump_data[:,2]-center[1], s=3, facecolors='none', linewidth=0.3, color='r', alpha=0.5)
        if show_particles:
            coords = data.field_data[type]["Coordinates"]
            ax.scatter(coords[:,0], coords[:,1], marker='o', s=1e-1,edgecolor='', facecolors='black')
        ax.set_xlim([-rmax,rmax])
        ax.set_ylim([-rmax,rmax])
        ax.set_xlabel("$x$ $(\mathrm{kpc})$")
        ax.set_ylabel("$y$ $(\mathrm{kpc})$")
        plt.title("$t = %g\\mathrm{Myr}$"%(data.time*979))
        plt.savefig(plotname, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
