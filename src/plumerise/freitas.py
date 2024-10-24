"""
A Python interface to CPTEC's Plume Rise Model.

"""

from numpy      import zeros, ones, meshgrid, linspace, any, \
                       pi, sin, cos, arccos, arange, array, \
                       savez, NaN, isnan

from scipy import optimize as opt

from datetime   import date, timedelta
from glob       import glob

from dozier     import DOZIER
from VegType    import IGBP
from PlumeRise_ import *       # f2py extension
from gfio       import GFIO

from pyobs           import NPZ
from pyobs.binObs_   import binareas
from pyobs.minx      import MINXs
from MAPL.constants  import *

import eta

__VERSION__ = 2.1
__CVSTAG__  = '@CVSTAG'
__AMISS__   = 1.E+20

Bioma = [ 'Tropical Forest',
          'Extra-Tropical Forest',
          'Savanna',
          'Grassland' ]

DAY = timedelta(seconds=60*60*24)

#----------------------------------------------------------------------------------------------

#----
class MINXs_PR(MINXs):

    """
    Extension of the MINXs class adding the Freitas Plume Rise
    functionality. This class handles non-gridded, observation
    location fires.
    """
    
    def getPlume1(self,i,
                  hflux_kW=None,
                  frp_MW=None,ffac=None,
                  area=None,afac=None,
                  Nominal=False,Verbose=False):
 
        """
        Runs the Plume Rise extension to compute the extent of the plume.
        On input,
           i         ---  index to operate on
           frp_MW    ---  fire radiative power in MW
           area      ---  firea are im m2
           Nominal   ---  by default, areas and heat fluxes are from a
                          Dozier type algorithm, If Nominal=True,
                             area = 20e4 (20 ha)
                             hflux = FRP / pixel_area
           
	"""

        ptop = 1. # Pascal

        # Default areas, heat flux
        # ------------------------
        if Nominal:
            d_area = 20e4  #  typical fire size in m2 (20 ha)
        else: 
            d_area = 1e6 * self.mod14.farea[i]      # km2 --> m2

        # Area
        # ----
        if area == None:
            area = d_area # m2

        # FRP
        # ---
        if frp_MW == None:
            frp_MW = self.mod14.frp[i] # MW

        # Heat Flux
        # ---------
        if hflux_kW == None:
            hflux_kW = 1e3 * frp_MW / area     # kW/m2

        # Scaled FRP
        # ----------
        if ffac != None:
            hflux_kW = ffac * hflux_kW # kW/m2
            
        # Scaled Area
        # -----------
        if afac!=None:
            if afac==0:
                afac = 1e-8
            area = afac * area
            hflux_kW = hflux_kW / afac
        
        u = self.sample.u[i]
        v = self.sample.v[i]
        T = self.sample.t[i]
        q = self.sample.qv[i]
        delp = self.sample.delp[i]

        if delp.min()<=0 or T.min()<=0:
            return NaN

        # Ensure arrays are top-down as in GEOS-5
        # ---------------------------------------
        if delp[0] > delp[-1]:
            u = u[::-1]
            v = v[::-1]
            T = T[::-1]
            q = q[::-1]
            delp = delp[::-1]

        # Run plume rise model
        # --------------------
        p,z,k,rc = plume(u,v,T,q,delp,ptop,hflux_kW,area)
        if rc:
            raise ValueError, "error on return from <plume>, rc = %d"%rc

        if z==-1.:
            z = NaN  # self.sample.pblh[i]
            
        if Verbose:
            print " ", self.tyme[i], "| %8.2f | %8.2f %8.2f | %8.2f"%\
                (self.mod14.fdist[i], area/1e4, hflux_kW, z)

        return z

    def getPlume(self,I=None,Verbose=True,**kwopts):
 
        """
        Runs the Plume Rise extension to compute the extent of the plume.
	"""

        z_plume = __AMISS__ * ones(self.N)

        R = arange(self.N)
        if I is not None:
            R = R[I]

        if Verbose:
            print ""
            print "                    Plume Height Estimates"
            print ""
            print "  --------------------|----------|-------------------|----------"
            print "                      | Distance |   Fire Properties |  Plume"
            print "    MINX Date/Time    |  to Fire |   Area   Heat Flx |  Height"
            print "                      |    km    |    ha      kW     |    km"
            print "  --------------------|----------|-------------------|----------"
            

#       Loop over time
#       --------------
        for i in R:
            z_plume[i] = self.getPlume1(i,Verbose=Verbose,**kwopts)
        if Verbose:
            print "  --------------------|----------|-------------------|----------"

        return array(z_plume)

    def getOptBrute(self,I=None,Verbose=True,**kwopts):
 
        """
        Runs the Plume Rise extension to compute brute force optimal value of (hflux,area)
        to match MISR plume height.
	"""

        z_opt = __AMISS__ * ones(self.N)
        h_opt = __AMISS__ * ones(self.N)
        a_opt = __AMISS__ * ones(self.N)
        
        R = arange(self.N)
        if I is not None:
            R = R[I]

        Hflux_kW = linspace(1,100,10)
        Area = linspace(0.1e4,2e4,10)

        print Hflux_kW
        print Area

        if Verbose:
            print ""
            print "                    Plume Height Estimates"
            print ""
            print "  --------------------|----------|-------------------|----------"
            print "                      | Observed |    Opt Properties |  Optimal "
            print "    MINX Date/Time    |  Height  |   Area   Heat Flx |  Height"
            print "                      |    km    |    ha      kW     |    km"
            print "  --------------------|----------|-------------------|----------"
            

#       Loop over time
#       --------------
        for i in R:
            z = m.z[i]
            e = 1e20 # overestimate
            for hflux_kW in Hflux_kW:
                for area in Area:
                    z_ = self.getPlume1(i,Verbose=False,hflux_kW=hflux_kW,area=area)
                    if isnan(z)==False:
                        e_ = (z-z_)**2
                        if e_<e:
                            e = e_
                            z_opt[i] = z_
                            h_opt[i] = hflux_kW
                            a_opt[i] = area
            if e<1e20:
                if Verbose:
                    print " ", self.tyme[i], "| %8.2f | %8.2f %8.2f | %8.2f"%\
                          (z, a_opt[i]/1e4, h_opt[i], z_opt[i])
                                        
        if Verbose:
            print "  --------------------|----------|-------------------|----------"

        return (z_opt,h_opt,a_opt)


#----
###    def getFires(self,mod14_path='/nobackup/MODIS/Level2/MOD14', Verbose=True):
    def getFires(self,mod14_path='/Users/adasilva/workspace/MOD14',npzFile=None,Verbose=True):
        """
        Retrieves Level2 MOD14 fire data for each MINX fire.
        """
        from dozier import DOZIER

        self.mod14 = []
        d2r = pi / 180.
        a = MAPL_RADIUS/1000. # Earth radius in km
        dt = timedelta(seconds=5*60)

        self.mod14 = MOD14(self.N)
        
        if Verbose:
            print ""
            print "                   Fire Heat Flux Estimates"
            print ""
            print "  --------------------|----------|-------------------|----------"
            print "                      | Distance |    FRP Estimates  |   Fire"
            print "    MINX Date/Time    |  to Fire |   MINX      MODIS | Heat Flux"
            print "                      |    km    |    MW         MW  |  kW/m2"
            print "  --------------------|----------|-------------------|----------"

        for i in range(self.N):

            # Get fire granule for this particular day
            # ----------------------------------------
            t = self.tyme[i]
            p = mod14_path
            m = DOZIER(_getGran(t-dt,p) + _getGran(t,p) + _getGran(t+dt,p))

            # Select closest fires
            # --------------------
            x0 = cos(d2r*self.lat_f[i]) * cos(d2r*self.lon_f[i])
            y0 = cos(d2r*self.lat_f[i]) * sin(d2r*self.lon_f[i])
            z0 = sin(d2r*self.lat_f[i])
            dx = x0*cos(d2r*m.lat) * cos(d2r*m.lon)
            dy = y0*cos(d2r*m.lat) * sin(d2r*m.lon)
            dz = z0*sin(d2r*m.lat) 
            s  = a * arccos(dx+dy+dz) # great circle distance
            j = s.argmin()

            # Estimate fire heat flux
            # -----------------------
            m.classic_var() # classic Dozier
            self.mod14.hflux[i] = m.hflux[j]
            self.mod14.fdist[i] = s[j]
            self.mod14.frp[i]  = m.pow[j]
            self.mod14.pixar[i] = m.pixar[j]
            self.mod14.farea[i] = m.farea[j]
            self.mod14.qa[i] = m.m[j] # bolean
            if Verbose:
                print " ", t, "| %8.2f | %8.2f %8.2f | %8.2f"%\
                    (self.mod14.fdist[i], self.mod14.frp[i], self.mod14.frp[i], \
                     self.mod14.hflux[i] )

        # Save fire properties in NPZ file for later
        # ------------------------------------------
        if npzFile!=None:
            savez(npzFile,**self.mod14.__dict__)
                  
        if Verbose:
            print "  --------------------|----------|-------------------|----------"

#---
    def getOptF(self,Verbose=True):
        """
        Find optimal fire modified bstar to match observed plume height.
        """

        if Verbose:
            print ""
            print "    Fire Modified bstar Optimization"
            print ""
            print "Plume | f_opt  |    J     |  Ni | Nf"
            print "------|--------|----------|-----|-----"

        self.f_opt = ones(self.N)
        self.z_opt = ones(self.N)
        for i in range(self.N):
#            xmin, fval, iter, fcalls = opt.brent(CostFuncF,args=(self,i),brack=(1.,4),full_output=True)
            xmin, fval, iter, fcalls = opt.brent(CostFuncF,args=(self,i),full_output=True)
            if isnan(fval):
                self.f_opt[i] = NaN
                self.z_opt[i] = NaN
            else:
                ffac = xmin**2
                self.f_opt[i] = ffac
                self.z_opt[i] = self.getPlume1(i,ffac=self.f_opt[i])

            if Verbose:
                print "%5d | %6.2f | %8.2f | %3d | %3d "%(i, ffac, fval, iter, fcalls)

        if Verbose:
            print "------|--------|----------|-----|-----"

#---
    def getOptA(self,Verbose=True):
        """
        Find optimal fire modified bstar to match observed plume height.
        """

        if Verbose:
            print ""
            print "    Fire Modified bstar Optimization"
            print ""
            print "Plume | f_opt  |    J     |  Ni | Nf"
            print "------|--------|----------|-----|-----"

        self.f_opt = ones(self.N)
        self.z_opt = ones(self.N)
        for i in range(self.N):
            xmin, fval, iter, fcalls = opt.brent(CostFuncA,args=(self,i),full_output=True)
            afac = xmin**2
            self.f_opt[i] = afac
            self.z_opt[i] = self.getPlume1(i,afac=self.f_opt[i])
            if Verbose:
                print "%5d | %6.2f | %8.2f | %3d | %3d "%(i, afac, fval, iter, fcalls)

        if Verbose:
            print "------|--------|----------|-----|-----"

#---
    def getOptFanneal(self,Verbose=True):
        """
        Find optimal fire hflux scaling to match observed plume height.
        """

        if Verbose:
            print ""
            print "    Fire Modified bstar Optimization"
            print ""
            print "Plume | f_opt  |    J     |  Ni | Nf"
            print "------|--------|----------|-----|-----"

        self.f_opt = ones(self.N)
        self.z_opt = ones(self.N)
        for i in range(self.N):
            xmin, fval, T, fcalls, iter, accept, retval = opt.anneal(CostFuncF,1.0,args=(self,i),full_output=True)
            ffac = xmin**2
            self.f_opt[i] = ffac
            self.z_opt[i] = self.getPlume1(i,ffac=self.f_opt[i])
            if Verbose:
                print "%5d | %6.2f | %8.2f | %3d | %3d "%(i, ffac, fval, iter, fcalls)

        if Verbose:
            print "------|--------|----------|-----|-----"

#---
    def getOptFbnd(self,Verbose=True):
        """
        Find optimal fire hflux scaling to match observed plume height.
        """

        if Verbose:
            print ""
            print "    Fire Modified bstar Optimization"
            print ""
            print "Plume | f_opt  |    J     |  Ni | Nf"
            print "------|--------|----------|-----|-----"

        self.f_opt = ones(self.N)
        self.z_opt = ones(self.N)
        for i in range(self.N):
            xmin,fval,ier,fcalls  = opt.fminbound(CostFuncF,0.,2.,args=(self,i),full_output=True)
            ffac = xmin**2
            self.f_opt[i] = ffac
            self.z_opt[i] = self.getPlume1(i,ffac=self.f_opt[i])
            if Verbose:
                print "%5d | %6.2f | %8.2f | %3d | %3d "%(i, ffac, fval, fcalls, fcalls)

        if Verbose:
            print "------|--------|----------|-----|-----"

#----
def CostFuncF(f,m,i):
    z = m.getPlume1(i,ffac=f**2)
    return 1e-6 * (m.z[i]-z)**2

def CostFuncA(f,m,i):
    z = m.getPlume1(i,afac=f**2)
    return 1e-6 * (m.z[i]-z)**2

#----------------------------------------------------------------------------------------------

class MOD14(object):
    def __init__(self,N):
        """
        Simple container class for fire properties.
        """
        self.hflux = __AMISS__ * ones(N)  # fire heat flux estimate
        self.fdist = __AMISS__ * ones(N)  # distance to detected fire
        self.frp   = __AMISS__ * ones(N)  # nearest MODIS FRP
        self.pixar = __AMISS__ * ones(N)  # pixel area
        self.farea = __AMISS__ * ones(N)  # fire area
        self.qa    = __AMISS__ * ones(N)  # quality flag
        return
    
#----------------------------------------------------------------------------------------------
class PLUME_L2(IGBP,DOZIER):

    """
    Extension of the MxD14,IGBP and DOZIER classes, adding the
    Plume Rise functionality. This class handles non-gridded,
    observation location fires.
    """
    
    def getPlume(self,met):
 
        """
        Runs the Plume Rise extension to compute the extent of the plume.
        On input,

        met --- MET object for computing Met fields

        Notice that the Dozier must have been run and produced the farea
        attribute. In addition, the veg attribute with the biome type must
        have been defined as well.
	"""

        if self.algo != 'dozier':
            raise ValueError, 'only Dozier algorithm supported for now'

        if self.veg is None:
            raise ValueError, 'veg attribute with biome type has not been defined'


#       Initialize Plume Rise ranges
#       ----------------------------
        ntd = met.ntd
        N = self.lon.size
        self.p_plume = zeros((ntd,N,2))
        self.k_plume = zeros((ntd,N,2))
        self.z_plume = zeros((ntd,N,2))
        yyyy = self.yyyy[N/2]
        jjj = self.jjj[N/2]

#       Loop over time
#       --------------
        for t in range(1,ntd+1):
    
#           Interpolate met fields to fire location
#           ---------------------------------------
            met.interp(yyyy,jjj,t,lon=self.lon,lat=self.lat)

#           Compute plume rise for this time
#           --------------------------------
            farea = self.r_F * self.farea # reduce area by flaming fraction
            p, z, k = getPlume(farea,self.veg,met,t,ntd,self.verb)  

#           Save in the approapriate containers
#           -----------------------------------
            self.p_plume[t-1,:,:] = p[:,:]
            self.z_plume[t-1,:,:] = z[:,:]
            self.k_plume[t-1,:,:] = k[:,:]

#...............................................................................

class PLUME_L3(object):

    """
    Extension of the MxD14,IGBP and DOZIER classes, adding the
    Plume Rise functionality. This class handles non-gridded,
    observation location fires.
    """

    def __init__(self,plume,refine=4,res=None):
        """
        Create a gridded Plume Rise object from a Level 2
        PLUME_L2 object *plume*. The grid resolution is
        specified by

        refine  -- refinement level for a base 4x5 GEOS-5 grid
                       refine=1  produces a   4  x  5    grid
                       refine=2  produces a   2  x2.50   grid
                       refine=4  produces a   1  x1,25   grid
                       refine=8  produces a  0.50x0.625  grid
                       refine=16 produces a  0.25x0.3125 grid

        Alternatively, one can specify the grid resolution with a
        single letter:

        res     -- single letter denoting GEOS-5 resolution,
                       res='a'  produces a   4  x  5    grid
                       res='b'  produces a   2  x2.50   grid
                       res='c'  produces a   1  x1,25   grid
                       res='d'  produces a  0.50x0.625  grid
                       res='e'  produces a  0.25x0.3125 grid

                   NOTE: *res*, if specified, supersedes *refine*.

        After initialization only the FRP weighted average area is
        set, for each gridbox/biome, along with the coordinates of the
        global grid.
        
        """

        N = plume.lon.size
        self.verb = plume.verb

#       Output grid resolution
#       ----------------------
        if res is not None:
            if res=='a': refine = 1 
            if res=='b': refine = 2
            if res=='c': refine = 4
            if res=='d': refine = 8
            if res=='e': refine = 16

#       Lat lon grid
#       ------------
        dx = 5. / refine
        dy = 4. / refine
        im = int(360. / dx)
        jm = int(180. / dy + 1)
        self.im = im
        self.jm = jm
        self.glon = linspace(-180.,180.,im,endpoint=False)
        self.glat = linspace(-90.,90.,jm)
        Lat, Lon  = meshgrid(self.glat,self.glon)  # shape should be (im,jm)

        self.yyyy = plume.yyyy[N/2]
        self.jjj  = plume.jjj[N/2]
        self.date = date((int(self.yyyy),1,1)) + (int(self.jjj) - 1)*DAY
        self.col  = plume.col
        
#       Supperobed fire attributes for each biome
#       These will have 1D arrays for each biome, using
#       a standard sparse matrix storage
#       -----------------------------------------------
        self.bioma = [1,2,3,4]
        NONE       = [None,None,None,None] 
        self.idx   = NONE[:]  # non-zero indices for each biome
        self.area  = NONE[:]  # non-zero areas   for 
        self.r_F   = NONE[:]  # corresponding flaming fraction
        self.lon   = NONE[:]  # corresponding lon
        self.lat   = NONE[:]  # corresponding lat

#       Plume extent to be filled later
#       -------------------------------
        self.p_plume = NONE[:]
        self.k_plume = NONE[:]
        self.z_plume = NONE[:]

#       Grid box average of fire flaming area, weighted by FRP
#       -----------------------------------------------------
        for b in self.bioma:

#           Compute average area in gridbox for this biome
#           ----------------------------------------------
            Frac = zeros((im,jm))
            Area = zeros((im,jm))
            FRP  = zeros((im,jm))
            i = (plume.veg==b)
            if any(i):
                blon = plume.lon[i]
                blat = plume.lat[i]
                bfrac = plume.r_F[i] * plume.pow[i] 
                barea = plume.r_F[i] * plume.farea[i] * plume.pow[i] # notice flaming fraction 
                bfrp  = plume.pow[i]
                Area +=  binareas(blon,blat,barea,im,jm) # to be normalized
                Frac +=  binareas(blon,blat,bfrac,im,jm) # to be normalized
                FRP  +=  binareas(blon,blat,bfrp, im,jm)
                I = (FRP>0.0)
                if any(I):
                    Area[I] = Area[I] / FRP[I]
                    Frac[I] = Frac[I] / FRP[I]

#           Use sparse matrix storage scheme
#           --------------------------------
            I = Area.nonzero()
            if any(I):
                self.area[b-1] = Area[I]
                self.r_F[b-1]  = Frac[I] # average flaming/total energy fraction
                self.lon[b-1]  = Lon[I]
                self.lat[b-1]  = Lat[I]
                self.idx[b-1]  = I       # save indices for going to global grid

#---
    def getPlume(self,met):
 
        """
        Runs the Plume Rise extension to compute the extent of the plume.
        On input,

        met --- MET object for computing Met Fields at obs location.

	"""

        ntd = met.ntd # number of time steps per day
        self.ntd = ntd
        
#       Loop over bioma
#       ---------------
        for i in range(len(self.bioma)):

#           No data for this bioma, nothing to do
#           -------------------------------------
            if self.idx[i] is None:
                if self.verb>0:
                    print "[x] no data for %s"%Bioma[i] 
                continue

            lon = self.lon[i]
            lat = self.lat[i]
            area = self.area[i]
            N = lon.size
            veg = self.bioma[i] * ones(N)

            p_plume = zeros((ntd,2,N))
            z_plume = zeros((ntd,2,N))
            k_plume = zeros((ntd,2,N))
                            
            if self.verb>0:
                print "[ ] got %d burning gridboxes in %s"%(N,Bioma[i]) 

#           Loop over time 
#           --------------
            for t in range(1,ntd+1):

#               Interpolate met fields to fire locations
#               ----------------------------------------
                met.interp(self.yyyy,self.jjj,t,lon=lon,lat=lat)

#               Compute plume rise for this time, biome
#               ---------------------------------------
                p, z, k = getPlume(area,veg,met,t,ntd,self.verb)  

#               Save in the approapriate containers
#               -----------------------------------
                p_plume[t-1,:,:] = p.T[:,:]
                z_plume[t-1,:,:] = z.T[:,:]
                k_plume[t-1,:,:] = k.T[:,:]

#           Plume extent for this biome (sparse storage)
#           --------------------------------------------
            self.p_plume[i] = p_plume
            self.z_plume[i] = z_plume
            self.k_plume[i] = k_plume

#---
    def write(self,filename=None,dir='.',expid='qfed2',tag=None):
       """
       Writes gridded Area and FRP to file.
       """

       vtitle = {}
       vtitle['fa'] = 'Flaming Area'
       vtitle['ff'] = 'Fraction of Flaming Energy'
       vtitle['p2'] = 'Plume Bottom Pressure'
       vtitle['p1'] = 'Plume Top Pressure' 
       vtitle['z2'] = 'Plume Bottom Height' 
       vtitle['z1'] = 'Plume Top Height' 
       vtitle['k2'] = 'Plume Bottom Vertical Index' 
       vtitle['k1'] = 'Plume Top Vertical Index' 
                 
       vunits = {}
       vunits['fa'] = 'km2'
       vunits['ff'] = '1'
       vunits['p2'] = 'Pa'
       vunits['p1'] = 'Pa'
       vunits['z2'] = 'meter'
       vunits['z1'] = 'meter'
       vunits['k2'] = '1'
       vunits['k1'] = '1'
                 
       btitle = {}
       btitle['tf'] = 'Tropical Forest' 
       btitle['xf'] = 'Extra-Tropical Forest'
       btitle['sv'] = 'Savanna'
       btitle['gl'] = 'Grassland'

#      Create master variable list
#      ---------------------------
       Vname  = []
       Vtitle = []
       Vunits = []
       for v in vtitle.keys():
           vt = vtitle[v]
           vu = vunits[v]
           for b in btitle.keys(): 
               bt = btitle[b]
               Vname.append(v+'_'+b)
               Vtitle.append(vt+' ('+bt+')')
               Vunits.append(v)

#      Global metadata
#      ---------------
       title = 'QFED Level3c v%3.1f (%s) Gridded Plume Rise Estimates' % (__VERSION__, _getTagName(tag))
       source = 'NASA/GSFC/GMAO GEOS-5 Aerosol Group'
       contact = 'arlindo.dasilva@nasa.gov'

#      Time/date handling
#      ------------------
       if self.date is None:
           print "[x] did not find matching files, skipped writing an output file"
           return

       if 24%self.ntd != 0:
           raise ValueError,"invalid number of times per day (%d),"%self.ntd\
                 +"it must be a divisor of 24."
       else:
           dT = 240000/self.ntd # timestep in hhmmss format
           NHMS = range(0,240000,dT)

       nymd = 10000*self.date.year + 100*self.date.month + self.date.day
       nhms = NHMS[0]
       col = self.col

#      Create output file name
#      -----------------------
       if filename is None:
           filename = '%s/%s.plumerise.%s.%d.nc'%(dir,expid,col,nymd)
       self.filename = filename
       f = GFIO()
       f.create(filename,Vname, nymd, nhms,
                lon=self.glon, lat=self.glat,
                vtitle=Vtitle, vunits=Vunits,
                timinc=dT, amiss=__AMISS__,
                title=title, source=source, contact=contact)

#      Write out Plume Rise variables
#      ------------------------------
       d = (self.im,self.jm)
       for t in range(self.ntd):
           nhms = NHMS[t]
           b = 0
           for bn in btitle.keys():
               I = self.idx[b]
               f_area  = self.area[b]
               f_frac  = self.r_F[b]
               p_plume = self.p_plume[b]
               z_plume = self.z_plume[b]
               k_plume = self.k_plume[b]
               _writeOne(f,'fa_'+bn,nymd,nhms,I,f_area, t,0,d)
               _writeOne(f,'ff_'+bn,nymd,nhms,I,f_frac, t,0,d)
               _writeOne(f,'p1_'+bn,nymd,nhms,I,p_plume,t,0,d)
               _writeOne(f,'p1_'+bn,nymd,nhms,I,p_plume,t,0,d)
               _writeOne(f,'p2_'+bn,nymd,nhms,I,p_plume,t,1,d)
               _writeOne(f,'z1_'+bn,nymd,nhms,I,z_plume,t,0,d)
               _writeOne(f,'z2_'+bn,nymd,nhms,I,z_plume,t,1,d)
               _writeOne(f,'k1_'+bn,nymd,nhms,I,k_plume,t,0,d)
               _writeOne(f,'k2_'+bn,nymd,nhms,I,k_plume,t,1,d)
               b += 1

       try:
           f.close()
       except:
           pass

       if self.verb >=1:
           print "[w] Wrote file "+filename

#..............................................................................

#
#                                    Static Methods
#                                    --------------
#

def _writeOne(f,vname,nymd,nhms,I,S,t,k,d):
    """
    Write one sparse variable to a GFIO file.
    """
    A = zeros(d) + __AMISS__
    if I is not None:
        if len(S.shape)==3:
            A[I] = S[t,k,:]
        elif len(S.shape)==1:
            A[I] = S[:]
        else:
            raise ValueError, 'invalid S rank = %d'%len(S.shape)
            
    f.write(vname,nymd,nhms,A)


def getPlume(farea,veg,met,t,ntd,Verb=0):
    
    """
    Runs the Plume Rise extension to compute the extent of the plume.

          p, z, k = getPlume(farea,veg,met,t,ntd)

    where p, z and k are nd-arrays of shape(N,2), N being the
    number of observations (as in met.lon.size). On input,

    farea --- (flaming) fire area
    veg   --- biome type
    
    """

    N = met.lon.size
    nominal_area = 1.e6 # 1 km^2: pixar and farea are in units of km2
    km = met.lev.size
    ptop = met.ptop
    ktop = met.ktop # 1-offset

    if Verb:
        if N>100:
            Np = range(0,N,N/10)
        elif N>10:
            Np = range(0,N,N/10)
        else:
            Np = range(N)
        print ""
        print "                   Plume Rise Estimation for t=%d"%t
        print "                   ------------------------------"
        print ""
        print "  %  |    Lon    Lat  b |   p_bot    p_top  |  z_bot z_top  |  k   k"     
        print "     |    deg    deg    |    mb       mb    |   km     km   | bot top"
        print "---- |  ------ ------ - | -------- -------- | ------ ------ | --- ---"

#   Allocate space
#   --------------
    p_plume = zeros((N,2))
    k_plume = zeros((N,2))
    z_plume = zeros((N,2))
        
#   Compute plume extent, one fire at a time
#   ----------------------------------------
    for i in range(N):

        u = met.fields['u'][i]
        v = met.fields['v'][i]
        T = met.fields['t'][i]
        q = met.fields['qv'][i]
        delp = met.fields['delp'][i]

#       Ensure arrays are top-down as in GEOS-5
#       ---------------------------------------
        if delp[0] > delp[-1]:
            u = u[::-1]
            v = v[::-1]
            T = T[::-1]
            q = q[::-1]
            delp = delp[::-1]

#       Units:
#           farea - km2 (must multiply by nominal area for m2)
#            area  - m2 as required by plume rise model
#       ------------------------------------------------------   
        area = farea[i] * nominal_area
        veg_ = veg[i]

#       Run plume rise model
#       --------------------
        p1, p2, z1, z2, k1, k2, rc = \
            biome(u, v, T, q, delp, ptop, area, veg_)

        k1, k2 = (k1+ktop-1, k2+ktop-1)

        p_plume[i,:] = (p1, p2)
        k_plume[i,:] = (k1, k2)
        z_plume[i,:] = (z1, z2)

        if Verb:
            if i in Np:
                ip = int(0.5+100.*i/N)
                print "%3d%% | %7.2f %6.2f %d | %8.2f %8.2f | %6.2f %6.2f | %3d %3d "%\
                      (ip,met.lon[i],met.lat[i],veg[i], \
                       p2/100,p1/100,z2/1000,z1/1000,k2,k1)

    return (p_plume, z_plume, k_plume)


def _getTagName(tag):
    if tag != None:
        tag_name = tag
    else:    
        if __CVSTAG__ not in (None, ''):
            tag_name = __CVSTAG__
        else:
            tag_name = 'unknown'

    return tag_name

#----
def _getGran(t,mod14_path):
    doy = t.date().toordinal() - date(t.year-1,12,31).toordinal()
    hhmm = "%02d%02d"%(t.hour,5*(t.minute/5))
    patt = mod14_path+'/%4d/%03d/MOD14.A%4d%03d.%s.005.*.hdf'%(t.year,doy,t.year,doy,hhmm)
#    print '--> ', patt
    return glob(patt)

#---

#..............................................................................

if __name__ == "__main__":

#    m = MINXs_PR('/home/adasilva/workspace/misrPlumes/canada2008/Plumes*.txt')

    m = MINXs_PR('/home/adasilva/workspace/misrPlumes/Siberia2006/Plumes*.txt')
    m.sampleLoadz('merraero.npz')
    m.mod14 = NPZ('mod14.npz')
#    z_plume = m.getPlume()

def Mac():
    m = MINXs_PR('/Users/adasilva/workspace.local/misrPlumes/canada2008/Plumes*.txt')
#    m.sampleLoadz('/Users/adasilva/workspace.local/misrPlumes/canada2008/merra.npz')
#    m.sampleLoadz('merra.npz')
#    m.mod14 = NPZ('mod14.npz')
#    z_plume = m.getPlume()
