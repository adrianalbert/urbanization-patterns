import numpy as np
from skimage import morphology
import scipy.misc
import copy
import scipy as sp


class City():
    """
    Interface to several common analysis tasks on a spatial distribution (map) representing observed quantities for a city.
    """
    def __init__(self, M, mask=None, **kwargs):
        '''
        M is a map of C channels, each representing a spatial quantity.
        kwargs are other city attributes (such as its name, population etc.)
        '''
        self.M = M if M.ndim == 3 else M[:,np.newaxis]
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        if mask is not None:
            self.mask = mask.astype(float)
            self.mask[mask==0] = np.nan
        else:
            self.mask = np.ones(self.M.shape[:2])
        self.M = self.M * self.mask[...,np.newaxis]

    def analyze(self):
        '''
        Perform spatial statistics analysis for given map.
        '''
        self.compute_average()
        self.compute_regions()
        self.compute_fractal_dim()
        self.compute_profile(method="radial")

    def compute_average(self, c=None):
        c=range(self.M.shape[2]) if c is None else c if type(c)==list else [c]
        avg_areas = np.nanmean(self.M[:,:,c], (0,1))
        if not hasattr(self, 'avg_areas'):
            self.avg_areas = {}
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            self.avg_areas[x] = avg_areas[y]           
        return self.avg_areas

    def compute_regions(self, c=None):
        c=range(self.M.shape[2]) if c is None else c if type(c)==list else [c]
        if not hasattr(self, 'regions'):
            self.regions = {}
            self.masks_regions = {}
            self.areas_distr = {}
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            regions, mask_regions = compute_patch_areas(self.M[:,:,y])
            log_counts, areas = compute_patch_area_distribution(regions, mask_regions)
            log_counts[log_counts<0] = 0
            self.regions[x], self.masks_regions[x], self.areas_distr[x] = regions, mask_regions, (log_counts[:10],areas[:10]) 
        return self.regions

    def compute_fractal_dim(self, c=None):
        c=range(self.M.shape[2]) if c is None else c if type(c)==list else [c]
        if not hasattr(self, 'fractal_dim'):
            self.fractal_dim = {}
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            fract_dim,_,_ = fractal_dimension(self.M[:,:,y])
            self.fractal_dim[x] = fract_dim
        return self.fractal_dim

    def compute_profile(self, c=None, loc=None, method="radial",**kwargs):
        c=range(self.M.shape[2]) if c is None else c if type(c)==list else [c]
        H,W,C = self.M.shape
        if loc is None:
            x0, y0 = H/2, W/2
        else:
            x0, y0 = loc
        if not hasattr(self, 'profiles'):
            self.profiles = {}
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            if method == "raysampling":
                mu,se=compute_profile_raysampling(self.M[:,:,y],x0,y0,**kwargs)
            elif method == "radial":
                mu,se = compute_profile_radial(self.M[:,:,y], x0, y0, **kwargs)
            else:
                return None
            self.profiles[x] = (mu, se)
        return self.profiles

    def get_regions(self, c=0, patches=[0]):
        mask = self.mask_regions[c].copy()
        to_modify = [self.regions[c][p][0] for p in patches]
        mask_sel = np.zeros(mask.shape)
        regions = []
        for m in to_modify:
            mask_sel[mask == m] = 1
            bnds = np.nonzero(mask[:,:,c] == m)
            regions.append(bnds)
        self.M[:,:,c][mask_sel[:,:,c]>0] *= (1 + amount)
        return img, regions


def compute_patch_areas(M):
    mask = morphology.label(M)
    areas = []
    for i in np.arange(1,mask.max()):
        areas.append((i,(mask==i).sum()))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas, mask

def compute_patch_area_distribution(areas, mask):
    area_sizes = [x[1] for x in areas]
    log_area_bins = np.logspace(np.log(min(area_sizes)), np.log(max(area_sizes)), 20, base=np.exp(1))
    N_counts, areas = np.histogram(area_sizes, bins=log_area_bins)
    return np.log(N_counts), areas

def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert(len(Z.shape) == 2)
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes[1:]), np.log(counts[:-1]), 1)
    return -coeffs[0], np.log(sizes), np.log(counts)


def get_regions(img0, patches=[0], c=0, amount=0.0):
    img = copy.copy(img0)
    areas, mask = compute_patch_areas(img)
    to_modify = [areas[p][0] for p in patches]
    mask_sel = np.zeros(mask.shape)
    regions = []
    for m in to_modify:
        mask_sel[mask == m] = 1
        bnds = np.nonzero(mask[:,:,c] == m)
        regions.append(bnds)
    img[:,:,c][mask_sel[:,:,c]>0] *= (1 + amount)
    return img, regions


def compute_profile_radial(M, x0, y0, step=10, **kwargs):
    W, H = M.shape
    mu = []
    sd = []
    y,x = np.ogrid[-1:H-1,-1:W-1]
    n_steps = int(H/(1.5*step))+1
    last_mask = []
    for n in np.arange(1,n_steps+1):
        R = n*step
        mx, my = np.where((x-x0)**2 + (y-y0)**2 <= R*R)
        mask = zip(mx,my)
        dmask= list(set(mask) - set(last_mask))     
        dM = np.array([M[i,j] for i,j in dmask])
        mu.append(np.nanmean(dM))
        sd.append(np.nanstd(dM))
        last_mask = mask
    mu = np.array(mu); 
    # scale = mu.max() + 1e-5; 
    scale = 1
    mu /= scale
    sd = np.array(sd); 
    sd /= np.sqrt(scale)
    return mu, sd
    

def compute_profile_raysampling(img, x0, y0, **kwargs):
    theta, rays = extract_rays(np.nan_to_num(img), x0, y0, **kwargs)
    rays_mu = np.nanmean(np.abs(rays), 0); 
    scale   = rays_mu.max() + 1e-5; 
    rays_mu = rays_mu / scale
    rays_se = np.nanstd(np.abs(rays), 0); 
    rays_se = rays_se / np.sqrt(scale)
    return rays_mu, rays_se


def extract_rays(img, x0, y0, step=10, n_samples=200):
    n = n_samples / 4
    H,W = img.shape
    x = np.random.randint(0, W-1, n).tolist() + np.random.randint(0, W-1, n).tolist() + np.zeros(n).tolist() + np.repeat(W-1,n).tolist()
    y = np.zeros(n).tolist() + np.repeat(H-1,n).tolist() + \
        np.random.randint(0, H-1, n).tolist()+np.random.randint(0, H-1, n).tolist()
    xy = zip(x, y)
    
    # for each endpoint, extract ray
    theta = []
    rays = []
    len_ray = int(H/(1.5*step))+1
    rays_binned = np.zeros((len(xy), len_ray))
    for i,(x1,y1) in enumerate(xy):
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        n_steps = int(np.ceil(d / step))
        ray_x = np.linspace(x0, x1, n_steps)
        ray_y = np.linspace(y0, y1, n_steps)
        r = scipy.ndimage.map_coordinates(img, np.vstack((ray_x, ray_y)))
        t = np.degrees(np.arccos((x1-x0)/float(d)))
        theta.append(t)
        cutoff = min([len(r), len_ray])
        rays_binned[i,:cutoff] = r[:cutoff]
    
    return theta, rays_binned
