import numpy as np
from skimage import morphology
import scipy.misc
import copy
import scipy as sp


class City():
    """
    Interface to several common analysis tasks on a spatial distribution (map) representing observed quantities for a city.
    """
    def __init__(self, M, mask=None, bounds=None, **kwargs):
        '''
        M is a map of C channels, each representing a spatial quantity.
        kwargs are other city attributes (such as its name, population etc.)
        mask == 0 represents areas where development is not possible (e.g., water bodies), and mask == 1 indicates areas that could be developed.
        bounds==1 encodes city administrative boundaries.
        '''
        self.M = M if M.ndim == 3 else M[:,np.newaxis]
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        if mask is not None:
            self.mask = mask.astype(float)
            self.mask[mask==0] = np.nan
        else:
            self.mask = np.ones(self.M.shape[:2])
        self.bounds = np.ones(self.M.shape[:2]) if bounds is None else (bounds>0).astype(int)
        newmask = self.mask * (~np.isnan(self.mask))
        self.M = self.M * newmask[...,np.newaxis]

    def analyze(self, within_bounds=False, step=1):
        '''
        Perform spatial statistics analysis for given map.
        '''
        self.compute_average(within_bounds=within_bounds)
        self.compute_regions(within_bounds=within_bounds)
        self.compute_fractal_dim(within_bounds=within_bounds)
        self.compute_profile(method="radial", within_bounds=within_bounds, step=step, bounds_layer=True)

    def _get_maps(self, c=None, within_bounds=False):
        c=range(self.M.shape[2]) if c is None else c if type(c)==list else [c]
        M = self.M[...,c]
        if within_bounds:
            M[self.bounds==0] = np.nan
        return M, c

    def compute_average(self, c=None, within_bounds=False):
        M, c = self._get_maps(c=c, within_bounds=within_bounds)
        suffix = '_bnds' if within_bounds else ''
        avg_areas = np.nanmean(M, (0,1))
        sum_areas = np.nansum(M, (0,1))
        if not hasattr(self, 'avg_areas'+suffix):
            setattr(self, 'avg_areas'+suffix, {})
            setattr(self, 'sum_areas'+suffix, {})
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            getattr(self, 'avg_areas'+suffix)[x] = avg_areas[y]           
            getattr(self, 'sum_areas'+suffix)[x] = sum_areas[y]           
        return getattr(self, 'avg_areas'+suffix)

    def compute_regions(self, c=None, within_bounds=False):
        M, c = self._get_maps(c=c, within_bounds=within_bounds)
        suffix = '_bnds' if within_bounds else ''
        if not hasattr(self,'regions'+suffix):
            setattr(self, 'regions' + suffix, {})
            setattr(self, 'masks_regions' + suffix, {})
            setattr(self, 'areas_distr' + suffix, {})
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            regions, mask_regions = compute_patch_areas(M[...,y])
            log_counts, areas = compute_patch_area_distribution(regions, mask_regions)
            if log_counts is None:
                continue
            log_counts[log_counts<0] = 0
            getattr(self, 'regions'+suffix)[x], getattr(self, 'masks_regions'+suffix)[x], getattr(self, 'areas_distr'+suffix)[x] = regions, mask_regions, (log_counts[:10],areas[:10]) 
        return getattr(self, 'regions'+suffix)

    def compute_fractal_dim(self, c=None, within_bounds=False):
        M, c = self._get_maps(c=c, within_bounds=within_bounds)
        suffix = '_bnds' if within_bounds else ''
        if not hasattr(self, 'fractal_dim'+suffix):
            setattr(self, 'fractal_dim'+suffix, {})
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        for x,y in zip(s,c):
            fract_dim,_,_ = fractal_dimension(M[...,y])
            getattr(self, 'fractal_dim'+suffix)[x] = fract_dim
        return getattr(self, 'fractal_dim'+suffix)

    def compute_profile(self, c=None, center=None, method="radial",within_bounds=False, bounds_layer=False, **kwargs):
        M, c = self._get_maps(c=c, within_bounds=within_bounds)
        suffix = '_bnds' if within_bounds else ''
        H,W,C = M.shape
        if center is None:
            x0, y0 = H/2, W/2
        else:
            x0, y0 = center
        if not hasattr(self, 'profiles'+suffix):
            setattr(self, 'profiles'+suffix, {})
        s = [self.sources[x] for x in c] if hasattr(self, 'sources') else c
        f_profile = compute_profile_raysampling if method=="raysampling" else compute_profile_radial
        for x,y in zip(s,c):
            mu,se=f_profile(M[...,y], x0, y0, **kwargs)
            getattr(self, 'profiles'+suffix)[x] = (mu, se)
        if bounds_layer and hasattr(self, 'bounds'):
            mu,se = f_profile(self.bounds, x0, y0, **kwargs)
            getattr(self, 'profiles'+suffix)['bnds'] = (mu,se)
        return getattr(self, "profiles"+suffix)
 
    def get_regions(self, c=0, patches=[0], within_bounds=False):
        M, _ = self._get_maps(c=c, within_bounds=within_bounds)
        suffix = '_bnds' if within_bounds else ''
        mask = getattr(self, 'mask_regions'+suffix)[c].copy()
        to_modify = [getattr(self, 'regions'+suffix)[c][p][0] for p in patches]
        mask_sel = np.zeros(mask.shape)
        regions = []
        for m in to_modify:
            mask_sel[mask == m] = 1
            bnds = np.nonzero(mask[:,:,c] == m)
            regions.append(bnds)
        M[...,c][mask_sel[:,:,c]>0] *= (1 + amount)
        return img, regions


def compute_patch_areas(M):
    mask = morphology.label(M)
    areas = []
    for i in np.arange(1,mask.max()):
        areas.append((i,(mask==i).sum()))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas, mask

def compute_patch_area_distribution(areas, mask):
    if len(areas)==0:
        return None, None
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


def compute_profile_radial(M, x0, y0, step=10, scale=False, **kwargs):
    W, H = M.shape
    W, H = W/2, H/2
    mu = []
    sd = []
    y,x = np.ogrid[-1:H-1,-1:W-1]
    n_steps = int(H/step)+1
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
    scale = mu.max() + 1e-5 if scale else 1
    mu /= scale
    sd = np.array(sd); 
    sd /= np.sqrt(scale)
    return mu, sd
    

def compute_profile_raysampling(img, x0, y0, scale=False, **kwargs):
    theta, rays = extract_rays(np.nan_to_num(img), x0, y0, **kwargs)
    rays_mu = np.nanmean(np.abs(rays), 0); 
    scale = mu.max() + 1e-5 if scale else 1
    rays_mu = rays_mu / scale
    rays_se = np.nanstd(np.abs(rays), 0); 
    rays_se = rays_se / np.sqrt(scale)
    return rays_mu, rays_se


def extract_rays(img, x0, y0, step=10, n_samples=200):
    n = n_samples / 4
    H,W = img.shape
    H,W = H/2, W/2
    x = np.random.randint(0, W-1, n).tolist() + np.random.randint(0, W-1, n).tolist() + np.zeros(n).tolist() + np.repeat(W-1,n).tolist()
    y = np.zeros(n).tolist() + np.repeat(H-1,n).tolist() + \
        np.random.randint(0, H-1, n).tolist()+np.random.randint(0, H-1, n).tolist()
    xy = zip(x, y)
    
    # for each endpoint, extract ray
    theta = []
    rays = []
    len_ray = int(H/step)+1
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
