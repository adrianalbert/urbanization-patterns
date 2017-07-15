import numpy as np
from skimage import morphology
import scipy.misc
import copy


class City(Object):
    """
    Interface to several common analysis tasks on a spatial distribution (map) representing observed quantities for a city.
    """
    def __init__(self, M, **kwargs):
        '''
        M is a map of k channels, each representing a spatial quantity.
        kwargs are other city attributes (such as its name, population etc.)
        '''
        self.M = M
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

    def analyze(self, c=0):
        '''
        Perform spatial statistics analysis for given map (by channel).
        '''
        self.compute_regions(c=c)
        self.compute_fractal_dim(c=c)
        self.compute_profiles(c=c)


    def compute_regions(self, c=0):
        M = self.M.shape if self.M.shape.dim==2 else M[:,:,c]
        self.regions, self.mask_regions = compute_patch_areas(M)
        self.areas_distr = compute_patch_area_distribution(M)
        return self.regions

    def compute_fractal_dim(self, c=0):
        M = self.M.shape if self.M.shape.dim==2 else M[:,:,c]
        self.fractal_dim = fractal_dimension(M)  
        return self.fractal_dim

    def get_regions(self, c=0, patches=[0]):
        mask = self.mask.copy()
        to_modify = [self.regions[p][0] for p in patches]
        mask_sel = np.zeros(mask.shape)
        regions = []
        for m in to_modify:
            mask_sel[mask == m] = 1
            bnds = np.nonzero(mask[:,:,c] == m)
            regions.append(bnds)
        img[:,:,c][mask_sel[:,:,c]>0] *= (1 + amount)
        return img, regions

    def compute_profile(self, c=0, loc=None, step=10, n_samples=100):
        M = self.M.shape if self.M.shape.dim==2 else M[:,:,c]
        H,W = M.shape
        if loc is None:
            x0, y0 = H/2, W/2
        else:
            x0, y0 = loc
        theta, rays_binned = extract_rays(img, x0, y0, step=step, \
                                n_samples=n_samples)

        # TODO: compute mean & standard dev for profiles with distance
        self.profiles = rays_binned 


def compute_patch_areas(M):
    mask = morphology.label(M)
    areas = []
    for i in np.arange(1,mask.max()):
        areas.append((i,(mask==i).sum()))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas, mask

def compute_patch_area_distribution(img, bins=None):
    areas, mask = compute_patch_areas(img)
    area_sizes = [x[1] for x in areas]
    N_counts, bins = np.histogram(area_sizes, bins=bins)
    mask = None; area_sizes = None
    return np.log(N_counts), bins

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


def extract_rays(img, x0, y0, step=10, n_samples=100):
    n = n_samples / 4
    H,W = img.shape
    x = np.random.randint(0, W, n).tolist() + np.random.randint(0, W, n).tolist() + \
        np.zeros(n).tolist() + np.zeros(n).tolist()
    y = np.zeros(n).tolist() + np.zeros(n).tolist() + \
        np.random.randint(0, H, n).tolist() + np.random.randint(0, H, n).tolist() 
    xy = zip(x, y)
    
    # for each endpoint, extract ray
    theta = []
    rays = []
    rays_binned = np.zeros((len(xy), int(H/step)+1))
    for i,(x1,y1) in enumerate(xy):
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        n_steps = int(np.ceil(d / step))
        ray_x = np.linspace(x0, x1, n_steps)
        ray_y = np.linspace(y0, y1, n_steps)
        r = scipy.ndimage.map_coordinates(img, np.vstack((ray_x, ray_y)))
        t = np.degrees(np.arccos((x1-x0)/float(d)))
        theta.append(t)
        cutoff = min([len(r), int(H/step)+1])
        rays_binned[i,:cutoff] = r[:cutoff]
    
    return theta, rays_binned
