import matplotlib.pyplot as plt
from matplotlib import cm, colors
from ads.sph_harm import Yij, Ybasis
import numpy as np
import os

def make_fig(xsph, v, directory, lmax=30):
    # GET BASES FOR MAX l = λ 
    Y = Yij(xsph, lmax=lmax)
    #print(len(Yb))
    
    # SOLVE THE COEFFICIENTS
    U,S,V = np.linalg.svd(Y, full_matrices=False)
    ahat = np.matmul( np.linalg.pinv(np.matmul(U, np.matmul( np.diag(S), V))), np.array(v))
    #print(ahat)
    
    # PLOT THE APPROXIMATE FUNCTION
    theta, phi = np.linspace(0, np.pi, 75), np.linspace(0, 2*np.pi, 150)
    THETA, PHI = np.meshgrid(theta, phi)
    R = np.zeros(shape=THETA.shape)
    lam = int(np.sqrt(len(ahat)))
    ind = 0
    for l in range(lam):
        for m in np.linspace(-l, l, 2*l+1):
            piece = Ybasis(l, int(m), THETA, PHI)
            R += ahat[ind] * piece * 627.5
            ind += 1
    R0 = np.max(R)*2.
    X = (R+R0) * np.sin(THETA) * np.cos(PHI)
    Y = (R+R0) * np.sin(THETA) * np.sin(PHI)
    Z = (R+R0) * np.cos(THETA)
    norm = colors.Normalize()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d', frame_on=False)
    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, #cmap=plt.get_cmap('cividis'),
        facecolors=cm.inferno(norm(R)),
        linewidth=.2, antialiased=True, alpha=.5, shade=False)
    #ax.plot_wireframe(
    #    X, Y, Z, rstride=1, cstride=1,
    #    color='b',
    #    linewidth=0.2, antialiased=False, alpha=.5)
    #ax.quiver(0, 0, R0, *xx[0], length=0.7, normalize=False,
    #                 arrow_length_ratio = 0.5, color='k')
    #ax.quiver(0, 0, R0, *x[0], length=R0/2, normalize=False,
    #                 arrow_length_ratio = 0.3, color='k')
    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='inferno'), orientation='vertical')
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Hide grid
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.view_init(elev=10, azim=120.)
    
    fig.savefig(os.path.join(directory,'V_l{}.png'.format(lmax)), transparent=True)
