#
# Ben Ghertner 2025
#
# A quick interactive interface to view CM1 outputs
# may need to be modified for your particular simulations
#
# File paths must be hardcoded to output and background files
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons, Slider
import xarray as xr

### PARAMETERS ###

#slice in the y direction to view
idy = 0

#You shouldn't change these...
Tref = 273. #[K]
pref = 1e5  #[Pa]

###############################################################
#                                                             #
#                  Enter your file paths here                 #
#                                                             #
###############################################################

input_file_path = '../thesis_figures/cm1DispersionTest/domain_1000/cm1out.nc'
base_file_path = '../thesis_figures/cm1DispersionTest/domain_1000/cm1base.nc'

###############################################################

#number of quiver plot arrows in each direction
quivnum = 20

### HELPER FUNCTIONS ###

def nicecontour(X, Y, F, ax, c='k', c0='b', levels=None, num=5):
    
    if levels is None:
        maxval = np.max(np.abs(F))
        if maxval == 0.:
            maxval = 1.
        levels = np.linspace(0, maxval, num=num+1)[1:]
    
    ax.contour(X,Y,F,levels=levels,colors=c)
    ax.contour(X,Y,F,levels=np.flip(-levels),colors=c)
    ax.contour(X,Y,F,levels=[0],colors=c0,linewidths=1., zorder=3)

    return levels[1] - levels[0]

def nicecontourf(X, Y, F, ax, levels=None, num=5, pos=True, cmap='coolwarm'):

    if levels is None:
        maxval = np.max(np.abs(F))
        if maxval == 0.:
            maxval = 1.
        if pos:
            levels = np.linspace(0, maxval, num=num+1)
        else:
            levels = np.linspace(-maxval, maxval, num=2*num)
    
    contf = ax.contourf(X,Y,F,levels=levels, cmap=cmap)
    
    return contf

def nicequiver(X, Y, U, V, skipx, skipy, ax):
    Uh = (U[:,:-1] + U[:,1:])/2
    Vh = (V[:-1,:] + V[1:,:])/2
    quiver = ax.quiver(X[::skipx], Y[::skipy], Uh[::skipy,::skipx], Vh[::skipy,::skipx],
                       pivot='mid',angles='xy', scale_units='xy', zorder=2)
    return quiver

#TODO write this method...
def get_streamfunc(X,Y,U,V):
    return

def rslf(T, p, method=1):
    eps = 287.04/461.5
    if method == 1:
        esl = 611.2 * np.exp( 17.67 * ( T  - 273.15 ) / ( T  - 29.65 ) )
        esl = np.minimum(esl, p*0.5)
        return eps*esl/(p-esl)
    elif method == 2:
        return 380.00*np.exp(17.2693882-4097.8531/(T-35.86))/p
    else:
        esl = 611.2 * np.exp( 17.67 * ( T  - 273.15 ) / ( T  - 29.65 ) )
        esl = np.minimum(esl, p*0.5)
        return eps*esl/(p-esl)
    
def thetal(pref, p, T, rT):
    rv = rslf(T,p)
    rl = np.maximum(rT - rv,0)

    cpT = cpd + cpv*rT
    RT = Rd + Rv*rT
    lv = (cpv-cl)*(T-273.15) + 2501000
    
    expM = ((Rd + rv*Rv)/RT)**(RT/cpT)*(rT/rv)**(rT*Rv/cpT) #mixing terms
    thl = T*(pref/p)**(RT/cpT)*np.exp(-lv*rl/cpT/T)*expM

    return thl

### IMPORT AND PROCESS DATA ###

print('Loading data. This may take a minute...')

Rd = 287.05
Rv = 461.51
cpd= 1.0057e3
cpv= 1870
cl = 4190
g = 9.81
rref = rslf(Tref,pref)
lvref = (cpv-cl)*(Tref-273.15) + 2501000

dataset = xr.open_dataset(input_file_path)
basedata = xr.open_dataset(base_file_path)

tmax_idx = int(1e5)

time = (dataset['time'][:tmax_idx].astype(float).to_numpy()*1e-9) #[s]
xh = dataset['xh'].values #[km]
xf = dataset['xf'].values #[km]
zf = dataset['zf'].values #[km]
zh = dataset['zh'].values #[km]

w = dataset['w'][:,:,idy,:tmax_idx].to_numpy()
u = dataset['u'][:,:,idy,:tmax_idx].values
u0 = basedata['u'][0,:,0,0:1].values

prs = (dataset['prs'][:,:,idy,:tmax_idx]).to_numpy()
T = (prs/pref)**(Rd/cpd)*(dataset['th'][:,:,0,:tmax_idx]).to_numpy()
rv = (dataset['qv'][:,:,idy,:tmax_idx]).to_numpy()
rl = (dataset['ql'][:,:,idy,:tmax_idx]).to_numpy()
rT = rv + rl
N2 = (dataset['nm'][:,:,idy,:tmax_idx]).to_numpy()

prs0 = (basedata['prs'][0,:,0,:]).to_numpy()
T0 = (prs0/pref)**(Rd/cpd)*(basedata['th'][0,:,0,:]).to_numpy()
rT0 = ((basedata['qv'][0,:,0,:]) + (basedata['ql'][0,:,0,:])).to_numpy()

thl0 = thetal(pref,prs0,T0,rT0)

thl = thetal(pref,prs,T,rT)
thlpmax = np.nanmax(np.abs(thl/thl0 - 1)[np.isfinite(thl)])

thlplevs = np.linspace(-thlpmax, thlpmax, num=20)

#fixed w levs
wmax = np.nanmax(np.abs(w))
wmax = np.minimum(wmax, 0.5)
wlevs = np.linspace(0, wmax, num=12)[1:]

### PLOTTING CODE ###

# Global variables to track state
idt = 0
current_mode = 'Vertical Velocity'
filled_mode = 'BVF'
CI = 0
CI_unit = '[m/s]'
quiver_visible = False
contpert_on = True
contfpert_on = True

# Sample data generation (replace this with your actual data)
time_steps = time.size
nx, ny = xh.size, zh.size
skipx = int(nx/quivnum)
skipy = int(ny/quivnum)

# Initialize plot
fig, ax = plt.subplots()
fig.set_size_inches(11,7)
plt.subplots_adjust(left=0.065, right=0.7, top=0.92, bottom=0.15)

CI = nicecontour(xh, zf, w[0], ax)
contf = nicecontourf(xh, zf, N2[0], ax)
cb = fig.colorbar(contf, fraction=0.075, format='{x:.1e}')

quiver = nicequiver(xh, zh, u[0], w[0], skipx, skipy, ax)
quiver.set_visible(False)

ql_mask = np.where(rl[0] > 0, 1, np.nan)
ax.contourf(xh, zh, ql_mask, cmap='gray', alpha=0.3)

# Radio buttons for mode selection
rax = plt.axes([0.75, 0.75, 0.2, 0.15])
radio = RadioButtons(rax, ('Vertical Velocity', 'Horizontal Velocity', 'Vorticity'))
rax.axis('off')

# Radio buttons for filled mode selection
raxf = plt.axes([0.75, 0.5, 0.2, 0.2])
radiof = RadioButtons(raxf, ('BVF', 'Richardson Num.', 'LW Pot. Temp.', 'Mixing Ratio'))
raxf.axis('off')

# Check button for quiver plot
cax_quiver = plt.axes([0.75, 0.02, 0.2, 0.08])
check = CheckButtons(cax_quiver, ['Quiver?'], [False])

# Check button for contour pert
contpert = plt.axes([0.75, 0.12, 0.2, 0.08])
contcheck = CheckButtons(contpert, ['cont pert?'], [True])

# Check button for filled contour pert
contfpert = plt.axes([0.75, 0.22, 0.2, 0.08])
contfcheck = CheckButtons(contfpert, ['filled cont pert?'], [True])

# Buttons for scrolling through time and label for current time
button_ax = plt.axes([0.02, 0.02, 0.1, 0.04])
button_back = Button(button_ax, 'Previous')
button_ax_next = plt.axes([0.51, 0.02, 0.1, 0.04])
button_next = Button(button_ax_next, 'Next')

# Slider for time control
slider_ax = plt.axes([0.15, 0.02, 0.32, 0.04], facecolor='lightgoldenrodyellow')
time_slider = Slider(slider_ax, '', 0, time_steps-1, valinit=0, valstep=1)

# Save button
button_save_ax = plt.axes([0.63, 0.02, 0.1, 0.04])
button_save = Button(button_save_ax, 'Save Plot')


def update_plot():
    global idt, current_mode, CI, filled_mode, cb, quiver_visible
    cb.remove()
    ax.clear()
    title_str = ''

    ### CONTOUR PLOT ###
    if current_mode == 'Horizontal Velocity':
        if contpert_on:
            CI = nicecontour(xf, zh, u[idt]-u0, ax)
            title_str += r'Contour: $\tilde{u}$,  '
        else:
            CI = nicecontour(xf, zh, u[idt], ax)
            title_str += r'Contour: $u$,  '
        CI_unit='[m/s]'
        
    elif current_mode == 'Vertical Velocity':
        CI = nicecontour(xh, zf, w[idt], ax, levels=wlevs)
        CI_unit='[m/s]'
        title_str += r'Contour: $\tilde{w}$,  '

    elif current_mode == 'Vorticity':
        if contpert_on:
            uz = (u[idt,1:,:]-u0[1:,:] - u[idt,:-1,:]+u0[:-1,:])/(zh[1] - zh[0])
            title_str += r'Contour: $\tilde{\eta}$,  '
        else:
            uz = (u[idt,1:,:] - u[idt,:-1,:])/(zh[1] - zh[0])/1000
            title_str += r'Contour: $\eta$,  '
        wx = (w[idt,:,1:] - w[idt,:,:-1])/(xh[1] - xh[0])/1000
        CI = nicecontour(xf[1:-1], zf[1:-1], uz[:,1:-1] - wx[1:-1,:], ax)
        CI_unit = r'[s$^{-1}$]'
    
    ### QUIVER PLOT ###
    if quiver_visible:
        nicequiver(xh, zh, u[idt], w[idt], skipx, skipy, ax)

    ### FILLED CONTOUR PLOT ###
    if filled_mode == 'BVF':
        contf = nicecontourf(xh, zf, N2[idt], ax, levels=np.linspace(np.min(N2[idt]), np.max(N2[idt]), num=7))
        cb = fig.colorbar(contf, fraction=0.075, format='{x:.1e}')
        title_str += r'Filled: $N^2$,  '
        if np.min(N2[idt]) < 0:
            ax.contour(xh, zf, N2[idt], levels=[0], colors='r', zorder=4, linewidths=1., linestyles='--')

    elif filled_mode == 'Richardson Num.':
        uh = (u[idt,:,:-1] + u[idt,:,1:])/2
        uprime = (uh[1:,:] - uh[:-1,:])/(zf[1] - zf[0])/1000
        Ri = N2[idt,1:-1,:]/uprime**2
        contf = nicecontourf(xh, zf[1:-1], Ri, ax, levels=np.array([0.,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.]),cmap='Purples_r')
        ax.contour(xh, zf[1:-1], Ri, levels=[1/4], colors='r', zorder=4, linewidths=1., linestyles='--')
        cb = fig.colorbar(contf, fraction=0.075, format='{x:.1e}')
        title_str += r'Filled: $N^2$,  '

    elif filled_mode == 'LW Pot. Temp.':
        thl = thetal(pref, prs[idt], T[idt], rT[idt])
        if contfpert_on:
            contf = nicecontourf(xh, zh, thl/thl0 - 1, ax, pos=False, levels=thlplevs)
            title_str += r'Filled: $\tilde{\theta}_\ell$,  '
        else:
            contf = nicecontourf(xh, zh, thl, ax, pos=True, levels=np.linspace(np.min(thl), np.max(thl), num=7))
            title_str += r'Filled: $\theta_\ell$,  '
        cb = fig.colorbar(contf, fraction=0.075, format='{x:.1e}')

    elif filled_mode == 'Mixing Ratio':
        if contfpert_on:
            contf = nicecontourf(xh, zh, rT[idt]/rT0 - 1, ax, pos=False)
            title_str += r'Filled: $\tilde{r}_T$,  '
        else:
            contf = nicecontourf(xh, zh, rT[idt], ax, pos=True)
            title_str += r'Filled: $r_T$,  '
        cb = fig.colorbar(contf, fraction=0.075, format='{x:.1e}')
    
    ### DRAW CLOUD ###
    ql_mask = np.where(rl[idt] > 0, 1, np.nan)
    ax.contourf(xh, zh, ql_mask, cmap='gray', alpha=0.3)
    ax.contour(xh, zh, rl[idt], levels=[0], colors='cyan', zorder=3, linewidths=1.)
    ax.contour(xh, zh, rl[idt], levels=[1e-5], colors='cyan', zorder=3, linewidths=1., linestyles='--')

    ### TITLES AND AXIS LIMITS ###
    title_str += f'Time: {time[idt]:.0f}[s],  CI: {CI:.2e}' + CI_unit
    ax.set_xlim(np.min(xf),np.max(xf))
    ax.set_ylim(np.min(zh),np.max(zh))
    ax.set(xlabel=r'$x$[m]', ylabel=r'$z$[m]',title=title_str)
    plt.draw()

### BUTTONS AND KEY EVENTS ###
def next_time(event):
    global idt
    if idt < time_steps - 1:
        idt += 1
        time_slider.set_val(idt)
    update_plot()

def prev_time(event):
    global idt
    if idt > 0:
        idt -= 1
        time_slider.set_val(idt)
    update_plot()

def mode_func(label):
    global current_mode
    current_mode = label
    update_plot()

def mode_funcf(label):
    global filled_mode
    filled_mode = label
    update_plot()

def save_plot(event):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('cm1plot.png', bbox_inches=extent.expanded(1.45,1.3), dpi=300)
    update_plot()

def toggle_quiver(label):
    global quiver_visible
    quiver_visible = not quiver_visible
    update_plot()

def toggle_cont(label):
    global contpert_on
    contpert_on = not contpert_on
    update_plot()

def toggle_contf(label):
    global contfpert_on
    contfpert_on = not contfpert_on
    update_plot()

def update_time(val):
    global idt
    idt = int(val)
    update_plot()

def on_key(event):
    if event.key == 'right':
        next_time(None)
    elif event.key == 'left':
        prev_time(None)
    elif event.key == 'ctrl+s':
        save_plot(None)

radio.on_clicked(mode_func)
radiof.on_clicked(mode_funcf)
check.on_clicked(toggle_quiver)
contcheck.on_clicked(toggle_cont)
contfcheck.on_clicked(toggle_contf)
button_next.on_clicked(next_time)
button_back.on_clicked(prev_time)
button_save.on_clicked(save_plot)
time_slider.on_changed(update_time)
fig.canvas.mpl_connect('key_press_event', on_key)

### INITIALIZE PLOT ###
update_plot()
plt.show()
