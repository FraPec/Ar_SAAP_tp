import numpy as np
import matplotlib.pyplot as plt
import os

if __name__=='__main__':
    path = 'data/rho=1.20.npy'
    directory = os.fsencode(path)
    T = 2.0
    N = 2048
    rho = 1.2
    
    DeltaU_mean, k_Einstein, P_mean, P_std, lamb_v = np.load(path, allow_pickle=True)

    
    stirling_logNfact = N * np.log(N) - N
    m = 1.0
    de_broglie = 1 / np.sqrt(2 * np.pi * m * T)
    F_Einstein = - T * 3/2 * N *np.log(2 * np.pi * T / k_Einstein) + 3 * N * T * np.log(de_broglie)
    
    
    plt.figure()
    plt.plot(lamb_v, -DeltaU_mean/N, label=rf'$\rho = {rho:.2f}$')
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r'$\lambda$', fontsize=35)
    plt.ylabel(r'$\frac{\langle U_{harmonic} - U_{SAAP}\rangle_{\lambda}}{N}$', fontsize=35)
    plt.title(rf'$\rho = {rho:.2f}$, T = {T:.2f}, {len(lamb_v)} linearly spaced $\lambda$', fontsize=25)
    plt.ylim(bottom=min(-DeltaU_mean/N))
    plt.fill_between(lamb_v, -DeltaU_mean/N, 0, color='lightblue', alpha=0.6, label=r'F_{Einstein} - F_{SAAP}')    
    plt.legend(fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

