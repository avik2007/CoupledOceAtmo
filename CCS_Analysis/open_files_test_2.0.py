import numpy as np
#import pylab as plt
import sys 
sys.path.append('/nobackup/htorresg/DopplerScat/modelling/GS/programs/tools/')
sys.path.append('/u/htorresg/Experiment_CCS/programas/tools')
import handle_mitgcm as model
#import tools as io
#import IO as io
#import Fronto_analysis_v2 as fronto
import grid_to_fronto
from matplotlib.colors import LogNorm
import params_500mSNAPS as p
#import matplotlib as mpl
#mpl.rcParams.update({'figure.autolayout':True})
#mpl.rcParams['pdf.fonttype']=42
#mpl.rcParams['ps.fonttype']=42
#mpl.rcParams['font.family']='sans'
#mpl.rc('xtick',labelsize=11)
#mpl.rc('ytick',labelsize=11)
#mpl.rcParams['axes.linewidth']=1.4


def Ri_estimation(dudz,dvdz,dbdx,dbdy,N2,f):
    Grad_b = np.sqrt(dbdx**2 + dbdy**2)
    Ri_twb = ((f**2)*N2)/(Grad_b**2)

    Ri = N2/(dudz**2 + dvdz**2)
    return Ri_twb,Ri



def nu(Ri):
    nu_o = 100*1e-4 # m2/s
    nu_b = 5*1e-4 # m2/s
    alpha = 10
    kappa_b = 10*1e-4
    n = 2
    ### viscosity
    nu = nu_o/(1 + alpha*Ri)**n + nu_b
    ### diffusivity
    kappa = nu/(1 + alpha*Ri) + kappa_b
    return nu,kappa
    

def main(grid,maxlevel,c):

    print('hello')
    I=0
    import params_500mSNAPS as p

    i=c.timesteps[0]
    print(i)
    """
    KPPhbl=c.load_2d_data(p.dirc+'KPPhbl.%010i.data'%i,'tracer')
    KPPhbl=KPPhbl[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    KPPhbl = np.expand_dims(KPPhbl, 0)
    print(KPPhbl.shape)
        ## vertical diffusion coefficient
    KPPdifT = c.loadding_3D_data(p.dirc+'KPPdiffT.%010i.data'%i,maxlevel,'tracer')
    KPPdifT = KPPdifT[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    KPPdifT = np.expand_dims(KPPdifT, 0)
    
    u = c.loadding_3D_data(p.dirc+'UVEL.%010i.data'%i,maxlevel,'uvel')
    u = u[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    u = np.expand_dims(u,0)
    v = c.loadding_3D_data(p.dirc+'VVEL.%010i.data'%i,maxlevel,'vvel')
    v = v[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    v = np.expand_dims(v,0)
    w = c.loadding_3D_data(p.dirc+'WVEL.%010i.data'%i,maxlevel,'tracer')
    w = w[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    w = np.expand_dims(w,0)
    
    v = c.loadding_3D_data(p.dirc+'VVEL.%010i.data'%i,maxlevel,'vvel')
    v = v[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    v = np.expand_dims(v,0)
    
    t = c.loadding_3D_data(p.dirc+'THETA.%010i.data'%i,maxlevel,'tracer')
    t = t[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    t = np.expand_dims(t,0)

    tx = c.load_2d_data(p.dirc+'oceTAUX.%010i.data'%i,'uvel')
    tx = tx[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    tx = np.expand_dims(tx,0)
    ty = c.load_2d_data(p.dirc+'oceTAUY.%010i.data'%i,'vvel')
    ty = ty[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    ty = np.expand_dims(ty,0)
    
    KPPviscA = c.loadding_3D_data(p.dirc+'KPPviscA.%010i.data'%i,maxlevel,'tracer')
    KPPviscA = KPPviscA[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    KPPviscA= np.expand_dims(KPPviscA, 0)
    """
    ##Qnet = c.load_2d_data(p.dirc+'oceQnet.%010i.data'%i,'tracer')
    #Qnet=Qnet[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    #Qnet = np.expand_dims(Qnet,0)
    t = c.loadding_3D_data(p.dirc+'T.%010i.data'%i,maxlevel,'tracer')
    t = t[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    t = np.expand_dims(t,0)
    s = c.loadding_3D_data(p.dirc+'S.%010i.data'%i,maxlevel,'tracer')
    s = s[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    s = np.expand_dims(s,0)
    u = c.loadding_3D_data(p.dirc+'U.%010i.data'%i,maxlevel,'uvel')
    u = u[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    u = np.expand_dims(u,0)
    v = c.loadding_3D_data(p.dirc+'V.%010i.data'%i,maxlevel,'vvel')
    v = v[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
    v = np.expand_dims(v,0)
    #for now, I only want 1 day's worth of time
    final_time = 24
    for i in c.timesteps[1:final_time]:
        print(i)
        """
        t = c.loadding_3D_data(p.dirc+'THETA.%010i.data'%i,maxlevel,'tracer')
        t = t[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        s = c.loadding_3D_data(p.dirc+'SALT.%010i.data'%i,maxlevel,'tracer')
        s = s[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        u = c.loadding_3D_data(p.dirc+'UVEL.%010i.data'%i,maxlevel,'uvel')
        u = u[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        v = c.loadding_3D_data(p.dirc+'VVEL.%010i.data'%i,maxlevel,'vvel')
        v = v[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        w = c.loadding_3D_data(p.dirc+'WVEL.%010i.data'%i,maxlevel,'tracer')
        w = w[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
	tx = c.load_2d_data(p.dirc+'oceTAUX.%010i.data'%i,'uvel')
        tx = tx[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        ty = c.load_2d_data(p.dirc+'oceTAUY.%010i.data'%i,'vvel')
        ty = ty[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        Qnet = c.load_2d_data(p.dirc+'oceQnet.%010i.data'%i,'tracer')
        Qnet=Qnet[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]

        ## KPP vertical eddy viscosity coefficient
        KPPv=c.loadding_3D_data(p.dirc+'KPPviscA.%010i.data'%i,maxlevel,'tracer')
    KPPv=KPPv[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
"""
        #KPPhbl_add=c.load_2d_data(p.dirc+'KPPhbl.%010i.data'%i,'tracer')
        #KPPhbl_add=KPPhbl_add[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #KPPhbl_add = np.expand_dims(KPPhbl_add, 0)
        #KPPhbl = np.concatenate((KPPhbl, KPPhbl_add))

        ## vertical diffusion coefficient
        #KPPdifT_add = c.loadding_3D_data(p.dirc+'KPPdiffT.%010i.data'%i,maxlevel,'tracer')
        #KPPdifT_add=KPPdifT_add[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #KPPdifT_add = np.expand_dims(KPPdifT_add, 0)
        #KPPdifT = np.concatenate((KPPdifT, KPPdifT_add))

        
        #wa = c.loadding_3D_data(p.dirc+'WVEL.%010i.data'%i,maxlevel,'tracer')
       # wa = wa[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #wa = np.expand_dims(wa,0)
        #w = np.concatenate((w,wa))

        #ua = c.loadding_3D_data(p.dirc+'UVEL.%010i.data'%i,maxlevel,'uvel')
        #ua = ua[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #ua = np.expand_dims(ua,0)
        #u = np.concatenate((u,ua))
        #va = c.loadding_3D_data(p.dirc+'VVEL.%010i.data'%i,maxlevel,'vvel')
        #va = va[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #va = np.expand_dims(va,0)
        #v = np.concatenate((v,va))
        """
        ta = c.loadding_3D_data(p.dirc+'THETA.%010i.data'%i,maxlevel,'tracer')
        ta = ta[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        ta = np.expand_dims(ta,0)
        t = np.concatenate((t,ta))
        txa = c.load_2d_data(p.dirc+'oceTAUX.%010i.data'%i,'uvel')
        txa = txa[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        txa = np.expand_dims(txa,0)
        tx = np.concatenate((tx,txa))
        tya = c.load_2d_data(p.dirc+'oceTAUY.%010i.data'%i,'vvel')
        tya = tya[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        tya = np.expand_dims(tya,0)
        ty = np.concatenate((ty,tya))
        
        KPPviscAa = c.loadding_3D_data(p.dirc+'KPPviscA.%010i.data'%i,maxlevel,'tracer')
        KPPviscAa = KPPviscAa[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        print(grid['ilnc'])
        KPPviscAa = np.expand_dims(KPPviscAa, 0)
        KPPviscA = np.concatenate((KPPviscA, KPPviscAa)) 
        """
        #Qneta = c.load_2d_data(p.dirc+'oceQnet.%010i.data'%i,'tracer')
        #Qneta=Qneta[grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        #Qneta = np.expand_dims(Qneta,0)
        #Qnet = np.concatenate((Qnet, Qneta))
        sa = c.loadding_3D_data(p.dirc+'S.%010i.data'%i,maxlevel,'tracer')
        sa = sa[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        sa = np.expand_dims(sa,0)
        s = np.concatenate((s,sa))
        ua = c.loadding_3D_data(p.dirc+'U.%010i.data'%i,maxlevel,'uvel')
        ua = ua[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        ua = np.expand_dims(ua,0)
        u = np.concatenate((u,ua))
        va = c.loadding_3D_data(p.dirc+'V.%010i.data'%i,maxlevel,'vvel')
        va = va[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        va = np.expand_dims(va,0)
        v = np.concatenate((v,va))
        ta = c.loadding_3D_data(p.dirc+'T.%010i.data'%i,maxlevel,'tracer')
        ta = ta[:,grid['ilnc'][0]:grid['ilnc'][1],grid['iltc'][0]:grid['iltc'][1]]
        ta = np.expand_dims(ta,0)
        t = np.concatenate((t,ta))
    np.save("S_test.npy", s)
    #np.save("Qnet_test.npy", Qnet)
    #np.save("KPPhbl_test.npy", KPPhbl)
    #np.save("KPPdiffT_test.npy", KPPdifT)
    np.save("V_test.npy",v)
    np.save("U_test.npy",u)
    #np.save("W_test.npy",w)
    np.save("T_test.npy", t)
    #np.save("Taux_test.npy", tx)
    #np.save("Tauy_test.npy", ty)
    #np.save("KPPviscA_test.npy", KPPviscA)
    
    """
        ### vertical mean profile of T ##
        Tmn = np.nanmean(t,axis=(1,2))
        Wmn = np.nanmean(w,axis=(1,2))

        

        ##########################
        ##
        ##   Frontogenesis
        ##
        ##########################
        data = fronto.fronto(c,s,t-Tmn[:,None,None],
                             u,v,w-Wmn[:,None,None],tx,ty,
                             Qnet,KPPv,KPPdifT,KPPhbl,grid,maxlevel)

        print(data)
        KPPv = data.KppviscA
        KdiffT = data.KdiffT
        #plt.imshow(KPPv[10,:,:],origin='lower')
        #plt.colorbar()
        #plt.show()
        
        ######## Richardson number ########
        Ri = data.N2/(data.dudz**2 + data.dvdz**2)


        Ri_twb,Ri_tot = Ri_estimation(data.dudz,
                                  data.dvdz,data.bx,data.by,data.N2,
                                  data.fo)



        ######## Eddy viscosity #########
        kappa_twb,dif_twb = nu(Ri_twb) 
        kappa_tot,dif_tot = nu(Ri_tot)

        print('Min kapp: ',np.nanmin(KPPv[14,:,:]))
        print('Max kapp: ',np.nanmax(KPPv[14,:,:]))


        ########
        layer = 6
        z = np.round(data.depthmd[layer,1,1])


        ###
        #prntout = '/nobackup/htorresg/DopplerScat/figures/'

        #plt.figure(figsize=(12,6.5))
        #plt.subplots_adjust(wspace=0.3)
        #plt.subplot(121)
        #plt.imshow(1/Ri_twb[6,:,:],vmin=1e-1,vmax=1e0,
        #           origin='lower',cmap='Blues_r',norm=LogNorm())
        #plt.colorbar(shrink=0.4)
        #plt.title(r'z = '+str(z)+'m, '+r'$Ri^{-1}_{TWB}=(|\nabla{b}|/fN)^{2}$',size=13)
        #plt.subplot(122)
        #plt.imshow(1/Ri[6,:,:],vmin=1e-1,vmax=1e0,norm=LogNorm(),
        #           origin='lower',cmap='Blues_r')
        #plt.colorbar(shrink=0.4)
        #plt.title(r'$Ri^{-1}_{tot}=(u_{z}^{2}+v_{2}^{2})/N^{2}$',size=13)
        #plt.savefig(prntout+'Richardson.png',dpi=450,format='png',
        #            bbox_inches='tight')


        #plt.figure(figsize=(12,6.5))
        #plt.subplots_adjust(wspace=0.3)
        #plt.subplot(121)
        #plt.imshow(kappa_tot[7,:,:],vmin=1e-4,vmax=2e-3,cmap='jet',
        #           norm=LogNorm(),origin='lower')
        #plt.title('z = '+str(z)+'m, '+r'$\kappa$ from Pacanowski')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')
        #plt.subplot(122)
        #plt.imshow(KPPv[6,:,:],vmin=1e-3,vmax=1e-1,norm=LogNorm(),
        #           cmap='jet',origin='lower')
        #plt.title(r'$\kappa$ from KPP')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')
        #plt.savefig(prntout+'Kappa.png',dpi=450,format='png',
        #            bbox_inches='tight')



        #plt.figure(figsize=(12,6.5))
        #plt.subplots_adjust(wspace=0.3)
        #plt.subplot(121)
        #plt.imshow(dif_tot[7,:,:],vmin=5e-5,vmax=3e-3,cmap='jet',
        #           norm=LogNorm(),origin='lower')
        #plt.title('z = '+str(z)+'m, '+r'$\nu$ from Pacanowski')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')
        #plt.subplot(122)
        #plt.imshow(KdiffT[6,:,:],vmin=1e-3,vmax=1e-1,norm=LogNorm(),
        #           cmap='jet',origin='lower')
        #plt.title(r'$\nu$ from KPP')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')
        #plt.savefig(prntout+'nu.png',dpi=450,format='png',
        #            bbox_inches='tight')


        #plt.figure(figsize=(11,5))
        #plt.subplot(121)
        #plt.imshow(KdiffT[layer,:,:],vmin=1e-3,vmax=1e-1,norm=LogNorm(),
        #           cmap='jet',origin='lower')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')
        #plt.title('Vertical diffusion coefficient',size=13)
        #plt.subplot(122)
        #plt.imshow(KPPv[layer,:,:],vmin=1e-3,vmax=1e-1,norm=LogNorm(),
        #           cmap='jet',origin='lower')
        #plt.colorbar(shrink=0.4).set_label(r'm$^{2}$/s')        
        #plt.title('Vertical eddy viscosity coefficient',size=13)
        #plt.savefig(prntout+'nu_and_kappa.png',dpi=450,format='png',
        #            bbox_inches='tight')


        #plt.figure(figsize=(10,4))
        #plt.subplot(121)
        #plt.plot(1/Ri_twb[layer,:,:],1/kappa_twb[layer,:,:],'.k')
        #plt.plot(1/Ri_tot[layer,:,:],1/kappa_tot[layer,:,:],'.b')
        #plt.legend()
        #plt.subplot(122)
        #plt.plot(1/Ri_twb[layer,:,:],1/KPPv[layer,:,:],'.k')
        #plt.plot(1/Ri_tot[layer,:,:],1/KPPv[layer,:,:],'.b')
        #plt.legend()
        #plt.figure()
        #plt.plot(KPPv[10,:,:],kappa[10,:,:],'.k')


        plt.figure(figsize=(13.5,4))
        plt.subplots_adjust(wspace=0.5)

        plt.subplot(131)
        plt.pcolormesh(grid['lonc'][...],grid['latc'][...],
                       4.2e6*data.w[layer,:,:]*data.t[layer,:,:],
                   vmin=-1000,vmax=1000,cmap='bwr')
        plt.colorbar(shrink=0.4)   
        plt.title(r'$\rho{C_{p}}W^{\prime}T^{\prime}$')
        plt.suptitle(c.date[I])
        plt.subplot(132)
        plt.pcolormesh(grid['lonc'][...],grid['latc'][...],
                       4.2e6*KdiffT[layer,:,:]*data.dTdz[layer,:,:],
                   vmin=-1000,vmax=1000,cmap='bwr')
        plt.colorbar(shrink=0.4)
        plt.title(r'$\rho{C_{p}}\kappa\frac{\partial{{T^{\prime}}}}{\partial{z}}$')
        plt.subplot(133)
        plt.pcolormesh(grid['lonc'][...],grid['latc'][...],
                       4.2e6*dif_tot[layer,:,:]*data.dTdz[layer,:,:],
                   vmin=-100,vmax=100,cmap='bwr')
        plt.colorbar(shrink=0.4)
        plt.title(r'$\rho{C_{p}}\kappa_{pwk}\frac{\partial{{T^{\prime}}}}{\partial{z}}$')
        prntout = '/nobackup/htorresg/DopplerScat/figures/seq/'
        nmout='VHF_VerAdv_VerDif_layer'+str(layer)+'_%010i'%I+'.png'
        plt.savefig(prntout+nmout,dpi=500,format='png',bbox_inches='tight')


        plt.clf()
        
        I += 1
    """

if __name__=='__main__':
   
    import params_500mSNAPS as p

    maxlevel=30

    ## subregion
    iln = [-127.6,-122.5]
    ilt = [36,39]


    ### handler
    ## model config
    c = model.LLChires(p.dirc,p.dirc,
                      p.nx,p.ny,p.nz,p.tini,p.tref,p.tend,
	              p.steps,p.rate,p.timedelta)

    ##
    print(p.tini)
    print(p.tref)
    print(p.tend)
    grid = grid_to_fronto.prepare(c,maxlevel,iln,ilt)
    print(grid.keys())



    main(grid,maxlevel,c)
    
