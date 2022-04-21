import gc
import sys
import gsw
import math
import warnings
import global_vars as glb
import numpy as np
import numpy.matlib
import fld_tools as ft
import spec_flux as sf

class FldsU:
    def __init__(self):
        self.fnu = FldsNU()
        self.dt = 1
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nt = 0
        self.U = []
        self.Ubp = []
        self.Uhp = []
        self.V = []
        self.Vbp = []
        self.Vhp = []
        self.W = []
        self.Wbp = []
        self.Whp = []
        self.S = []
        self.Sbp = []
        self.Shp = []
        self.T = []
        self.Tbp = []
        self.Thp = []
        self.N2 = []
        self.N2bp = []
        self.N2hp = []
        self.rho = []
        self.rhobp = []
        self.rhohp = []
        self.pot_rho_down = []
        self.pot_rho_downbp = []
        self.pot_rho_downhp = []
        self.thknss = []
        self.Eta = []
        self.Etabp = []
        self.Etahp = []
        self.Pbot = []
        self.Pbotbp = []
        self.Pbothp = []
        self.Aew = []
        self.Ans = []
        self.depth = []
        self.wkbj = False
        self.trim = False

        warnings.filterwarnings("ignore", message="divide by zero") 
        warnings.filterwarnings("ignore", message="invalid value encountered") 

    def load_nu_decomp_mitgcm(self,decomp_dir,dataroot,auxroot,iters,dims,full_interp=False,zoversamp=1,pre_taper=np.array([0,0,0])):
        (U_in,V_in,ugrid_in,vgrid_in) = self.fnu.load_mitgcm_data(dataroot,auxroot,iters,dims);
        self.fnu.load_decomp_data(decomp_dir);
        self.interpolate_decomp_from_fld_nu(U_in,V_in,ugrid_in,vgrid_in,full_interp,zoversamp,pre_taper)

    def load_nu_mitgcm(self,dataroot,auxroot,iters,dims,full_interp=False,zoversamp=1,pre_taper=np.array([0,0,0])):
        self.fnu.load_mitgcm_data(dataroot,auxroot,iters,dims)
        self.interpolate_from_fld_nu(full_interp,zoversamp,pre_taper)

    def load_nu_data(self,fdir,fnames,full_interp=False,zoversamp=1,pre_taper=np.array([0,0,0])):
        self.fnu.load_data(fdir,fnames);
        self.interpolate_from_fld_nu(full_interp,zoversamp,pre_taper);

    def interpolate_from_fld_nu(self, full_interp=False,zoversamp=1,pre_taper=np.array([0,0,0])):
        (uthknss,vthknss) = ft.get_uv_f(self.fnu.thknss)
        # uthknss = cat(1,uthknss, uthknss(1,:,:,:)); vthknss = cat(2,vthknss,vthknss(:,1,:,:));

        if (full_interp):
            (self.T,_,_,_) = ft.get_4D_uniform_field(self.fnu.T,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.S,_,_,_) = ft.get_4D_uniform_field(self.fnu.S,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.U,_,_,_) = ft.get_4D_uniform_field(self.fnu.U,ft.ang_to_dist(self.fnu.ugrid),uthknss)
            (self.V,self.dx,self.dy,self.dz) = ft.get_4D_uniform_field(self.fnu.V,ft.ang_to_dist(self.fnu.vgrid),vthknss)
            self.nz = self.fnu.nz
        else:
            nzu = round(zoversamp*self.fnu.nz)
            (self.T,_) = ft.get_4D_vert_uniform_field(self.fnu.T,self.fnu.thknss,nzu)
            (self.S,_) = ft.get_4D_vert_uniform_field(self.fnu.S,self.fnu.thknss,nzu)
            (self.U,_) = ft.get_4D_vert_uniform_field(self.fnu.U,uthknss,nzu,ns=True)
            (self.V,self.dz) = ft.get_4D_vert_uniform_field(self.fnu.V,vthknss,nzu,ns=True)
            self.dx = np.sum(self.fnu.dxu[:,0])/self.fnu.nx; self.dy = np.sum(self.fnu.dyv[0,:])/self.fnu.ny;
            self.nz = nzu;

        self.depth = self.fnu.depth;
        self.nx = self.fnu.nx; self.ny = self.fnu.ny; self.nt = self.fnu.nt;
            
        # recompute W:
        self.W = ft.get_uniform_vertical_vel(self.U,self.V,self.dx,self.dy,self.dz)
            
        # apply Taper filter
        if any(pre_taper):
            (self.U,self.V,self.W) = ft.taper_filter_3D_uvw_u(self.U,self.V,self.W,pre_taper)
            
        # apply bottom mask
        self.thknss = ft.apply_bathy_mitgcm(self.dz*np.ones([self.nx,self.ny,self.nz,self.nt]),self.depth,0.3)
        (self.Aew,self.Ans) = ft.get_face_areas(self.thknss,self.dx,self.dy)
        self.U[self.Aew==0]=0; self.V[self.Ans==0]=0; self.W[self.thknss==0]=0
            
        # recompute W
        self.W = ft.get_uniform_vertical_vel(self.U,self.V,self.dx,self.dy,self.dz)
        self.W[:,:,0,:] = 0*self.W[:,:,0,:]
        
        # test divergence
        tmp = ft.get_divergence(self.U,self.V,self.W,self.dx,self.dy,self.dz);
        tmp = np.sum(tmp*tmp)/np.sum(self.U[0:-1,:,:,:]**2 + self.V[:,0:-1,:,:]**2 + self.W**2);
        print('Divergence after Interpolate:' + str(tmp))

    def interpolate_decomp_from_fld_nu(self, U_in,V_in,ugrid_in,vgrid_in,full_interp=False,zoversamp=1,pre_taper=np.array([0,0,0])):
        (uthknss,vthknss) = ft.get_uv_f(self.fnu.thknss)
        # uthknss = cat(1,uthknss, uthknss(1,:,:,:)); vthknss = cat(2,vthknss,vthknss(:,1,:,:));
        
        if (full_interp):
            (self.T,_,_,_) = ft.get_4D_uniform_field(self.fnu.T,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.S,_,_,_) = ft.get_4D_uniform_field(self.fnu.S,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.U,_,_,_) = ft.get_4D_uniform_field(self.fnu.U,ft.ang_to_dist(self.fnu.ugrid),uthknss)
            (self.V,_,_,_) = ft.get_4D_uniform_field(self.fnu.V,ft.ang_to_dist(self.fnu.vgrid),vthknss)
            (U_in,_,_,_) = ft.get_4D_uniform_field(U_in,ft.ang_to_dist(ugrid_in),uthknss)
            (V_in,_,_,_) = ft.get_4D_uniform_field(V_in,ft.ang_to_dist(vgrid_in),vthknss)
#             (self.Tbp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Tbp,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
#             (self.Sbp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Sbp,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.Ubp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Ubp,ft.ang_to_dist(self.fnu.ugrid),uthknss)
            (self.Vbp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Vbp,ft.ang_to_dist(self.fnu.vgrid),vthknss)
#             (self.Thp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Thp,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
#             (self.Shp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Shp,ft.ang_to_dist(self.fnu.pgrid),self.fnu.thknss)
            (self.Uhp,_,_,_) = ft.get_4D_uniform_field(self.fnu.Uhp,ft.ang_to_dist(self.fnu.ugrid),uthknss)
            (self.Vhp,self.dx,self.dy,self.dz) = ft.get_4D_uniform_field(self.fnu.Vhp,ft.ang_to_dist(self.fnu.vgrid),vthknss)
            self.nz = self.fnu.nz
        else:
            nzu = round(zoversamp*self.fnu.nz)
            (self.T,_) = ft.get_4D_vert_uniform_field(self.fnu.T,self.fnu.thknss,nzu)
            (self.S,_) = ft.get_4D_vert_uniform_field(self.fnu.S,self.fnu.thknss,nzu)
#             (self.U,_) = ft.get_4D_vert_uniform_field(self.fnu.U,uthknss[:-1,:,:,:],nzu,ns=True)
#             (self.V,_) = ft.get_4D_vert_uniform_field(self.fnu.V,vthknss[:,:-1,:,:],nzu,ns=True)
            (U_in,_) = ft.get_4D_vert_uniform_field(U_in,uthknss,nzu,ns=True); self.U = U_in[:-1,:,:,:];
            (V_in,_) = ft.get_4D_vert_uniform_field(V_in,vthknss,nzu,ns=True); self.V = V_in[:,:-1,:,:];
#             (self.Tbp,_) = ft.get_4D_vert_uniform_field(self.fnu.Tbp,self.fnu.thknss,nzu)
#             (self.Sbp,_) = ft.get_4D_vert_uniform_field(self.fnu.Sbp,self.fnu.thknss,nzu)
            (self.Ubp,_) = ft.get_4D_vert_uniform_field(self.fnu.Ubp,uthknss[:-1,:,:,:],nzu,ns=True)
            (self.Vbp,_) = ft.get_4D_vert_uniform_field(self.fnu.Vbp,vthknss[:,:-1,:,:],nzu,ns=True)
#             (self.Thp,_) = ft.get_4D_vert_uniform_field(self.fnu.Thp,self.fnu.thknss,nzu)
#             (self.Shp,_) = ft.get_4D_vert_uniform_field(self.fnu.Shp,self.fnu.thknss,nzu)
            (self.Uhp,_) = ft.get_4D_vert_uniform_field(self.fnu.Uhp,uthknss[:-1,:,:,:],nzu,ns=True)
            (self.Vhp,self.dz) = ft.get_4D_vert_uniform_field(self.fnu.Vhp,vthknss[:,:-1,:,:],nzu,ns=True)
            self.dx = np.sum(self.fnu.dxu[:,0])/self.fnu.nx; self.dy = np.sum(self.fnu.dyv[0,:])/self.fnu.ny;
            self.nz = nzu;

        self.depth = self.fnu.depth;
        self.nx = self.fnu.nx; self.ny = self.fnu.ny; self.nt = self.fnu.nt;

        # recompute W:
        self.W = ft.get_uniform_vertical_vel(U_in,V_in,self.dx,self.dy,self.dz)
#         self.Whp = ft.get_uniform_vertical_vel_mod_grid(self.Uhp,self.Vhp,self.dx,self.dy,self.dz)
#         self.Wbp = ft.get_uniform_vertical_vel_mod_grid(self.Ubp,self.Vbp,self.dx,self.dy,self.dz)
            
        # apply Taper filter
        if any(pre_taper):
            (self.U,self.V,self.W) = ft.taper_filter_3D_uvw_u(self.U,self.V,self.W,pre_taper)
            (U_in,V_in,_) = ft.taper_filter_3D_uvw_u(U_in,V_in,self.W,pre_taper)
            (self.Uhp,self.Vhp,_) = ft.taper_filter_3D_uvw_u(self.Uhp,self.Vhp,self.W,pre_taper)
            (self.Ubp,self.Vbp,_) = ft.taper_filter_3D_uvw_u(self.Ubp,self.Vbp,self.W,pre_taper)
            
        # apply bottom mask
        self.thknss = ft.apply_bathy_mitgcm(self.dz*np.ones([self.nx,self.ny,self.nz,self.nt]),self.depth,0.3)
        (Aew,Ans) = ft.get_face_areas(self.thknss,self.dx,self.dy)
        (self.Aew,self.Ans) = ft.get_face_areas_mod_grid(self.thknss,self.dx,self.dy)
        self.U[self.Aew==0]=0; self.V[self.Ans==0]=0; self.W[self.thknss==0]=0
        U_in[Aew==0]=0; V_in[Ans==0]=0; 
        self.Uhp[self.Aew==0]=0; self.Vhp[self.Ans==0]=0; # self.Whp[self.thknss==0]=0
        self.Ubp[self.Aew==0]=0; self.Vbp[self.Ans==0]=0; # self.Wbp[self.thknss==0]=0
            
        # recompute W
        self.W = ft.get_uniform_vertical_vel(U_in,V_in,self.dx,self.dy,self.dz)
#         self.Whp = ft.get_uniform_vertical_vel_mod_grid(self.Uhp,self.Vhp,self.dx,self.dy,self.dz)
#         self.Wbp = ft.get_uniform_vertical_vel_mod_grid(self.Ubp,self.Vbp,self.dx,self.dy,self.dz)
        self.W[:,:,0,:] = 0*self.W[:,:,0,:]
#         self.Whp[:,:,0,:] = 0*self.Whp[:,:,0,:]
#         self.Wbp[:,:,0,:] = 0*self.Wbp[:,:,0,:]

        # test divergence
        tmp = ft.get_divergence(self.U,self.V,self.W,self.dx,self.dy,self.dz);
        tmp = np.sum(tmp*tmp)/np.sum(self.U**2 + self.V**2 + self.W**2);
        print('Divergence after Interpolate:' + str(tmp))

        # tmp = ft.get_divergence(self.Uhp,self.Vhp,self.Whp,self.dx,self.dy,self.dz);
        # tmp = np.sum(tmp*tmp)/np.sum(self.Uhp**2 + self.Vhp**2 + self.Whp**2);
        # print('HP Divergence after Interpolate:' + str(tmp))

        # tmp = ft.get_divergence(self.Ubp,self.Vbp,self.Wbp,self.dx,self.dy,self.dz);
        # tmp = np.sum(tmp*tmp)/np.sum(self.Ubp**2 + self.Vbp**2 + self.Wbp**2);
        # print('BP Divergence after Interpolate:' + str(tmp))

    def compute_rho_mitgcm(self):
        # get dims
        (nx,ny,nz,nt) = self.T.shape;
                
        rho0 = 1035; g = 9.81;
        p = self.dz*np.linspace(0.5,nz-0.5,nz)*g*rho0/10000 + 10;
        p = np.reshape(p,(1,1,nz,1));
        p = ft.repmat(p,(nx,ny,1,nt));
        SA = np.zeros([nx,ny,nz,nt]); CT = np.zeros([nx,ny,nz,nt]); self.rho = np.zeros([nx,ny,nz,nt]);
        for k in range(0,nz):
            for t in range(0,nt):
                SA[:,:,k,t] = np.maximum(gsw.SA_from_SP(self.S[:,:,k,t],p[:,:,k,t],
                                                    np.mean(np.mean(self.fnu.pgrid[:,:,0],axis=0,keepdims=True),axis=1,keepdims=True),
                                                    np.mean(np.mean(self.fnu.pgrid[:,:,1],axis=0,keepdims=True),axis=1,keepdims=True)),0);
                CT[:,:,k,t] = gsw.CT_from_t(SA[:,:,k,t],self.T[:,:,k,t],p[:,:,k,t]);
                self.rho[:,:,k,t] = gsw.rho(SA[:,:,k,t],CT[:,:,k,t],p[:,:,k,t]);

    def compute_FF_T_diag(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (dU,dV) = sf.compute_u_FF_T_diag(self.U,self.V,self.W,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (dU,dV);

    def compute_hp_FF_T_diag(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (dU,dV) = sf.compute_u_hp_FF_T_diag(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (dU,dV);

    def compute_decomp_FF_T_diag(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh) = sf.compute_u_decomp_FF_T_diag(self.U,self.V,self.W,
                                                                                                            self.Ubp,self.Vbp,
                                                                                                            self.Uhp,self.Vhp,
                                                                                                            self.dx,self.dy,self.dz,
                                                                                                            post_taper,trim_ml);
        return (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh);

    def compute_FF_T_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,m) = sf.compute_u_FF_T_m(self.U,self.V,self.W,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,m);

    def compute_hp_FF_T_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,m) = sf.compute_u_hp_FF_T_m(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,m);

    def compute_decomp_FF_T_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,T_KEdivV,m) = sf.compute_u_decomp_FF_T_m(self.U,self.V,self.W,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,T_KEdivV,m);

    def compute_KE_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,m) = sf.compute_u_KE_m(self.U,self.V,self.W,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,m);

    def compute_hp_KE_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,m) = sf.compute_u_hp_KE_m(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,m);

    def compute_decomp_KE_m(self,post_taper = np.array([1,1,0]),trim_ml = 0):
            
        (T,m) = sf.compute_u_decomp_KE_m(self.U,self.V,self.W,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        return (T,m);

    def compute_FF_F_m(self,post_taper = np.array([1,1,0]),trim_ml = 0,s_ave=False):
            
        (T,m) = sf.compute_u_FF_F_m(self.U,self.V,self.W,self.dx,self.dy,self.dz,post_taper,trim_ml);
        if s_ave:
            T = np.squeeze(np.sum(np.sum(T,axis=2),axis=1))/self.nx/self.ny;
        return (T,m);

    def compute_hp_FF_F_m(self,post_taper = np.array([1,1,0]),trim_ml = 0,s_ave=False):
            
        (T,m) = sf.compute_u_hp_FF_F_m(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        if s_ave:
            T = np.squeeze(np.sum(np.sum(T,axis=2),axis=1))/self.nx/self.ny;
        return (T,m);

    def compute_decomp_FF_F_m(self,post_taper = np.array([1,1,0]),trim_ml = 0,s_ave=False):
            
        (F,m) = sf.compute_u_decomp_FF_F_m(self.U,self.V,self.W,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.dx,self.dy,self.dz,post_taper,trim_ml);
        if s_ave:
            F = np.squeeze(np.sum(np.sum(F,axis=2),axis=1))/self.nx/self.ny;
        return (F,m);



class FldsNU:
    def __init__(self):
        self.dt = 1
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nt = 0
        self.U = []
        self.Ubp = []
        self.Uhp = []
        self.V = []
        self.Vbp = []
        self.Vhp = []
        self.W = []
        self.Wbp = []
        self.Whp = []
        self.S = []
        self.Sbp = []
        self.Shp = []
        self.T = []
        self.Tbp = []
        self.Thp = []
        self.N2 = []
        self.N2bp = []
        self.N2hp = []
        self.rho = []
        self.rhobp = []
        self.rhohp = []
        self.pot_rho_down = []
        self.pot_rho_downbp = []
        self.pot_rho_downhp = []
        self.Eta = []
        self.Etabp = []
        self.Etahp = []
        self.Pbot = []
        self.Pbotbp = []
        self.Pbothp = []
        self.thknss = []
        self.depth = []
        self.ugrid = []
        self.vgrid = []
        self.pgrid = []
        self.qgrid = []
        self.dxc = []  # assumed periodic
        self.dyc = []  # assumed periodic
        self.dxu = [] 
        self.dyu = []  # assumed periodic
        self.dxv = []  # assumed periodic
        self.dyv = []
        self.dxq = []
        self.dyq = []
        self.wkbj = False
        self.trim = False

        warnings.filterwarnings("ignore", message="divide by zero") 
        warnings.filterwarnings("ignore", message="invalid value encountered") 

    def load_decomp_data(self,decomp_dir):
        # init fields
        self.Uhp = ft.read_field(decomp_dir+'_0');
        print('max Uhp = ' + str(np.max(self.Uhp)));
        print('min Uhp = ' + str(np.min(self.Uhp)));
        self.Vhp = ft.read_field(decomp_dir+'_1');
        print('max Vhp = ' + str(np.max(self.Vhp)));
        print('min Vhp = ' + str(np.min(self.Vhp)));
        # self.Whp = ft.read_field(decomp_dir+'_2');
        # print('max Whp = ' + str(np.max(self.Whp)));
        # print('min Whp = ' + str(np.min(self.Whp)));
        # self.Thp = ft.read_field(decomp_dir+'_3');
        # print('max Thp = ' + str(np.max(self.Thp)));
        # print('min Thp = ' + str(np.min(self.Thp)));
        # self.Shp = ft.read_field(decomp_dir+'_4');
        # print('max Shp = ' + str(np.max(self.Shp)));
        # print('min Shp = ' + str(np.min(self.Shp)));
        # self.Etahp = ft.read_field(decomp_dir+'_5');
        # self.Etahp = self.Etahp[:,:,0:1,:];
        # print('max Etahp = ' + str(np.max(self.Etahp)));
        # print('min Etahp = ' + str(np.min(self.Etahp)));
        self.Ubp = ft.read_field(decomp_dir+'_6');
        print('max Ubp = ' + str(np.max(self.Ubp)));
        print('min Ubp = ' + str(np.min(self.Ubp)));
        self.Vbp = ft.read_field(decomp_dir+'_7');
        print('max Vbp = ' + str(np.max(self.Vbp)));
        print('min Vbp = ' + str(np.min(self.Vbp)));
        # self.Wbp = ft.read_field(decomp_dir+'_8');
        # print('max Wbp = ' + str(np.max(self.Wbp)));
        # print('min Wbp = ' + str(np.min(self.Wbp)));
        # self.Tbp = ft.read_field(decomp_dir+'_9');
        # print('max Tbp = ' + str(np.max(self.Tbp)));
        # print('min Tbp = ' + str(np.min(self.Tbp)));
        # self.Sbp = ft.read_field(decomp_dir+'_10');
        # print('max Sbp = ' + str(np.max(self.Sbp)));
        # print('min Sbp = ' + str(np.min(self.Sbp)));
        # self.Etabp = ft.read_field(decomp_dir+'_11');
        # self.Etabp = self.Etabp[:,:,0:1,:];
        # print('max Etabp = ' + str(np.max(self.Etabp)));
        # print('min Etabp = ' + str(np.min(self.Etabp)));

    def load_mitgcm_data(self,dataroot,auxroot,iters,dims):
        # Init Fields
        dims2D = dims.copy(); dims2D[2]=1;
        try:
            self.S = ft.rdmds(dataroot + 'S',iters,dims,'S');
            self.T = ft.rdmds(dataroot + 'T', iters,dims,'T');
        except:
            self.S = ft.rdmds(dataroot + 'Salt',iters,dims,'S');
            self.T = ft.rdmds(dataroot + 'Theta', iters,dims,'T');
        self.U = ft.rdmds(dataroot + 'U', iters,dims,'U');
        self.V = ft.rdmds(dataroot + 'V', iters,dims,'V');
        self.W = ft.rdmds(dataroot + 'W', iters,dims,'W');
        try:
            self.Eta = ft.rdmds(dataroot + 'Eta',iters,dims2D,'Eta');
        except:
            warnings.warn("Eta not found, setting to zeros...");
            self.Eta = np.zeros((nx,ny,1,1));

        # Set Dims
        nx = self.W.shape[0]; ny = self.W.shape[1]; nz = self.W.shape[2]; nt = self.W.shape[3];

        try:
            self.depth = -ft.read_bin(auxroot + 'BATHY_' + str(dims[0]) + 'x' + str(dims[1]) + '_Box56', (nx,ny));
        except:
            auxroot = auxroot + '../build/';
            self.depth = -ft.read_bin(auxroot + 'BATHY_' + str(dims[0]) + 'x' + str(dims[1]) + '_Box56', (nx,ny));

        print('min depth = ' + str(np.min(self.depth)) )
        print('max depth = ' + str(np.max(self.depth)) )
        self.depth = np.reshape(self.depth,(nx,ny,1,1));
        try: 
            self.thknss = ft.read_bin(auxroot + 'delR', [nz]);    
        except:
            Z = -ft.rdmds(auxroot + 'RF',[],[nz+1],'Z');    
            self.thknss = np.diff(Z,axis=0);
        self.thknss = np.reshape(self.thknss,(1,1,nz,1));
        self.thknss = ft.repmat(self.thknss,(nx,ny,1,nt));
        self.thknss = ft.apply_bathy_mitgcm(self.thknss,self.depth);
        print('min thknss = ' + str(np.min(self.thknss)) )
        print('max thknss = ' + str(np.max(self.thknss)) )
        
        # Load Grids
        lonc = ft.read_bin(auxroot + 'LONC', (nx,ny,1));
        lng = ft.read_bin(auxroot + 'LONG', (nx,ny,1));
        latc = ft.read_bin(auxroot + 'LATC', (nx,ny,1));
        latg = ft.read_bin(auxroot + 'LATG', (nx,ny,1));
            
        self.ugrid = np.concatenate((lng,latc),axis=2);
        self.vgrid = np.concatenate((lonc,latg),axis=2);
        self.pgrid = np.concatenate((lonc,latc),axis=2);
        self.qgrid = np.concatenate((lng,latc),axis=2);

        U_out = self.U[:,0:-1,:,:].copy();
        V_out = self.V[0:-1,:,:,:].copy();
        ugrid_out = self.ugrid[:,0:-1,:].copy();
        vgrid_out = self.vgrid[0:-1,:,:].copy();

        (self.dxq,self.dyq) = ft.get_spacing_from_grid(ft.ang_to_dist(self.qgrid));
        (self.dxc,self.dyc) = ft.get_spacing_from_grid(ft.ang_to_dist(self.pgrid[0:-1,0:-1,:]));
        (self.dxu,self.dyu) = ft.get_spacing_from_grid(ft.ang_to_dist(self.ugrid[:,0:-1,:]));
        (self.dxv,self.dyv) = ft.get_spacing_from_grid(ft.ang_to_dist(self.vgrid[0:-1,:,:]));
        
        # Trim down to size
        self.ugrid = self.ugrid[0:-1,0:-1,:];
        self.U = self.U[0:-1,0:-1,:,:];
        self.vgrid = self.vgrid[0:-1,0:-1,:];
        self.V = self.V[0:-1,0:-1,:,:];
        self.pgrid = self.pgrid[0:-1,0:-1,:];
        self.qgrid = self.qgrid[0:-1,0:-1,:];
        self.S = self.S[0:-1,0:-1,:,:];
        self.T = self.T[0:-1,0:-1,:,:];
        self.W = self.W[0:-1,0:-1,:,:];
        self.Eta = self.Eta[0:-1,0:-1,:];
        self.thknss = self.thknss[0:-1,0:-1,:,:];
        self.depth = self.depth[0:-1,0:-1,:];

        self.nx = nx-1; self.ny = ny-1; self.nz = nz; self.nt = nt;
        
        self.dxc = np.concatenate((self.dxc[-1:,:],self.dxc),axis=0);
        self.dyc = np.concatenate((self.dyc[:,-1:],self.dyc),axis=1);
        self.dyu = np.concatenate((self.dyu[:,-1:],self.dyu),axis=1);
        self.dxv = np.concatenate((self.dxv[-1:,:],self.dxv),axis=0);

        self.dxq = self.dxq[:,0:-1];
        self.dyq = self.dyq[0:-1,:];
        self.dyu = self.dyu[0:-1,:];
        self.dxv = self.dxv[:,0:-1];

        return (U_out,V_out,ugrid_out,vgrid_out);

    def load_mitgcm_field(self,fldname,dataroot,iters,dims):
        f = ft.rdmds(dataroot+fldname,iters,dims);
        if (len(f.shape)==4):
            return f[0:self.nx,0:self.ny,:,:]
        elif (len(f.shape)==3):
            return f[0:self.nx,0:self.ny,:]
        else:
            return f[0:self.nx,0:self.ny]

    def load_data(self,fdir,fnames):
        # load grids
        self.ugrid = ft.read_field(fdir + 'ugrid.' + fnames[0]);
        self.vgrid = ft.read_field(fdir + 'vgrid.' + fnames[0]);
        self.pgrid = ft.read_field(fdir + 'pgrid.' + fnames[0]);
        self.qgrid = ft.read_field(fdir + 'qgrid.' + fnames[0]);

        (self.dxc,self.dyc) = ft.get_spacing_from_grid(ft.ang_to_dist(self.pgrid));
        (self.dxu,self.dyu) = ft.get_spacing_from_grid(ft.ang_to_dist(self.ugrid));
        (self.dxv,self.dyv) = ft.get_spacing_from_grid(ft.ang_to_dist(self.vgrid));
        (self.dxq,self.dyq) = ft.get_spacing_from_grid(ft.ang_to_dist(self.qgrid));
        self.dxc = np.concatenate((self.dxc[-1:,:],self.dxc),axis=0); dxq = dxq[:,:-1];
        self.dyc = np.concatenate((self.dyc[:,-1:],self.dyc),axis=1); dyq = dxq[:-1,:];
        self.dyu = np.concatenate((self.dyu[:,-1:],self.dyu),axis=1); dyu = dyu[:-1,:]; # should double check this
        self.dxv = np.concatenate((self.dxv[-1:,:],self.dxv),axis=0); dxv = dxv[:,:-1]; # should double check this
            
        # init fields
        nfiles = len(fnames);
        self.S = np.zeros(ft.read_field(fdir + 'S.' + fnames[0]).shape[0:3] + (nfiles,));
        self.T = np.zeros(ft.read_field(fdir + 'T.' + fnames[0]).shape[0:3] + (nfiles,));
        self.U = np.zeros(ft.read_field(fdir + 'U.' + fnames[0]).shape[0:3] + (nfiles,));
        self.V = np.zeros(ft.read_field(fdir + 'V.' + fnames[0]).shape[0:3] + (nfiles,));
        self.W = np.zeros(ft.read_field(fdir + 'W.' + fnames[0]).shape[0:3] + (nfiles,));
        self.thknss = np.zeros(ft.read_field(fdir + 'thknss.' + fnames[0]).shape[0:3] + (nfiles,));

        for ifile in range(0,nfiles):
            self.S[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'S.' + fnames[ifile]);
            self.T[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'T.' + fnames[ifile]);
            self.U[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'U.' + fnames[ifile]);
            self.V[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'V.' + fnames[ifile]);
            self.W[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'W.' + fnames[ifile]);
            self.thknss[:,:,:,ifile:ifile+1] = ft.read_field(fdir + 'thknss.' + fnames[ifile]);
            
        self.depth = np.sum(self.thknss[:,:,:,0],axis=2,keepdims=True);
            
        # check size for T,S
        nx = self.thknss.shape[0]; ny = self.thknss.shape[1]; nz = self.thknss.shape[2];
        self.T = self.T[0:nx,0:ny,0:nz,:];
        self.S = self.S[0:nx,0:ny,0:nz,:];
        self.nx = nx; self.ny = ny; self.nz = nz; self.nt = nfiles;
        
    def recut_data(self,cfx,cfy):
        cnx0 = max(math.ceil(cfx[0]*self.nx),1)-1;
        cnx1 = min(math.ceil(cfx[1]*self.nx),self.nx);
        cny0 = max(math.ceil(cfy[0]*self.ny),1)-1;
        cny1 = min(math.ceil(cfy[1]*self.ny),self.ny);

        f = FldsNU();

        f.ugrid = self.ugrid[cnx0:cnx1,cny0:cny1,:];
        f.vgrid = self.vgrid[cnx0:cnx1,cny0:cny1,:];
        f.pgrid = self.pgrid[cnx0:cnx1,cny0:cny1,:];
        f.qgrid = self.qgrid[cnx0:cnx1,cny0:cny1,:];

        # init fields
        f.S = self.S[cnx0:cnx1,cny0:cny1,:,:];
        f.T = self.T[cnx0:cnx1,cny0:cny1,:,:];
        f.U = self.U[cnx0:cnx1,cny0:cny1,:,:];
        f.V = self.V[cnx0:cnx1,cny0:cny1,:,:];
        f.W = self.W[cnx0:cnx1,cny0:cny1,:,:];
        f.thknss = self.thknss[cnx0:cnx1,cny0:cny1,:,:];

        f.nx = f.W.shape[0]; f.ny = f.W.shape[1]; f.nz = f.W.shape[2]; f.nt = f.W.shape[3];

        return f;

    def write_regional_field_snapshots(self,fdir,fnames):
        for t in range(0,self.nt):
            ft.write_field(self.S[:,:,:,t],fdir + 'S.' + str(fnames[t]));
            ft.write_field(self.T[:,:,:,t],fdir + 'T.' + str(fnames[t]));
            ft.write_field(self.U[:,:,:,t],fdir + 'U.' + str(fnames[t]));
            ft.write_field(self.V[:,:,:,t],fdir + 'V.' + str(fnames[t]));
            ft.write_field(self.W[:,:,:,t],fdir + 'W.' + str(fnames[t]));
            ft.write_field(self.thknss[:,:,:,t],fdir + 'thknss.' + str(fnames[t]));
            ft.write_field(self.ugrid[:,:,:],fdir + 'ugrid.' + str(fnames[t]));
            ft.write_field(self.vgrid[:,:,:],fdir + 'vgrid.' + str(fnames[t]));
            ft.write_field(self.pgrid[:,:,:],fdir + 'pgrid.' + str(fnames[t]));
            ft.write_field(self.qgrid[:,:,:],fdir + 'qgrid.' + str(fnames[t]));

    def get_dx_dy_dz(self,zoversamp=1):
        
        nzu = round(zoversamp*self.nz);
        Lx = np.sum(self.dxu[:,0]); Ly = np.sum(self.dyv[0,:]); dx = Lx/self.nx; dy = Ly/self.ny;
        depth2d = np.sum(self.thknss,axis=2,keepdims=True); zBot = np.amax(depth2d); zSurf = 0;
        dz = (zBot-zSurf)/(nzu);

        return (dx,dy,dz);

    def compute_rho_mitgcm(self):
        # get dims
        (nx,ny,nz,nt) = self.T.shape;
                
        p = (np.cumsum(self.thknss,axis=2)-0.5*self.thknss)*glb.g*glb.rho0/10000 + 10;
        SA = np.zeros([nx,ny,nz,nt]); CT = np.zeros([nx,ny,nz,nt]); self.rho = np.zeros([nx,ny,nz,nt]);
        for k in range(0,nz):
            for t in range(0,nt):
                SA[:,:,k,t] = gsw.SA_from_SP(self.S[:,:,k,t],p[:,:,k,0],np.squeeze(self.pgrid[:,:,0]),np.squeeze(self.pgrid[:,:,1]));
                CT[:,:,k,t] = gsw.CT_from_t(SA[:,:,k,t],self.T[:,:,k,t],p[:,:,k,0]);
                self.rho[:,:,k,t] = gsw.rho(SA[:,:,k,t],CT[:,:,k,t],p[:,:,k,0]);

    def compute_pot_rho_down_mitgcm(self):
        # get dims
        (nx,ny,nz,nt) = self.T.shape;
                
        p = (np.cumsum(self.thknss,axis=2)-0.5*self.thknss)*glb.g*glb.rho0/10000 + 10;
        SA = np.zeros([nx,ny,nz,nt]); CT = np.zeros([nx,ny,nz,nt]); pot_rho = np.zeros([nx,ny,nz,nt]);
        for k in range(0,nz-1):
            for t in range(0,nt):
                SA[:,:,k,t] = gsw.SA_from_SP(self.S[:,:,k,t],p[:,:,k+1,0],np.squeeze(self.pgrid[:,:,0]),np.squeeze(self.pgrid[:,:,1]));
                CT[:,:,k,t] = gsw.CT_from_t(SA[:,:,k,t],self.T[:,:,k,t],p[:,:,k+1,0]);
                pot_rho[:,:,k,t] = gsw.rho(SA[:,:,k,t],CT[:,:,k,t],p[:,:,k+1,0]);

        return pot_rho;

    def compute_rhop_mitgcm(self):
        # get dims
        (nx,ny,nz,nt) = self.T.shape;

        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
                
        p = (np.cumsum(self.thknss,axis=2)-0.5*self.thknss)*glb.g*glb.rho0/10000 + 10;
        SA = np.zeros([nx,ny,nz,nt]); CT = np.zeros([nx,ny,nz,nt]); self.rhop = np.zeros([nx,ny,nz,nt]);
        for k in range(0,nz):
            for t in range(0,nt):
                SA[:,:,k,t] = gsw.SA_from_SP(self.S[:,:,k,t]-self.Sp[:,:,k,t],p[:,:,k,0],np.squeeze(self.pgrid[:,:,0]),np.squeeze(self.pgrid[:,:,1]));
                CT[:,:,k,t] = gsw.CT_from_t(SA[:,:,k,t],self.T[:,:,k,t]-self.Tp[:,:,k,t],p[:,:,k,0]);
                self.rhop[:,:,k,t] = self.rho[:,:,k,t] - gsw.rho(SA[:,:,k,t],CT[:,:,k,t],p[:,:,k,0]);

    def compute_pe_ke_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        (T,m) = sf.compute_nu_pe_ke_m(self.U,self.V,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_decomp_pe_ke_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        # because linear, just use Uhp, Vhp.  If NL is significant, switch to Etahp, rhop (computed above)...
        # or if you want to apply over a different time period than the one the hp was computed on...
        (T,m) = sf.compute_nu_decomp_pe_ke_m(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_pe_ke_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        (dU,dV,phiHyd) = sf.compute_nu_pe_ke_diag(self.U,self.V,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV,phiHyd);

    def compute_hp_pe_ke_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        (dU,dV,phiHyd) = sf.compute_nu_pe_ke_diag(self.Uhp,self.Vhp,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV,phiHyd);

    def compute_pe_ke_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        (dU,dV) = sf.compute_nu_pe_ke_diag_u(self.Uhp,self.Vhp,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_pe_ke_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        (dU,dV) = sf.compute_nu_pe_ke_diag_u(self.U,self.V,self.Eta,self.rho,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_kpp_sep_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (dUb,dVb,dUc,dVc,dUs,dVs) = sf.compute_nu_kpp_sep_diag(self.U,self.V,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUb,dVb,dUc,dVc,dUs,dVs);

    def compute_hp_kpp_sep_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (dUb,dVb,dUc,dVc,dUs,dVs) = sf.compute_nu_hp_kpp_sep_diag(self.U,self.V,self.Uhp,self.Vhp,
                                                                  self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                                                  self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                                                  self.dxq,self.dyq,self.ugrid,
                                                                  self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUb,dVb,dUc,dVc,dUs,dVs);

    def compute_kpp_sep_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (dUb,dVb,dUc,dVc,dUs,dVs) = sf.compute_nu_kpp_sep_diag_u(self.U,self.V,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUb,dVb,dUc,dVc,dUs,dVs);

    def compute_hp_kpp_sep_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (dUb,dVb,dUc,dVc,dUs,dVs) = sf.compute_nu_hp_kpp_sep_diag_u(self.U,self.V,self.Uhp,self.Vhp,
                                                                    self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                                                    self.dxq,self.dyq,self.ugrid,
                                                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUb,dVb,dUc,dVc,dUs,dVs);

    def compute_kpp_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_kpp_sep_m(self.U,self.V,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_hp_kpp_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_hp_kpp_sep_m(self.U,self.V,self.Uhp,self.Vhp,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_decomp_kpp_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_decomp_kpp_sep_m(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_kpp_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_kpp_sep_m_xy(self.U,self.V,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_hp_kpp_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_hp_kpp_sep_m_xy(self.U,self.V,self.Uhp,self.Vhp,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_decomp_kpp_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        if (type(self.rho) is list):
            self.compute_rho_mitgcm();
        if (type(self.pot_rho_down) is list):
            self.pot_rho_down = self.compute_pot_rho_down_mitgcm();
        (Tb,Tc,Ts,m) = sf.compute_nu_decomp_kpp_sep_m_xy(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.rho,self.pot_rho_down,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (Tb,Tc,Ts,m);

    def compute_leith_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_leith_m(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_hp_leith_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_hp_leith_m(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_leith_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_leith_m_xy(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_hp_leith_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_hp_leith_m_xy(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_leith_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_leith_sep_m(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_hp_leith_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_hp_leith_sep_m(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_decomp_leith_sep_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_decomp_leith_sep_m(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_leith_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_leith_sep_m_xy(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_hp_leith_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_hp_leith_sep_m_xy(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_decomp_leith_sep_m_xy(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (TD,TZ,m) = sf.compute_nu_decomp_leith_sep_m_xy(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (TD,TZ,m);

    def compute_leith_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dU,dV,ViscA4D,ViscA4Z) = sf.compute_nu_leith_diag(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV,ViscA4D,ViscA4Z);

    def compute_hp_leith_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dU,dV,ViscA4D,ViscA4Z) = sf.compute_nu_hp_leith_diag(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV,ViscA4D,ViscA4Z);

    def compute_leith_sep_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUD,dVD,dUZ,dVZ,ViscA4D,ViscA4Z) = sf.compute_nu_leith_sep_diag(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUD,dVD,dUZ,dVZ,ViscA4D,ViscA4Z);

    def compute_hp_leith_sep_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUD,dVD,dUZ,dVZ,ViscA4D,ViscA4Z) = sf.compute_nu_hp_leith_sep_diag(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUD,dVD,dUZ,dVZ,ViscA4D,ViscA4Z);

    def compute_leith_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dU,dV) = sf.compute_nu_leith_diag_u(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_leith_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dU,dV) = sf.compute_nu_hp_leith_diag_u(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_leith_sep_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUD,dVD,dUZ,dVZ) = sf.compute_nu_leith_sep_diag_u(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUD,dVD,dUZ,dVZ);

    def compute_hp_leith_sep_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUD,dVD,dUZ,dVZ) = sf.compute_nu_hp_leith_sep_diag_u(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dUD,dVD,dUZ,dVZ);

    def compute_KE_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_KE_m(self.U,self.V,self.W,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_hp_KE_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_hp_KE_m(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_decomp_KE_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_decomp_KE_m(self.U,self.V,self.W,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_FF_T_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_FF_T_m(self.U,self.V,self.W,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_hp_FF_T_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_hp_FF_T_m(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_decomp_FF_T_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,T_KEdivV,m) = sf.compute_nu_decomp_FF_T_m(self.U,self.V,self.W,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,T_KEdivV,m);

    def compute_FF_T_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_FF_T_diag(self.U,self.V,self.W,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_FF_T_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_hp_FF_T_diag(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_decomp_FF_T_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh) = sf.compute_nu_decomp_FF_T_diag(self.U,self.V,self.W,
                                                                                                             self.Ubp,self.Vbp,
                                                                                                             self.Uhp,self.Vhp,
                                                                                                             self.thknss,self.dxc,self.dyc,
                                                                                                             self.dxu,self.dyu,self.dxv,self.dyv,
                                                                                                             self.dxq,self.dyq,self.ugrid,
                                                                                                             self.vgrid,self.pgrid,post_taper,
                                                                                                             trim_ml,zoversamp);
        return (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh);

    def compute_decomp_FF_T_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):

        (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh) = sf.compute_nu_decomp_FF_T_diag_u(self.U,self.V,self.W,
                                                                                                             self.Ubp,self.Vbp,
                                                                                                             self.Uhp,self.Vhp,
                                                                                                             self.thknss,self.dxc,self.dyc,
                                                                                                             self.dxu,self.dyu,self.dxv,self.dyv,
                                                                                                             self.dxq,self.dyq,self.ugrid,
                                                                                                             self.vgrid,self.pgrid,post_taper,
                                                                                                             trim_ml,zoversamp);
        return (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh);

    def compute_FF_T_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_FF_T_diag_u(self.U,self.V,self.W,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_FF_T_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_hp_FF_T_diag_u(self.U,self.V,self.W,self.Uhp,self.Vhp,self.Whp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_qbotdrag_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_qbotdrag_m(self.U,self.V,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_hp_qbotdrag_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_hp_qbotdrag_m(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_decomp_qbotdrag_m(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (T,m) = sf.compute_nu_decomp_qbotdrag_m(self.U,self.V,self.Ubp,self.Vbp,self.Uhp,self.Vhp,self.thknss,self.dxc,
                                    self.dyc,self.dxu,self.dyu,self.dxv,self.dyv,
                                    self.dxq,self.dyq,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (T,m);

    def compute_qbotdrag_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_qbotdrag_diag(self.U,self.V,self.thknss,self.dxu,self.dyv,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_qbotdrag_diag(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_hp_qbotdrag_diag(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxu,self.dyv,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_qbotdrag_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_qbotdrag_diag_u(self.U,self.V,self.thknss,self.dxu,self.dyv,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def compute_hp_qbotdrag_diag_u(self,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.compute_nu_hp_qbotdrag_diag_u(self.U,self.V,self.Uhp,self.Vhp,self.thknss,self.dxu,self.dyv,self.ugrid,
                                    self.vgrid,self.pgrid,post_taper,trim_ml,zoversamp);
        return (dU,dV);

    def get_dU_from_fZon_fMer(self,fZonU,fMerU,fZonV,fMerV,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1):
        (dU,dV) = sf.get_nu_dU_from_fZon_fMer(fZonU,fMerU,fZonV,fMerV,self);
        return (dU,dV);









            

