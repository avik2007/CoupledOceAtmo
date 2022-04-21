
####################################
# :::::: Names ::::::::::::
region = 'Gulf Stream'


####################################
#::: Files and directories ::::::###
prnt = '/nobackup/htorresg/DopplerScat/modelling/'
chld = '/6hr_forcing/lwOBC/MITgcm/run/'
reg = 'GS'
dirc = prnt+reg+chld

#######################################
######################################
# ::::::: MITgcm information ::::::::

###################################
# :::: Dimensions :::::::
nx = 1200*2
ny = 600*2
nz = 88

###############################
# :::::: Timeline :::::::
tini = '20111001'
tend = '20111101'
tref = '20110913'

###############################
# ::::: paramenters :::::::
dt = 25 # time step of the model (seconds)
rate = 3600 # sampling rate (in seconds)
dtype = '>f4'
steps = rate/dt ##

# ::: smoothed files
dir_smoothed = '/nobackup/htorresg/DopplerScat/files/GS_smoothed/'

##################################
#################################

###################################
###################################
#:::::::: WaCM information ::::::::
prnt_wacm='/nobackup/htorresg/DopplerScat/Orbit/Aug_09_2019/interpolated/'
prnt_out = '/nobackup/htorresg/DopplerScat/files/GS_smoothed/'
 # ######################################
# 

