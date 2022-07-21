
####################################
# :::::: Names ::::::::::::
region = 'California_Current'


####################################
#::: Files and directories ::::::###
#prnt = '/nobackupp2/htorresg/DopplerScat/modelling/CCS/'
chld = 'HourlyOBC/run_500m/MITgcm/run/'
prnt = '/nobackupp17/htorresg/DopplerScat/modelling/CCS/'
dirc = prnt+chld

#######################################
######################################
# ::::::: MITgcm information ::::::::

###################################
# :::: Dimensions :::::::
nx = 1200*2
ny = 1200*2
nz = 87
dx = 500 ## m
dy = 500 ##

###############################
## from 12/01/2011 00:00
##  to 07/25/2012
# :::::: Timeline :::::::
tini = '20120719003000'
tend = '20120720043000'
tref = '20111201000000'

###############################
# ::::: paramenters :::::::
dt = 25 # time step of the model (seconds)
rate = 3600 # sampling rate (in seconds)
dtype = '>f4'
steps = rate/dt ##
timedelta = 1. ## hours


##################################
#################################

###################################
###################################
#:::::::: Extra info below :::::::::
 # ######################################
# 

