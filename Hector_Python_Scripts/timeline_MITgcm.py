import numpy as np

def timeline(initial_time,end_time,reference_time,
                 tdelta,rate,dt):
        """ Return timesteps corresponding to 
            the period selected

        Inputs:
        initial_time = '20200301000000' # YYYYMMDDHHMMSS
        end_time = '20200301000000'# YYYYMMDDHHMMSS
        reference_time (information in data.cal) = '20200119213000'
        timedelta = 1 hour
        dt = 45
        rate = 3600 # sampling rate (in seconds)
        """
        from dateutil.parser import parse
        from datetime import timedelta, datetime
        tim0 = parse(initial_time)
        tim1 = parse(end_time)
        tref = parse(reference_time)
        date=np.arange(tim0,tim1+timedelta(hours=tdelta),
                            timedelta(hours=tdelta)).astype(datetime)
        steps = int((tim0-tref).total_seconds()/rate)##self.rate
        i0 = steps*(rate/dt)
        del steps
        steps = int((tim1-tim0).total_seconds()/rate) ##self.rate
        i1 = i0+steps*(rate/dt)
        timesteps = np.arange(i0,i1+(rate / dt),(rate / dt))
        return timesteps,date
