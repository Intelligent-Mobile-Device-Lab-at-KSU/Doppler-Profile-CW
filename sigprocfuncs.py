import numpy as np
import array
import peakutils
import math
import statistics
from IPython import display
import time
import matplotlib.pyplot as pyplot

def plot_doppler(contrast,t_FTDP,f_FTDP,typ):
    ampmin=np.max(np.abs(typ.T))/contrast
    xx = t_FTDP
    yy = f_FTDP
    zz = 20*np.log10(np.maximum(abs(typ.T),ampmin) / ampmin)
    ax = pyplot.gca()
    ax.set(xlim=(min(xx), max(xx)), ylim=(min(yy), max(yy)))
    pyplot.imshow(zz, extent=[min(xx), max(xx), min(yy), max(yy)])
    #ax1.set_xticklabels(['', xx])
    pyplot.colorbar()
    #pyplot.tight_layout()
    pyplot.axis('auto')
    pyplot.show()

def read_complex_binary(inputfilename, CHUNK_SIZE):
    with open(inputfilename, 'rb') as fid: # b is important -> binary
        a=array.array("f")
        a.fromfile(fid, CHUNK_SIZE)
        a=np.reshape(a,(-1, 2)).T
        fid.close()
        v = a[0,:]+a[1,:]*1j
        #print(v)
    return v

def findLocs(MFxPos):
    locs = [0]
    m = .29
    x = MFxPos/max(MFxPos)
    while (len(locs) != 2) and (m <= 1):
        locs = peakutils.indexes(x, thres=0, min_dist=10000)
        res = []
        for ind in locs:
            if (x[ind]<m) is False:
                res.append(ind)
                
        locs = res #[pks locs]=findpeaks(MFxPos/max(MFxPos),'MINPEAKHEIGHT',m,'MINPEAKDISTANCE',10000)
        m = m + .05   
    return locs

def ProcessRawDataToAlignedDopplerProfile(fc,fs,fsc,doppler_window_size,win_adv,win_duration,display_option_1,display_option_2, display_option_3, dirpath, thefilename):
    #fig = pyplot.figure()
    #ax = pyplot.gca()
    # Settings
    f0 = fc
    Fs = fs
    Fsc = fsc
    FTDP_adv = win_adv
    winddur = win_duration
    dfwin = doppler_window_size

    # Will display the Aligned Doppler Spectrum for all processed windows in real-time
    displayDopplerProfile_in_RealTime = display_option_1

    # Will display the aligned Doppler Spectra (different from Doppler Profile)
    # in real-time.
    displayDopplerSpectra_in_RealTime = display_option_2

    # Will compute and display the Unaligned Doppler Profile
    showUnalignedDopplerProfile = display_option_3

    dirpath0 = dirpath
    fname = thefilename

    ## Define Constants

    ## Pre-Determine the frequency axis for the Doppler Profile (based on the window duration)
    fsc_view = Fsc # Hz, Center Frequency for Frequency-Time Doppler Profile (FTDP)
    wind = math.floor(winddur*Fs) # samples, (or nfft size) this should be the desired window
    FTDP_adv_samps = math.floor(FTDP_adv*Fs) # samples, sliding window advance amount

    f_FTDP = []
    t_FTDP = [0]
    f = (np.arange(0,wind)*(Fs/wind)).T
    f = f[f<(Fs/2)]
    
    # Aligned Indexing
    f0 = fsc_view
    th = 0
    myStep = 1
    while True:
        if (len(np.nonzero(abs(f-(f0-dfwin)) <= th)[0]) == 1) and (len(np.nonzero(abs(f-f0) <= th)[0]) == 1) and (len(np.nonzero(abs(f-(f0+dfwin-myStep)) <= th)[0]) == 1):
            break
        myStep = myStep + 1
        #th = th + 1;
    
    scindex_lower = int(np.nonzero(abs(f-(f0-dfwin)) <= th)[0][0])
    scindex_center = int(np.nonzero(abs(f-f0) <= th)[0][0])
    scindex_upper = int(np.nonzero(abs(f-(f0+dfwin-myStep)) <= th)[0][0])    
    f_FTDP = f[np.arange(scindex_lower,scindex_upper+1)]-f0 # Get FTDP frequency range
    
    # Unalgined Indexing
    dfwin_unaligned = 40000
    th = 1
    myStep = 1
    while False:
        if (len(np.nonzero(abs(f-(f0-dfwin)) <= th)[0]) == 1) and (len(np.nonzero(abs(f-f0) <= th)[0]) == 1) and (len(np.nonzero(abs(f-(f0+dfwin-myStep)) <= th)[0]) == 1):
            break
        myStep = myStep + 1
        #th = th + 1;
    scindex_lower_unaligned = int(np.nonzero(abs(f-(f0-dfwin_unaligned)) <= th)[0][0])
    scindex_center_unaligned = int(np.nonzero(abs(f-f0) <= th)[0][0])
    scindex_upper_unaligned = int(np.nonzero(abs(f-(f0+dfwin_unaligned-myStep)) <= th)[0][0])
    f_FTDP_unaligned = f[np.arange(scindex_lower_unaligned,scindex_upper_unaligned+1)]-f0
    
    ## Doppler Processing
    Fcfo = [] # Holder for CFO Measurement
    FTDP_Window_Doppler_Unaligned = [] # Holder for FTDP Unaligned Doppler
    FTDP_Window_Doppler = [] # Holder for FTDP Aligned Doppler
    FTDP_Window_Doppler = np.array(FTDP_Window_Doppler)
    FTDP_Window_Doppler_Unaligned = np.array(FTDP_Window_Doppler_Unaligned)
    #FTDP_Window_Doppler_Unaligned = np.array(FTDP_Window_Doppler_Unaligned)
    #FTDP_Window_Doppler = np.array(FTDP_Window_Doppler)
    FTDP_Window_ind = 1 # Value for Window Index
    nend = 1000000 # Controls how many windows are processed. Set to large number (1000000) as infinity
    noPeaksPreviously = 0 # Control variable in case the FFT does not reveal a strong peak
    
    x = read_complex_binary(dirpath0 + fname + '.dat',69000000) # Reads the complex-binary data
    
    L = len(x)
    for currSlideLoc in range(0, L-wind, FTDP_adv_samps):
        
        if  (FTDP_Window_ind > nend):
            break
        
        # Get window location and update t_FTDP (time vector)
        currTime = 0
        if currSlideLoc == 0:
           # For first segment (window) of signal
           windLoc = np.arange(1,wind+1)
        else:
            # For all other segments of signal
            windLoc = np.arange(currSlideLoc,currSlideLoc+wind)
            t_FTDP.append(t_FTDP[-1]+FTDP_adv)
            currTime = t_FTDP[-1]+FTDP_adv
    
        #print(windLoc)
        #Take large FFT over 1st window in this segment
        segment = x[windLoc]
        Fx = np.fft.fft(segment)
        MFxPos = abs(Fx[1:int(wind/2)])
        if showUnalignedDopplerProfile:
            Fx22 = np.fft.fft(np.multiply(np.hamming(wind),segment),wind)
            MFxPos22 = abs(Fx22[1:int(wind/2)])
            FTDP_Window_Doppler_Unaligned[FTDP_Window_ind,:,1] = MFxPos22[scindex_lower_unaligned:scindex_upper_unaligned]
    
        # Be sure only ONE massive peak exists (stronger in LOS than NLOS) if
        # not then need to skip over this signal segment. Time vector still
        # increases. For now it duplicates the window.

        # findLocs may need to be tunned for the specific Doppler profile
        # depending on the test equipment used. See findLocs.m, and edit the
        # parameters: 'MINPEAKHEIGHT',m,'MINPEAKDISTANCE',10e3, also notice it
        # is the locally normalized version of the current Doppler Spectra being processed, 
        # not the globally normalized version across the entire Doppler Profile. 
        
        #print(MFxPos.tolist())
        locs = peakutils.indexes(MFxPos, thres=0, min_dist=10000)
        locs = np.where(MFxPos == max(MFxPos[locs]))[0]
        #locs = findLocs(MFxPos)
        if not locs:
            print('Flag1: Empty locs')
            noPeaksPreviously = 1
            FTDP_Window_ind = FTDP_Window_ind + 1
            if FTDP_Window_ind != 1:
                FTDP_Window_Doppler[FTDP_Window_ind,:] = FTDP_Window_Doppler[FTDP_Window_ind-1,:]
                continue
    
        if len(locs) != 1:
            print('Flag2: locs length not one?')
            #pplot(f,MFxPos)
            break
        
        # Correct for CFO
        CFOfine = int(f[locs[0]]-Fsc)
        Fcfo.append(f[locs[0]]-Fsc)
        t1 = -1j*2*math.pi*CFOfine
        t2 = np.arange(1,len(segment)+1).T
        t3 = np.multiply(t1,t2)
        t4 = t3*(1/Fs)
        t5 = np.exp(t4)
        segment = np.multiply(segment,t5)
        #print(segment)
        #break
    
        # Obtain corrected spectrum, which is the Doppler Profile for the
        # current window.
        #print(len(segment))
        Fx = np.fft.fft(np.multiply(np.hamming(wind),segment))
        MFxPos = abs(Fx[1:int(wind/2)])
        locs = peakutils.indexes(MFxPos, thres=0, min_dist=10000)
        locs = np.where(MFxPos == max(MFxPos[locs]))[0]
        if len(locs) != 1:
            print('Flag3: 2nd locs length not one?')
            break
    
        if noPeaksPreviously:
            if len(FTDP_Window_Doppler)==0:
                FTDP_Window_Doppler = MFxPos[scindex_lower:scindex_upper+1]
            else:
                FTDP_Window_Doppler = np.vstack([FTDP_Window_Doppler, MFxPos[scindex_lower:scindex_upper+1]])
        
        else:
            # Need to code: Estimate Doppler Profiles between noPeaks segement 
            # and current segment. For now just treat as normal.
            # <code to estimate missed segments>
            #FTDP_Window_Doppler = MFxPos[scindex_lower:scindex_upper]
            if len(FTDP_Window_Doppler)==0:
                FTDP_Window_Doppler = MFxPos[scindex_lower:scindex_upper+1]
            else:
                FTDP_Window_Doppler = np.vstack([FTDP_Window_Doppler, MFxPos[scindex_lower:scindex_upper+1]])
        
        #print(FTDP_Window_Doppler)
        
            #imagesc(,,);
            #title(fname)
            #xlabel('Doppler Frequency (Hz)')
            #ylabel('Time (s)')
            #title(['WaterFall Plot: Window #' num2str(FTDP_Window_ind)])
            #%axis([min(t_FTDP) max(t_FTDP) min(f_FTDP) max(f_FTDP)])
            #colorbar
            #drawnow
            
        #if displayDopplerSpectra_in_RealTime:
            #figure(2)
            #plot(f_FTDP,flipud(MFxPos(scindex_lower:scindex_upper)))
            #xlabel('Doppler Frequency (Hz)')
            #ylabel('Energy (|Y|)')
            #title(['Doppler Spectra for Window #' num2str(FTDP_Window_ind)])
            #drawnow
            
        # Update the FTDP window index and report status
        FTDP_Window_ind = FTDP_Window_ind +1
        display.clear_output(wait=True)
        print('PRE-PROCESSING...')
        mystr = 'Working %s: %.2f%% Complete\r' % (fname,100*(currSlideLoc/(L-wind)))
        print(mystr)
        if displayDopplerProfile_in_RealTime and (len(t_FTDP)>5):#displayDopplerProfile_in_RealTime:
            yy = 156250-f[scindex_lower:scindex_upper+1]
            xx = t_FTDP
            zz = 20*np.log10(FTDP_Window_Doppler)
            ax.clear()
            ax.set(xlim=(min(xx), max(xx)), ylim=(min(yy), max(yy)))
            pyplot.imshow(zz, extent=[min(xx), max(xx), min(yy), max(yy)])
            pyplot.colorbar()
            pyplot.axis('auto')
            pyplot.show()
            #fig.canvas.draw()
            #fig.canvas.flush_events()
    
    #print(len(MFxPos[np.arange(scindex_lower,scindex_upper+1)]))
    #print(len(x))
    #print(f_FTDP_unaligned)
    return [t_FTDP, f_FTDP ,Fcfo, FTDP_Window_Doppler, FTDP_Window_Doppler_Unaligned]

#read_complex_binary('rfc_coll_01.dat',20000000)