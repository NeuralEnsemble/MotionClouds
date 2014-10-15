#!/usr/bin/env python
"""

Using psychopy to perform an experiment on competing clouds

(c) Laurent Perrinet - INT/CNRS

 See http://invibe.net/LaurentPerrinet/SciBlog/2012-12-12 for a small tutorial
 
"""
# width and height of your screen
w, h = 1920, 1200
w, h = 2560, 1440 # iMac 27''

# width and height of the stimulus
w_stim, h_stim = 1024, 1024


print('launching experiment')
from psychopy import visual, core, event, logging, gui, misc
logging.console.setLevel(logging.DEBUG)

import os, numpy
import MotionClouds as mc
import time
experiment = 'competing_v2_'
print('launching experiment')
from psychopy import visual, core, event, logging, gui, misc
logging.console.setLevel(logging.DEBUG)

#if no file use some defaults
info = {}
info['observer'] = 'anonymous'
info['screen_width'] = w
info['screen_height'] = h
info['nTrials'] = 50
info['N_X'] = mc.N_X # size of image
info['N_Y'] = mc.N_Y # size of image
info['N_frame_total'] = mc.N_frame # a full period. in time frames
info['N_frame'] = mc.N_frame # length of the presented period. in time frames

try:
    dlg = gui.DlgFromDict(info)
except:
    print('Could not load gui... running with defaut parameters')
    print(info)
    
info['timeStr'] = time.strftime("%b_%d_%H%M", time.localtime())
fileName = 'data/' + experiment + info['observer'] + '_' + info['timeStr'] + '.pickle'
#save to a file for future use (ie storing as defaults)
if dlg.OK:
    misc.toFile(fileName, info)
else:
    print('Interrupted gui... quitting')
    core.quit() #user cancelled. quit

print('generating data')


fx, fy, ft = mc.get_grids(info['N_X'], info['N_Y'], info['N_frame_total'])
color = mc.envelope_color(fx, fy, ft)
up = 2*mc.rectif(mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, V_X=+.5))) - 1
down = 2*mc.rectif(mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, V_X=-.5))) - 1

print('go!      ')
win = visual.Window([info['screen_width'], info['screen_height']], fullscr=True)

stim = visual.GratingStim(win, 
        size=(info['screen_height'], info['screen_height']), units='pix',
        interpolate=True,
        mask = 'gauss',
        autoLog=False)#this stim changes too much for autologging to be useful

wait_for_response = visual.TextStim(win, 
                        text = u"?", units='norm', height=0.15, color='DarkSlateBlue',
                        pos=[0., -0.], alignHoriz='center', alignVert='center' ) 
wait_for_next = visual.TextStim(win, 
                        text = u"+", units='norm', height=0.15, color='BlanchedAlmond',
                        pos=[0., -0.], alignHoriz='center', alignVert='center' ) 

def getResponse():
    event.clearEvents()#clear the event buffer to start with
    resp = None#initially
    while 1:#forever until we return a keypress
        for key in event.getKeys():
            #quit
            if key in ['escape', 'q']:
                win.close()
                core.quit()
                return None
            #valid response - check to see if correct
            elif key in ['down', 'up']:
                if key in ['down'] :return -1
                else: return 1
            else:
                print "hit DOWN or UP (or Esc) (You hit %s)" %key

clock = core.Clock()
FPS = 50.
def presentStimulus(C_A, C_B):
    """Present stimulus
    """
    phase_up = numpy.floor(numpy.random.rand() *(info['N_frame_total']-info['N_frame']))
    phase_down = numpy.floor(numpy.random.rand() *(info['N_frame_total']-info['N_frame']))
    clock.reset()
    for i_frame in range(info['N_frame']): # length of the stimulus
        stim.setTex(C_A * up[:, :, i_frame+phase_up]+C_B * down[:, :, i_frame+phase_down])
        stim.draw()
#        while clock.getTime() < i_frame/FPS:
#            print clock.getTime(), i_frame/FPS
#            print('waiting')
        win.flip()

results = numpy.zeros((2, info['nTrials']))
for i_trial in range(info['nTrials']):
    wait_for_next.draw()
    win.flip()
    core.wait(0.5)
    C_A = numpy.random.rand() # a random number between 0 and 1
    presentStimulus(C_A, 1. - C_A)
    wait_for_response.draw()
    win.flip()
    ans = getResponse()
    results[0, i_trial] = ans
    results[1, i_trial] = C_A

win.update()
core.wait(0.5)

win.close()

#save data
fileName = 'data/' + experiment + info['observer'] + '_' + info['timeStr']
numpy.save(fileName, results)

print('analyzing results')
# see the notebook but this gives a quick plot of the results
import pylab
from scipy.optimize import curve_fit
def sigmoid(c, c0, k):
    y = 1 / (1 + np.exp(-k*(c-c0)))
    return y

cdata, ydata = results[1, :], .5*results[0, : ]+.5
pylab.plot(cdata, ydata, 'o', label='data')

popt, pcov = curve_fit(sigmoid, cdata, ydata)
pylab.text(0.05, 0.8, 'mean : %0.2f , slope : %0.2f ' %(popt[0], popt[1]))
c = np.linspace(0, 1, 50)
y = sigmoid(c, *popt)
pylab.plot(c, y, label='fit')
pylab.ylim(-.05, 1.05)
pylab.legend(loc='best')
pylab.xlabel('contrast')
pylab.savefig('psychopy_competing.png')
