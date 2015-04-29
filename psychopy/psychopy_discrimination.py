#!/usr/bin/env python
"""

Using psychopy to perform an experiment on discriminating clouds

(c) Laurent Perrinet - INT/CNRS & Jonathan Vacher - CeReMaDe

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
fileName = 'data/discriminating_v2_' + info['observer'] + '_' + info['timeStr'] + '.pickle'
#save to a file for future use (ie storing as defaults)
if dlg.OK:
    misc.toFile(fileName, info)
else:
    print('Interrupted gui... quitting')
    core.quit() #user cancelled. quit

print('generating data')

alphas = [-1., -.5, 0., 0.5, 1., 1.5, 2.]
fx, fy, ft = mc.get_grids(info['N_X'], info['N_Y'], info['N_frame_total'])
colors = [mc.envelope_color(fx, fy, ft, alpha=alpha) for alpha in alphas]
slows = [2*mc.rectif(mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, V_Y=0., V_X = 1.1, B_sf = 10.))) - 1 for color in colors]
fasts = [2*mc.rectif(mc.random_cloud(color * mc.envelope_gabor(fx, fy, ft, V_Y=0., V_X = 0.9, B_sf = 10.))) - 1 for color in colors]

print('go!      ')
win = visual.Window([info['screen_width'], info['screen_height']], fullscr=True)

stimLeft = visual.GratingStim(win, 
                            size=(info['screen_width']/2, info['screen_width']/2), 
                            pos=(-info['screen_width']/4, 0), 
                            units='pix',
                            interpolate=True,
                            mask = 'gauss',
                            autoLog=False)#this stim changes too much for autologging to be useful

stimRight = visual.GratingStim(win, 
                            size=(info['screen_width']/2, info['screen_width']/2), 
                            pos=(info['screen_width']/4, 0), 
                            units='pix',
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
            elif key in ['left', 'right']:
                if key in ['left'] :return 0.
                else: return 1.
            else:
                print "hit LEFT or RIGHT (or Esc) (You hit %s)" %key

clock = core.Clock()
FPS = 50.
def presentStimulus(i_alpha, left):
    """
    Present stimulus
    
    TODO : switch randomly up / down
    
    """
    phase_up = numpy.floor(numpy.random.rand() *(info['N_frame_total']-info['N_frame']))
    phase_down = numpy.floor(numpy.random.rand() *(info['N_frame_total']-info['N_frame']))
    up = numpy.random.randint(2)*2 - 1
    clock.reset()
    for i_frame in range(info['N_frame']): # length of the stimulus
        wait_for_next.draw()
        stimLeft.setTex(left * fasts[i_alpha][:, :, up*i_frame+phase_up]+ (1-left) * slows[i_alpha][:, :, up*i_frame+phase_down])
        stimRight.setTex((1.-left) * fasts[i_alpha][:, :, up*i_frame+phase_up]+ left * slows[i_alpha][:, :, up*i_frame+phase_down])
        stimLeft.draw()
        stimRight.draw()
#        while clock.getTime() < i_frame/FPS:
#            print clock.getTime(), i_frame/FPS
#            print('waiting')
        win.flip()

n_alpha = len(alphas)
results = numpy.zeros((n_alpha, info['nTrials']))
for i_trial in range(info['nTrials']):
    wait_for_next.draw()
    win.flip()
    core.wait(0.5)
    left = numpy.random.randint(2) # a random number between 0 and 1
    i_alpha = numpy.random.randint(n_alpha) # a random number between 0 and 1
    presentStimulus(i_alpha, left)
    wait_for_response.draw()
    win.flip()
    results[i_alpha, i_trial] = 2*(left == getResponse())-1

win.update()
core.wait(0.5)

win.close()

#save data
info['alphas'] = alphas
info['results'] = results
#numpy.savez(fileName, results=results, alphas=alphas)
misc.toFile(fileName, info)

# see the notebook
#print('analyzing results')
# TODO: loop over all data + make a fit for each
#print 'alphas :', alphas
#print '# of trials :', numpy.abs(results).sum(axis=1)
#print 'average results: ', (results.sum(axis=1)/numpy.abs(results).sum(axis=1)*.5+.5)
