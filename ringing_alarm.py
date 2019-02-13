
# coding: utf-8

# In[ ]:


import pygame

def select_alarm(result) :
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1 :
        sound_alarm("nomal_alarm.wav")
    else :
        sound_alarm("short_alarm.mp3")

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    

