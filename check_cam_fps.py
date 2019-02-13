
# coding: utf-8

# In[ ]:


import time

def check_fps(prev_time) :
    cur_time = time.time() #Import the current time in seconds
    one_loop_time = cur_time - prev_time
    prev_time = cur_time
    fps = 1/one_loop_time
    return prev_time, fps

