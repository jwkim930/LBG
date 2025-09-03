import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import Event, MouseEvent

color_lamp_on = '#FFFD55'
color_lamp_off = '#383813'
color_switch_idle = '#EB3324'
color_switch_hover = '#992117'
color_switch_pressed = '#6E1811'
n = 10

# horizontal display range is 0 to 10
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
plt.axis('off')
plt.xlim(-1, 11)
plt.ylim(-1.2 * 10/n - 2, 1.5 * 10/n)

lamp_states = np.random.randint(0, 2, n)   # 1 is on, 0 is off
lamps = []
buttons = []
pressed: patches.Rectangle = None   # saves the instance of the button being pressed
for i in range(n):
    color_lamp = color_lamp_off if lamp_states[i] == 0 else color_lamp_on
    lamps.append(patches.Rectangle((i * 10/n, 0), 10/n, 1.3 * 10/n, facecolor=color_lamp, edgecolor='k'))
    buttons.append(patches.Rectangle((i * 10/n, -2), 10/n, -10/n, facecolor=color_switch_idle, edgecolor='k'))
    ax.add_patch(lamps[i])
    ax.add_patch(buttons[i])

def on_move(event: MouseEvent):
    if event.inaxes != ax:
        return
    for b in buttons:
        if isinstance(b, patches.Rectangle):
            inside, _ = b.contains(event)
            if inside and b is not pressed:
                b.set_facecolor(color_switch_hover)
            else:
                b.set_facecolor(color_switch_idle)
    fig.canvas.draw_idle()

def on_press(event: MouseEvent):
    global pressed
    for b in buttons:
        if b.contains(event)[0]:
            pressed = b
            b.set_facecolor(color_switch_pressed)
            break
    fig.canvas.draw_idle()

def on_release(event: MouseEvent):
    global pressed
    if pressed is None:
        return
    inside, _ = pressed.contains(event)
    if inside:
        pressed.set_facecolor(color_switch_hover)
        print("Pressed", buttons.index(pressed))
    pressed = None
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
