import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseEvent
from modulo_matrix import PrimeModMatrix

matrix_name = "classic"
n = 10

color_lamp_on = '#FFFD55'
color_lamp_off = '#383813'
color_switch_idle = '#EB3324'
color_switch_hover = '#992117'
color_switch_pressed = '#6E1811'

singular = False
if matrix_name == "classic":
    # switches adjacent lights
    matrix = PrimeModMatrix((n, n), 2)
    for i in range(n):
        if i - 1 >= 0:
            matrix[i - 1, i] = 1
        if 0 <= i < n:
            matrix[i, i] = 1
        if i + 1 < n:
            matrix[i + 1, i] = 1
    if matrix.row_reduced()[n-1, n-1] == 0:
        singular = True
elif matrix_name == "cyclic":
    # switches adjacent lights, two ends are considered connected
    matrix = PrimeModMatrix((n, n), 2)
    for i in range(n):
        matrix[(i-1) % n, i] = 1
        matrix[i, i] = 1
        matrix[(i+1) % n, i] = 1
    if matrix.row_reduced()[n-1, n-1] == 0:
        singular = True
else:
    # random matrix
    matrix = PrimeModMatrix.matrix(np.random.randint(0, 2, (n, n)), 2)
    while matrix.row_reduced()[n-1, n-1] == 0:
        # matrix is singular, regenerate
        matrix = PrimeModMatrix.matrix(np.random.randint(0, 2, (n, n)), 2)

# horizontal display range is 0 to 10
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
plt.axis('off')
plt.xlim(-1, 11)
plt.ylim(-1.2 * 10/n - 2, 1.5 * 10/n)

lamp_states = PrimeModMatrix.matrix(np.random.randint(0, 2, (n, 1)), 2)   # 1 is on, 0 is off
lamps: list[patches.Rectangle] = []
buttons: list[patches.Rectangle] = []
hovered: patches.Rectangle = None   # saves the instance of the button being hovered over
pressed: patches.Rectangle = None   # saves the instance of the button being pressed
for i in range(n):
    color_lamp = color_lamp_off if lamp_states[i, 0] == 0 else color_lamp_on
    lamps.append(patches.Rectangle((i * 10/n, 0), 10/n, 1.3 * 10/n, facecolor=color_lamp, edgecolor='k'))
    buttons.append(patches.Rectangle((i * 10/n, -2), 10/n, -10/n, facecolor=color_switch_idle, edgecolor='k'))
    ax.add_patch(lamps[i])
    ax.add_patch(buttons[i])

if not singular:
    solution = matrix.hstack(lamp_states).row_reduced().get_column(n).transpose()
    print(solution)
else:
    print("WARNING: non-singular matrix used; there's either no solution or multiple solutions")

def find_button(x: float, y: float) -> patches.Rectangle | None:
    if x < 0 or x > 10 or y < -2 - 10/n or y > -2:
        # not on a button
        return None
    return buttons[math.floor(n * x / 10)]

def on_move(event: MouseEvent):
    global hovered
    if event.inaxes != ax:
        return
    if hovered:
        hovered.set_facecolor(color_switch_idle)
        hovered = None
    b = find_button(event.xdata, event.ydata)
    if isinstance(b, patches.Rectangle) and b is not pressed:
        b.set_facecolor(color_switch_hover)
        hovered = b
    fig.canvas.draw_idle()

def on_press(event: MouseEvent):
    global pressed
    if event.inaxes != ax:
        return
    b = find_button(event.xdata, event.ydata)
    if isinstance(b, patches.Rectangle):
        pressed = b
        b.set_facecolor(color_switch_pressed)
    fig.canvas.draw_idle()

def on_release(event: MouseEvent):
    global pressed, lamp_states
    if pressed is None:
        return
    inside, _ = pressed.contains(event)
    if inside:
        button_i = buttons.index(pressed)
        pressed.set_facecolor(color_switch_hover)
        lamp_states += matrix.get_column(button_i)
        for j in range(n):
            lamps[j].set_facecolor(color_lamp_on if lamp_states[j, 0] == 1 else color_lamp_off)
        # update the solution matrix
        if not singular:
            solution[0, button_i] += 1
            print(solution)
    pressed = None
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
