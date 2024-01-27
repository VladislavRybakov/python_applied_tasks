import ffmpeg
from vispy import app, scene
from vispy.scene import ViewBox
from vispy.geometry import Rect
from funcs import *
app.use_app('pyglet')

video = False

asp = 16 / 9
h = 720
w = int(asp * h)
N = 1000
dt = 0.025

perception = 1/30
vrange=(0.1, 0.5)

k = 1
coeffs = np.array([1.0,  # alignment
                   3.0,  # cohesion
                   0.03,  # separation
                   0.0006,  # walls
                   0.03   # noise
                   ])

# k = 2
# coeffs = np.array([0.5,  # alignment
#                    0.5,  # cohesion
#                    0.01,  # separation
#                    0.1,  # walls
#                    0.1   # noise
#                    ])

# k = 3
# coeffs = np.array([20.0,  # alignment
#                    5.06,  # cohesion
#                    1.2,  # separation
#                    0.0002,  # walls
#                    0.01   # noise
#                    ])


boids = np.zeros((N, 7), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)

canvas = scene.SceneCanvas(keys='interactive',
                           bgcolor='#f8f9f4', #87ceeb #ffc781
                           show=True,
                           size=(w, h))

view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

# агенты
arrows = scene.Arrow(arrows=directions(boids),
                     arrow_color='#5d3277', #572854
                     arrow_size=6,
                     connect='segments',
                     parent=view.scene)

# рисуем границы
scene.Line(pos=np.array([[0, 0],
                         [asp, 0],
                         [asp, 1],
                         [0, 1],
                         [0, 0]
                         ]),
           color='#a16db7',
           connect='strip',
           method='gl',
           parent=view.scene
           )

text = scene.Text('0.0 fps', parent=view.scene, color='#a16db7')
text.pos = 0.55, 0.95

dzeros = np.zeros((N, N), dtype=float)

if video:
    fname = f"boids_c{k}_N{N}.mp4"
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{w}x{h}".format(), r=60)
        .output(fname, pix_fmt='yuv420p', preset='slower', r=60, t=60)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


def update(event):
    global process
    global video
    herding(boids, perception, coeffs, asp, dzeros)
    propagate(boids, dt, vrange)

    periodic_walls(boids, asp)
    arrows.set_data(arrows=directions(boids))

    text.text = '\tN = {}\ta = {}\tc = {}\ts = {}\tw = {}\tn = {}\t FPS = {}'.format(N, *coeffs, round(canvas.fps, 2))
    if video:
        frame = canvas.render(alpha=False)
        try:
            process.stdin.write(frame.tobytes())
        except:
            video = False
            exit()
    else:
        canvas.update(event)


timer = app.Timer(interval=0, start=True, connect=update)

if __name__ == '__main__':
    canvas.measure_fps()
    app.run()
    if video:
        process.stdin.close()
        process.wait()
