from PIL import Image 
from images2gif import writeGif
import numpy as np
import imageio

def draw(mode: str):
    outfilename = "Goal_Arm3_8Hazards_" + mode +".gif"
    # outfilename = outfilename.encode()
    # print(type(outfilename))   
    filenames = []
    for i in range(1, 51):
        filename = "/home/yuqing/GUARD_FSRL/examples/mlp/figure/Goal_Arm3_8Hazards" + mode + str(i) + ".png"
        filenames.append(filename)
    # frames = []
    # for image_name in filenames: 
    #     im = Image.open(image_name)
    #     im = im.convert("RGB")
    #     im = np.array(im)
    #     frames.append(im)
    # writeGif(outfilename, frames, duration=0.01, subRectangles=False)
    frames = []
    for image_name in filenames:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(outfilename, frames, 'GIF', duration = 0.1)

def read():
    name = "imgtrain.txt"
    img = np.loadtxt(name)
    img = img.reshape((10, 1080, 1920, 3))
    frames = []
    for i in range(img.shape[0]):
        frames.append(img[i])
    imageio.mimsave("output.gif", frames, 'GIF', duration = 0.1)



if __name__ == "__main__":
    # draw("test")
    read()
    # draw("train")