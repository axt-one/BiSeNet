#!/usr/bin/env python3
# coding=utf-8

import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from scipy import interpolate
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import rospy
import rosgraph
from geometry_msgs.msg import *
from speech_recognition_msgs.msg import *
from speech_recognition_msgs.srv import *


import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

import time

t1 = time.time()

torch.set_grad_enabled(False)
np.random.seed(123)
#tex_water = {'num':0, 'width':256, 'height':256, 'name':'./test6.png'}
tex_water = {'num':0, 'width':1024, 'height':256, 'name':'./test8.png'}
tex_fireflow = {'num':1, 'width':64, 'height':64, 'name':'./fireflow.png'}
tex_splash = {'num':2, 'width':64, 'height':64, 'name':'./splash.png'}
tex_firesword = {'num':3, 'width':64, 'height':64, 'name':'./firesword.png'}
tex_background = {'num':4, 'width':64, 'height':64, 'name':'./background.png'}
tex_mode = {'num':5, 'width':512, 'height':512, 'name':'./mode.png'}
tex_video = {'num':6, 'width':512, 'height':512}
tex_foreground = {'num':7, 'width':512, 'height':512}
tex_label = {'num':8, 'width':512, 'height':512}
tex_list = [tex_water, tex_fireflow, tex_splash, tex_firesword, tex_background, tex_mode, tex_video, tex_foreground, tex_label]
drawmode = 0 #0:水の呼吸, 1:ヒノカミ神楽
mode = 1
drawframe = 0

box = []
orbit = np.zeros((2,2,10)) #2点x2次元x10フレーム


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
parse.add_argument('--video', type=int, default=0)
args = parse.parse_args()
cfg = cfg_factory[args.model]

# define model
net = model_factory[cfg.model_type](3)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
#net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

if args.video == 0:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(f'/home/mech-user/Videos/video{args.video}.mp4')
ImageWidth_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ImageHeight_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def roscb(msg):
    global drawmode, drawframe
    rospy.loginfo(msg.transcript[0])
    if mode == 2:
        if msg.transcript[0] == "水の呼吸":
            drawmode = 0
            drawframe = 1
        if msg.transcript[0] == "ヒノカミ神楽":
            drawmode = 1

    

def rosinit():
    if rosgraph.is_master_online():
        rospy.init_node("sample_julius")
        rospy.Subscriber("/speech_to_text", SpeechRecognitionCandidates, roscb)

def texparaminit():
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)#GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)#GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)

def loadtex(tex):
    glBindTexture(GL_TEXTURE_2D, tex['num'])
    texparaminit()
    teximage = cv2.imread(tex['name'], cv2.IMREAD_UNCHANGED).astype('uint8')
    teximage = cv2.cvtColor(teximage, cv2.COLOR_BGRA2RGBA)
    teximage = cv2.flip(teximage, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex['width'], tex['height'], 0, GL_RGBA, GL_UNSIGNED_BYTE, teximage.tobytes())


def init():
    global tex_list
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texnums=glGenTextures(len(tex_list))
    for i, tex in enumerate(tex_list):
        tex['num'] = texnums[i]
    loadtex(tex_water)
    loadtex(tex_splash)
    loadtex(tex_firesword)
    loadtex(tex_fireflow)
    loadtex(tex_background)
    loadtex(tex_mode)
    glEnable(GL_TEXTURE_2D)

def spline(x,y,point,deg):
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u = np.linspace(0,1,num=point,endpoint=True) 
    spline = interpolate.splev(u,tck)
    return spline[0],spline[1]

def draworbit():
    for i in range(orbit.shape[2]):
        glBegin(GL_LINES)
        glVertex3f(orbit[0][0][i], orbit[0][1][i], 0.1)
        glVertex3f(orbit[1][0][i], orbit[1][1][i], 0.1)
        glEnd()

def drawflow():
    glBindTexture(GL_TEXTURE_2D, tex_water['num'])
    n = orbit.shape[2]
    glBegin(GL_TRIANGLE_STRIP)
    for i in range(n):
        if np.all(orbit[:,:,i] == 0): continue
        glTexCoord2f(i/n, 0.0)
        glVertex3f(orbit[0][0][i], orbit[0][1][i], 0.1)
        glTexCoord2f(i/n, 1.0)
        glVertex3f(orbit[1][0][i], orbit[1][1][i], 0.1)
    glEnd()

def drawsplash(coord, size):
    glBindTexture(GL_TEXTURE_2D, tex_splash['num'])
    glBegin(GL_QUADS)
    glTexCoord2f(1.0, 0.0); glVertex3f(coord[0]+size[0], coord[1], 0.2)
    glTexCoord2f(1.0, 1.0); glVertex3f(coord[0]+size[0], coord[1]+size[1], 0.2)
    glTexCoord2f(0.0, 1.0); glVertex3f(coord[0], coord[1]+size[1], 0.2)
    glTexCoord2f(0.0, 0.0); glVertex3f(coord[0], coord[1], 0.2)
    glEnd()


def drawflow2():
    index = np.where(np.all(orbit.reshape(4,-1) == 0, axis=0))[0]
    orbit_not_zero = np.delete(orbit.reshape(4,-1), index, 1)
    index2 = np.unique(orbit_not_zero[0:2,:], axis=1, return_index=True)[1]
    index3 = np.unique(orbit_not_zero[2:4,:], axis=1, return_index=True)[1]
    index4 = np.sort(np.array(list(set(index2) & set(index3))))
    if len(index4) <= 3:
        return
    orbit_not_zero = orbit_not_zero[:,index4].reshape(2,2,-1)

    x1, y1 = spline(orbit_not_zero[0,0,:], orbit_not_zero[0,1,:], 50, 3)
    x2, y2 = spline(orbit_not_zero[1,0,:], orbit_not_zero[1,1,:], 50, 3)
    n = len(x1)
    if drawmode == 0:
        glBindTexture(GL_TEXTURE_2D, tex_water['num'])
    elif drawmode == 1:
        glBindTexture(GL_TEXTURE_2D, tex_fireflow['num'])
    glBegin(GL_TRIANGLE_STRIP)
    # splist = []
    for i in range(len(x1)):
        #if np.random.randint(10) == 0:
        #    splist += [[x2[i], y1[i]]]
        glTexCoord2f(i/n, 0.0)#(1+np.sin(i/len(x1)*5*np.pi))*0.01)
        glVertex3f(x1[i], y1[i], 0.1)
        glTexCoord2f(i/n, 1.0)
        glVertex3f(x2[i], y2[i], 0.1)
    glEnd()

    if drawmode == 0:
        for i in range(orbit_not_zero.shape[2]):
            size = np.array([10*i-i*i, 10*i-i*i])*ImageWidth_*0.002
            drawsplash(orbit_not_zero[0,:,i], size)
            drawsplash(orbit_not_zero[1,:,i], size)
        
        #for j in range(5):
        #    coord = np.average(orbit_not_zero[:,:,i],axis=0,weights=[j/5,1-j/5]) + 20*np.sin([j*np.pi/4, j*np.pi/2])
        #    drawsplash(coord, [20*i-2*i*i,20*i-2*i*i])

    
    #for coord in splist:
    #    drawsplash(coord, np.random.randint(30,60,(2)))


def drawsword():
    if not len(box) == 0:
        r1 = 2*box[0] - box[3]
        r2 = 2*box[1] - box[2]
        r3 = 2*box[2] - box[1]
        r4 = 2*box[3] - box[0]
        glBindTexture(GL_TEXTURE_2D, tex_firesword['num'])
        glBegin(GL_QUADS)#0-1,2-3長辺;1-2,3-0短辺
        glTexCoord2f(1.0, 0.0); glVertex3f(r1[0], r1[1], 0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f(r2[0], r2[1], 0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f(r3[0], r3[1], 0.5)
        glTexCoord2f(0.0, 0.0); glVertex3f(r4[0], r4[1], 0.5)
        glEnd()

def drawcambus(texnum, z):
    glBindTexture(GL_TEXTURE_2D, texnum)
    glBegin(GL_QUADS)
    glTexCoord2f(1.0, 0.0); glVertex3f(ImageWidth_, 0.0, z)
    glTexCoord2f(1.0, 1.0); glVertex3f(ImageWidth_, ImageHeight_, z)
    glTexCoord2f(0.0, 1.0); glVertex3f(0.0, ImageHeight_, z)
    glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 0.0, z)
    glEnd()

def selectmode():
    glColor4f(0.0, 0.0, 1.0, 0.5)
    glRectf(0.0, 0.0, ImageWidth_, ImageHeight_)

    glPushMatrix()
    #glRasterPos3f(100,50,0.9)
    glTranslatef(100,50,0.9)
    glLineWidth(10)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    text = b'mode'
    glutStrokeString(GLUT_STROKE_ROMAN, text)
    #glutBitmapString(GLUT_BITMAP_HELVETICA_18, text)
    glPopMatrix()


def display():
    # global t1
    # print(time.time() - t1)
    # t1 = time.time()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #glDisable(GL_DEPTH_TEST)

    drawcambus(tex_video['num'], -0.9)

    if mode == 0:
        drawcambus(tex_mode['num'], -0.8)
        #selectmode()
    elif mode == 1 or mode == 2:
        if drawmode == 0:
            drawflow2()
        elif drawmode == 1:
            drawcambus(tex_background['num'], -0.8)
            drawcambus(tex_foreground['num'], -0.7)
            drawflow2()
            drawsword()
    elif mode == 3:
        drawcambus(tex_label['num'], -0.8)

    glutSwapBuffers()
    #glFlush()

def captureImage():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (tex_video['width'], tex_video['height']))
        frame = cv2.flip(frame, 0)
        #frame = np.dstack([frame, np.full((tex_video['height'], tex_video['width']), 255, dtype='uint8')])
        glBindTexture(GL_TEXTURE_2D, tex_video['num'])
        texparaminit()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_video['width'], tex_video['height'], 0, GL_RGB, GL_UNSIGNED_BYTE, frame.tobytes())

def idle():
    global box, orbit, drawframe, t1
    captureImage()
    ret, im_raw = cap.read()
    if ret:
        im = cv2.resize(im_raw, (160,96))
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()
        #t1 = time.time()
        out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        #print(1 / (time.time() - t1))
        #t1 = time.time()
        
        ### make texture of foreground ###
        #if (mode == 0 or mode == 1 ) and drawmode == 1:
        if mode == 1 or mode == 2:
            foreground_mask = np.where(out != 0, 255, 0).astype('uint8')
            foreground_mask = cv2.resize(foreground_mask, (im_raw.shape[1], im_raw.shape[0]))#, interpolation=cv2.INTER_CUBIC)
            foreground_tex = np.dstack([im_raw, foreground_mask])#np.where(foreground_mask > 100, 255, 0).astype('uint8')
            foreground_tex = cv2.cvtColor(foreground_tex, cv2.COLOR_BGRA2RGBA)
            foreground_tex = cv2.resize(foreground_tex, (tex_foreground['width'], tex_foreground['height']))
            foreground_tex = cv2.flip(foreground_tex, 0)
            glBindTexture(GL_TEXTURE_2D, tex_foreground['num'])
            texparaminit()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_video['width'], tex_video['height'], 0, GL_RGBA, GL_UNSIGNED_BYTE, foreground_tex.tobytes())

        if mode == 3:
            palette = np.array([[0,255,255,127],[0,0,255,127],[255,0,0,127]])
            label_image = palette[out].astype('uint8')
            label_image = cv2.resize(label_image, (tex_label['width'], tex_label['height']))
            label_image = cv2.flip(label_image, 0)
            glBindTexture(GL_TEXTURE_2D, tex_label['num'])
            texparaminit()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_label['width'], tex_label['height'], 0, GL_RGBA, GL_UNSIGNED_BYTE, label_image.tobytes())

        if drawframe > 0:
            index = np.where(np.any(orbit.reshape(4,-1) != 0, axis=0))[0]
            orbit[:,1,index] += ImageHeight_*0.05
            drawframe += 1
            if drawframe == 6:
                drawframe = 0
            

        out = cv2.flip(out, 0)
        out_sword = cv2.resize(np.where(out == 2, 255, 0).astype('uint8'), (160, 90))

        if int(cv2.__version__[0]) >= 4:
            contours, _= cv2.findContours(out_sword,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _= cv2.findContours(out_sword,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        
        ### calc parameter about detected sword ###
        if not len(contours) == 0:
            tmp = -1
            rect = None
            for i in range(len(contours)):
                rect_i = cv2.minAreaRect(contours[i])
                length = np.max(rect_i[1])
                if length > tmp:
                    tmp = length
                    rect = rect_i            
            box = cv2.boxPoints(rect)
            if rect[1][0] > rect[1][1]:
                box = np.roll(box, 1, axis=0)
            box[:,0] = box[:,0]*ImageWidth_/out_sword.shape[1]
            box[:,1] = box[:,1]*ImageHeight_/out_sword.shape[0]
            orbit = np.roll(orbit, 1, axis=2)
            orbit[:,:,0] = box[:2,:]
        else:
            box=[]
            orbit = np.roll(orbit, 1, axis=2)
            orbit[:,:,0] = np.zeros((2,2))

    glutPostRedisplay()

def reshape(w, h):
    a = ImageWidth_/ImageHeight_
    glViewport(0, 0, int(h*a), h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, ImageWidth_, 0.0, ImageHeight_)


def _keyfunc (c, x, y):
    global mode, drawmode, drawframe
    if c == b'q':
        cap.release()
        sys.exit (0)
    if c == b'm':
        mode = 0
    if mode == 0:
        if c == b'1':
            mode = 1
            drawmode = 0
        if c == b'2':
            mode = 2
            drawmode = -1
        if c == b'3':
            mode = 3
    if mode == 1:
        if c == b'1':
            drawmode = 0
            drawframe = 1
        if c == b'2':
            drawmode = 1
        if c == b'3':
            drawmode = -1


def main():
    rosinit()
    glutInit([])
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(ImageWidth_, ImageHeight_)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("gazo syori no kokyu")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(_keyfunc)
    glutIdleFunc(idle)
    #glutFullScreen()
    glutMainLoop()

if __name__ == "__main__":
    main()