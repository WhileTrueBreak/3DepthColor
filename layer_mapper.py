import numpy as np
from stl import mesh
from scipy.ndimage import shift
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import fast_simplification
import itertools
import pymeshlab
import argparse
import trimesh
import random
import pyfqmr
import sys
import os

import cv2

from color_solver import *

INIT_LAYER_HEIGHT = 0.16
LAYER_HEIGHT = 0.08

MAX_WIDTH = 63
MAX_HEIGHT = 88

def getFaceSquare(ix, iy, vw, vh):
    topleft = ix%vw+iy*vw
    topright = (ix+1)%vw+iy*vw
    bottomleft = ix%vw+(iy+1)*vw
    bottomright = (ix+1)%vw+(iy+1)*vw

    return np.array([[topleft,bottomright,topright],
                     [topleft,bottomleft,bottomright]], dtype=np.uint32)

def genFaces(w, h):
    topfaces = np.zeros((w*h*2,3),dtype=np.uint32)
    nexti = 0
    for x in range(w):
        for y in range(h):
            fs = getFaceSquare(x,y,w+1,h+1)
            topfaces[nexti]=fs[0]
            topfaces[nexti+1]=fs[1]
            nexti += 2
    
    # lasttop = faceVertices.shape[0]
    lasttop = (w+1)*(h+1)
    corners = [lasttop, lasttop+1, lasttop+2, lasttop+3, lasttop+4, lasttop+5, lasttop+6, lasttop+7]

    faces = np.array([[corners[4],corners[5],corners[6]], # -z
                      [corners[6],corners[7],corners[4]],
                      [corners[0],corners[1],corners[5]], # -y
                      [corners[5],corners[4],corners[0]],
                      [corners[1],corners[2],corners[6]], # +x
                      [corners[6],corners[5],corners[1]],
                      [corners[2],corners[3],corners[7]], # +y
                      [corners[7],corners[6],corners[2]],
                      [corners[3],corners[0],corners[4]], # -x
                      [corners[4],corners[7],corners[3]]], dtype=np.uint32)

    faces = np.concatenate((faces, topfaces), axis=0)
    return faces

def averageVertexValue(img, x, y):
    sample = 0
    total = [0,0,0]
    offsets = [(0,0),(-1,0),(-1,-1),(0,-1)]
    for offset in offsets:
        nx = x+offset[0]
        ny = y+offset[1]
        if nx < 0 or ny < 0 or nx >= img.shape[1] or ny >= img.shape[0]: continue
        sample += 1
        total[0] += img[ny,nx,0]
        total[1] += img[ny,nx,1]
        total[2] += img[ny,nx,2]
    return (total[0]/sample, total[1]/sample, total[2]/sample)

def calcVertexImageInfo(img, i, j, palette):
    paletteColors = [e[0] for e in palette]
    paletteLayerSize = [e[1] for e in palette[1:]]
    runningSum = np.cumsum([0]+paletteLayerSize)
    avgColor = np.array(averageVertexValue(img,i,j))
    index, ratio, err = getColorRatio(avgColor, paletteColors)
    value = np.array([index, ratio, err])
    color = getColor(index, ratio, paletteColors)
    # color = np.array(palette[round(index-1)])*(1-ratio)+np.array(palette[round(index)])*ratio
    blendLayers = round(ratio*paletteLayerSize[index-1])
    depth = (runningSum[index-1]+blendLayers)*LAYER_HEIGHT
    return value, color, depth

def calcVertexImageInfoColJob(img, i, vh, palette, values, colors, depths):
    for j in range(vh):
        values[j,i], colors[j,i], depths[j,i] = calcVertexImageInfo(img, i, j, palette)

def genImageInfo(img, vw, vh, palette):
    values = np.zeros((vh,vw,3))
    colors = np.zeros((vh,vw,3))
    depths = np.zeros((vh,vw))
    c = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(vw):
            futures.append(executor.submit(calcVertexImageInfoColJob, img, i, vh, palette, values, colors, depths))
        c = 0
        l = len(str(vw))
        print(f"[{" "*l}/{vw}]",end="")
        for j,future in enumerate(futures):
            future.result()
            c += 1
            print(f"\r[{c}{" "*(l-len(str(c)))}/{vw}]",end="")
        print(f"")
    return colors, depths, values

def calcVertexColors(img):
    vw, vh = img.shape[1]+1, img.shape[0]+1
    counts = np.full((vh,vw),4)
    counts[0,0] = counts[0,-1] = counts[-1,0] = counts[-1,-1] = 1
    counts[0,1:-1] = 2
    counts[-1,1:-1] = 2
    counts[1:-1,0] = 2
    counts[1:-1,-1] = 2
    h,w,_ = img.shape
    vertexColors = np.zeros((vh,vw,3))
    vertexColors[:h,:w] += img
    vertexColors[:h,1:w+1] += img
    vertexColors[1:h+1,1:w+1] += img
    vertexColors[1:h+1,:w] += img
    vertexColors /= np.repeat(counts[:,:,np.newaxis], 3, axis=2)
    return vertexColors

def calcVertexInfo(vertexColors, palette, layers):
    cumLayers = shift(np.cumsum(layers[1:]),1,cval=0)
    l1, l2 = palette[:-1], palette[1:]
    lowerIndex, minRatio, minErr = colorRatioSolverNP(vertexColors, l1, l2)
    upperIndex = lowerIndex+1
    values = np.stack((upperIndex, minRatio, minErr), axis=2)
    # ratioScale = 1-np.sqrt(-minRatio**2+1) # quarter circle 
    ratioScale = minRatio**2 # quadratic
    blendLayers = np.rint(layers[upperIndex]*ratioScale)
    depths = (cumLayers[lowerIndex]+blendLayers)*LAYER_HEIGHT
    ratioFull = np.repeat(minRatio[:,:,np.newaxis], 3, axis=2)
    colors = palette[upperIndex]*ratioFull+palette[lowerIndex]*(1-ratioFull)
    return colors, depths, values

def simplfyMesh(vertices, faces):
    simMesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(simMesh)

    ms.meshing_decimation_quadric_edge_collapse(
        preserveboundary=True, preservenormal=True, preservetopology=True, qualitythr=1,

    )

    simMesh = ms.current_mesh()
    simVertices = simMesh.vertex_matrix()   
    simFaces = simMesh.face_matrix()
    return simVertices, simFaces
    # return vertices, faces

def genMesh(w, h):
    vh, vw = h+1, w+1
    vertexColors = np.zeros((vw*vh+8, 3))
    faceVertices = np.zeros((vw*vh, 3))
    l = len(str(vw*vh))
    print(f"[{" "*l}/{vw*vh}]",end="")
    c = 0
    for i in range(vw):
        for j in range(vh):
            faceVertices[i%vw+j*vw] = np.array([i, j, depths[j,i]])
            vertexColors[i%vw+j*vw] = colors[j,i]
        c += vh
        if i % 10 == 0:
            print(f"\r[{c}{" "*(l-len(str(c)))}/{vw*vh}]",end="")
    print(f"\r[{vw*vh}/{vw*vh}]")
    firstColorHeight = palette[0][1]*LAYER_HEIGHT+INIT_LAYER_HEIGHT
    lowerVertices = np.array([[0,0,0],[w,0,0],[w,h,0],[0,h,0],[0,0,-firstColorHeight],[w,0,-firstColorHeight],[w,h,-firstColorHeight],[0,h,-firstColorHeight]])
    vertices = np.concatenate((faceVertices, lowerVertices), axis=0)
    # fill edges from bump to solid
    counter = 0
    vend = vertices.shape[0]
    topEdgeVertexIndices = np.zeros((vw+vw+vh+vh-4))
    edgeVertices = np.zeros((vw+vw+vh+vh-4,3))
    botEdgeVertexIndices = np.zeros((vw+vw+vh+vh-4))
    for y in range(vh):
        index = 0%vw+y*vw
        topEdgeVertexIndices[counter] = index
        edgeVertices[counter] = np.array([0,y,0])
        botEdgeVertexIndices[counter] = vend+counter
        counter += 1
    for x in range(1,vw-1):
        index = x%vw+(vh-1)*vw
        topEdgeVertexIndices[counter] = index
        edgeVertices[counter] = np.array([x,vh-1,0])
        botEdgeVertexIndices[counter] = vend+counter
        counter += 1
    for y in range(vh-1,-1,-1):
        index = (vw-1)%vw+y*vw
        topEdgeVertexIndices[counter] = index
        edgeVertices[counter] = np.array([vw-1,y,0])
        botEdgeVertexIndices[counter] = vend+counter
        counter += 1
    for x in range(vw-2,0,-1):
        index = x%vw+0*vw
        topEdgeVertexIndices[counter] = index
        edgeVertices[counter] = np.array([x,0,0])
        botEdgeVertexIndices[counter] = vend+counter
        counter += 1
    vertices = np.concatenate((vertices, edgeVertices), axis=0)
    vertexColors = np.concatenate((vertexColors,np.zeros((counter, 3))), axis=0)

    faces = genFaces(w,h)
    # calc faces for edges
    edgeLen = 2*w+2*h
    edgeFaces = np.zeros((edgeLen*2,3), dtype=np.uint32)
    for n in range(edgeLen):
        edgeFaces[2*n  ] = np.array([topEdgeVertexIndices[n],topEdgeVertexIndices[(n+1)%edgeLen],botEdgeVertexIndices[n]])
        edgeFaces[2*n+1] = np.array([botEdgeVertexIndices[n],botEdgeVertexIndices[(n+1)%edgeLen],topEdgeVertexIndices[(n+1)%edgeLen]])
    faces = np.concatenate((faces, edgeFaces), axis=0)
    return faces, vertices, vertexColors

def scaleVertices(vertices, maxW, maxH):
    width, height, depthMax = np.max(vertices, axis=0)
    scaleWidth = maxW/width
    scaleHeight = maxH/height
    scale = min(scaleWidth, scaleHeight)
    vertices[:,:2] = vertices[:,:2]*scale

def savePLY(filename, face, vertices, colors):
    plymesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertexColors)
    meshfileColor = f'res/{filename}_bump.ply'
    plymesh.export(meshfileColor)
    print(f"Saved ply to {meshfileColor}")

def saveSTL(filename, faces, vertices):
    stlmesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    meshfile = f'res/{filename}_bump.stl'
    for i,f in enumerate(faces):
        for j in range(3):
            stlmesh.vectors[i][j] = vertices[f[j],:]
    stlmesh.save(meshfile)
    print(f"Saved stl to {meshfile}")

def getPaletteTD(colors):
    print('Manual TD for each color:')
    newTD = np.zeros(colors.shape[0])
    for i in range(len(colors)):
        td = float(input(f'\tTD for {normColor2Hex(colors[i])}: '))
        newTD[i] = td
    return newTD

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image', help='path to image file', type=str, required=True)
    argparser.add_argument('-m', '--manual-td', help='manual transmission distance for each layer', required=False, action=argparse.BooleanOptionalAction)
    argparser.add_argument('-f', '--filaments', help='number of filaments', required=False, default=4, type=int)
    argparser.add_argument('-l', '--layer-height', help='height of layer', required=False, default=0.08, type=float)
    argparser.add_argument('-mw', '--max-width', help='max width of model in mm', required=False, default=63, type=float)
    argparser.add_argument('-mh', '--max-height', help='max height of model in mm', required=False, default=88, type=float)

    args = argparser.parse_args()

    LAYER_HEIGHT = args.layer_height
    MAX_HEIGHT = args.max_height
    MAX_WIDTH = args.max_width

    numFilaments = args.filaments
    path = args.image

    filename = os.path.basename(path).split('.')[0]
    img = cv2.imread(path)
    img = cv2.flip(img, 0)
    img = cv2.filter2D(img, -1, np.ones((3,3),dtype=np.float32)/9)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    h, w, _ = img.shape
    if w*h > 2000000: img = scaleImg(img, 2000000, interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape

    print("Generating palette...")
    filaments = getFilaments()
    layerColors = colorGradientMatch(img, filaments, maxFilaments=numFilaments, maxIterations=20, threshold=0.001)
    TDPerColor = np.full(layerColors.shape[0],0.32)
    if args.manual_td:
        TDPerColor = getPaletteTD(layerColors)
    palette = [(color.tolist(),round(TDPerColor[i]/LAYER_HEIGHT)) for i,color in enumerate(layerColors)]

    print("Processing image...")
    vertexColors = calcVertexColors(img)
    colors, depths, values = calcVertexInfo(vertexColors, layerColors, TDPerColor[:len(layerColors)]/LAYER_HEIGHT)

    print("Computing mesh...")
    faces, vertices, vertexColors = genMesh(w, h)
    scaleVertices(vertices, MAX_WIDTH, MAX_HEIGHT)

    print("Simplifying mesh...")
    # simVerts, simFaces = simplfyMesh(vertices, faces)
    simVerts, simFaces = vertices, faces
    print(f"Simplified {vertices.shape[0]} -> {simVerts.shape[0]} vertices")
    print(f"Simplified {faces.shape[0]} -> {simFaces.shape[0]} faces")

    print("Saving mesh...")
    savePLY(filename, faces, vertices, vertexColors)
    saveSTL(filename, simFaces, simVerts)

    print("")
    width, height, depthMax = np.max(vertices, axis=0)
    _, _, depthMin = np.min(vertices, axis=0)
    print(f"Layer height: {LAYER_HEIGHT}mm")
    print(f"Initial layer height: {INIT_LAYER_HEIGHT}mm")
    print(f"Print size: {round(width,2)} x {round(height,2)} x {round(depthMax-depthMin,2)} mm")
    cumLayer = np.cumsum([e[1] for e in palette])
    print("")
    print(f"Start color: {normColor2Hex(palette[0][0])}")
    for i in range(1, len(palette)):
        print(f"Filament change at layer {cumLayer[i-1]+2} to {normColor2Hex(palette[i][0])}")
