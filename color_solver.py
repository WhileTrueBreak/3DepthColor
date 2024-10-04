import os
import cv2
import csv
import sys
import math
import time
import numpy as np
import itertools
import colorsys
import argparse
from skimage import io, color
from hashlib import sha256
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt

def colorRatioSolver(color, p1, p2):

    r1,g1,b1 = color
    r2,g2,b2 = p1
    r3,g3,b3 = p2

    # solve ratio with min square sum error
    r23 = r3-r2
    g23 = g3-g2
    b23 = b3-b2
    sumSquareDiff = r23*r23+g23*g23+b23*b23
    ratio = 0
    if sumSquareDiff != 0:
        ratio = (r1*r23-r2*r23+g1*g23-g2*g23+b1*b23-b2*b23)/sumSquareDiff
        ratio = min(max(ratio,0),1)

    # get square sum error
    r_err = r1-ratio*r23-r2
    g_err = g1-ratio*g23-g2
    b_err = b1-ratio*b23-b2
    error = r_err*r_err+g_err*g_err+b_err*b_err

    return ratio, error

def colorRatioSolverNP(cSet, p1, p2, llr):
    lDiff = p2-p1
    cPoints = np.copy(p1[0])
    cSamples = p2[:,0]/llr+1 # num samples along line

    cSet = np.repeat(cSet[:,:,np.newaxis,:],p1.shape[0],axis=2)
    pl1Diff = cSet - p1[None,None,:,:]
    du = np.sum(lDiff**2, axis=1) # square dist
    avgHD = np.sqrt(du)/(4*cSamples) # average dist along line given samples

    nu = np.sum(pl1Diff*lDiff[None,None,:,:], axis=3)
    u = np.zeros(nu.shape)
    u = np.divide(nu, du, out=u, where=du!=0)
    ratio = np.clip(u, 0, 1) # ratio along line 
    distVec = pl1Diff-(ratio[:,:,:,None]*lDiff[None,None,:,:])
    sqDist = np.sum(distVec**2, axis=3)
    dist = np.sqrt(sqDist)
    adjDist = np.sqrt(avgHD[None,None,:]**2+sqDist) # adjusted distance via avg hoz dist

    mask = (u > 0) & (u < 1)
    adjDist = np.where(mask, adjDist, dist)

    minIndex = np.argmin(adjDist, axis=2)
    row = np.arange(minIndex.shape[0])[:,None]
    col = np.arange(minIndex.shape[1])
    minRatio = ratio[row,col,minIndex]
    minErr = adjDist[row,col,minIndex]

    # p21Diff = p2-p1
    # p21SquareDist = np.sum(p21Diff**2, axis=1)
    # cP = np.repeat(colors[:,:,np.newaxis,:],p21Diff.shape[0],axis=2)*p21Diff
    # p1P = p1*p21Diff
    # p1P = p1P.reshape(1,1,p1P.shape[0],p1P.shape[1])
    # p1P = np.tile(p1P, (cP.shape[0],cP.shape[1],1,1))
    # cp1PDiffSum = np.sum(cP-p1P, axis=3)

    # p21SquareDist = p21SquareDist.reshape(1,1,p21SquareDist.shape[0])
    # p21SquareDist = np.tile(p21SquareDist, (cp1PDiffSum.shape[0],cp1PDiffSum.shape[1],1))
    
    # ratio = np.zeros(cp1PDiffSum.shape)
    # ratio = np.divide(cp1PDiffSum, p21SquareDist, out=ratio, where=p21SquareDist!=0)
    # ratio = np.clip(ratio, 0, 1)

    # err = np.repeat(colors[:,:,np.newaxis,:],ratio.shape[2],axis=2)
    # p21Ratio = np.repeat(ratio[:,:,:,np.newaxis],p21Diff.shape[1],axis=3)*p21Diff
    # p1Full = p1.reshape(1,1,p1.shape[0],p1.shape[1])
    # p1Full = np.tile(p1Full, (err.shape[0],err.shape[1],1,1))
    # err = err-p21Ratio-p1Full
    # err = np.sum(err**2, axis=3)

    # minIndex = np.argmin(err, axis=2)
    # row = np.arange(minIndex.shape[0])[:,None]
    # col = np.arange(minIndex.shape[1])
    # minRatio = ratio[row,col,minIndex]
    # minErr = err[row,col,minIndex]
    return minIndex, minRatio, minErr

def getColorRatio(color, palette):
    bestRatio = -1
    bestError = float('inf')
    paletteIndex = 0
    # try all sequential colors
    for i in range(1, len(palette)):
        ratio, err = colorRatioSolver(color, palette[i-1], palette[i])
        if err < bestError:
            bestError = err
            bestRatio = ratio
            paletteIndex = i
    return paletteIndex, bestRatio, bestError

def getColor(index, ratio, palette):
    return np.array([
                palette[index][0]*ratio+palette[index-1][0]*(1-ratio),
                palette[index][1]*ratio+palette[index-1][1]*(1-ratio),
                palette[index][2]*ratio+palette[index-1][2]*(1-ratio)
            ])

def normalizeColorFromHex(hex):
    hex = hex.lstrip('#')
    r = int(hex[0:2], 16)/255
    g = int(hex[2:4], 16)/255
    b = int(hex[4:6], 16)/255
    return np.array([r,g,b])

def getFilaments():
    filaments = []
    with open('filaments.csv') as file:
        csvfile = csv.reader(file, delimiter=',')
        for filament in csvfile:
            filaments.append({'name':filament[0], 'color':filament[1], 'td':float(filament[2])})
    for i,f in enumerate(filaments):
        filaments[i]['color'] = normalizeColorFromHex(filaments[i]['color'])
    return filaments

def scaleImg(img, pixels, interpolation=cv2.INTER_NEAREST):
    currPixels = img.shape[0]*img.shape[1]
    scale = math.sqrt(pixels/currPixels)
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation = interpolation)

def avgError(pixels, palette):
    error = 0
    for i in range(pixels.shape[0]):
        index, ratio, err = getColorRatio(pixels[i], palette)
        error += err
    error /= pixels.shape[0]
    return error

def kCluster(pixels, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

def colorKClusterMatch(img, filaments, maxFilaments=4):
    img = scaleImg(img, 10000)
    colors = [filament['color'] for filament in filaments]
    pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    labPixels = np.apply_along_axis(color.rgb2lab, 1, pixels)

    centers = kCluster(labPixels, maxFilaments-2)
    centers = np.apply_along_axis(color.lab2rgb, 1, centers)

    closestFilament = lambda y: min(colors, key=lambda x: colorDiff(x, y))
    darkest = color.lab2rgb(min(labPixels, key=lambda x: x[0]))
    brightest = color.lab2rgb(max(labPixels, key=lambda x: x[0]))
    paletteColors = [darkest, brightest] + [closestFilament(center) for center in centers]
    selectedColors = [closestFilament(c) for c in paletteColors]
    uniqueColors = []
    for ccolor in selectedColors:
        if not any(np.array_equal(ccolor, unique) for unique in uniqueColors):
            uniqueColors.append(ccolor)
    bestError = float('inf')
    bestPalette = None
    for palette in itertools.permutations(uniqueColors):
        error = avgError(pixels, palette)
        if error >= bestError: continue
        bestError = error
        bestPalette = palette
    print(f'Generated colors with err: {bestError}')
    return bestPalette

def colorDiff(c1, c2):
    c1 = color.rgb2lab(c1)
    c2 = color.rgb2lab(c2)
    return np.linalg.norm(c1-c2)

def colorPlot(img, palette):
    pixels = getImgColors(img)

    pixelsLAB = np.apply_along_axis(color.rgb2lab, 1, pixels)
    palette = np.apply_along_axis(color.rgb2lab, 1, palette)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,100)
    ax.set_ylim(-128,127)
    ax.set_zlim(-128,127)
    # ax.scatter(pixels[:,0], pixels[:,1], pixels[:,2], c=pixels)
    ax.scatter(pixelsLAB[:,0], pixelsLAB[:,1], pixelsLAB[:,2], c=pixels, zorder=0)
    ax.scatter([c[0] for c in palette], [c[1] for c in palette], [c[2] for c in palette], color=(0,0,0), zorder=10)
    ax.plot([c[0] for c in palette], [c[1] for c in palette], [c[2] for c in palette], color=(0,0,0), zorder=10)
    plt.show()

def colorPlotPoints(img, points):
    pixels = getImgColors(img)
    pixelsLAB = np.apply_along_axis(color.rgb2lab, 1, pixels)
    points = np.apply_along_axis(color.rgb2lab, 1, points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,100)
    ax.set_ylim(-128,127)
    ax.set_zlim(-128,127)
    # ax.scatter(pixelsLAB[:,0], pixelsLAB[:,1], pixelsLAB[:,2], c=pixels, zorder=0)
    ax.scatter(points[:,0], points[:,1], points[:,2], color=(0,0,0), zorder=100000)
    plt.show()

def pointLineSegDist(p1, l1, l2):
    x1, y1, z1 = l1
    x2, y2, z2 = l2
    px, py, pz = p1
    dem = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    u = -1
    if dem != 0:
        u = ((px - x1) * (x2 - x1) + 
            (py - y1) * (y2 - y1) + 
            (pz - z1) * (z2 - z1)) / dem
    if u <= 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2 + (pz - z1)**2)
    elif u >= 1:
        return np.sqrt((px - x2)**2 + (py - y2)**2 + (pz - z2)**2)
    else:
        return np.sqrt(((px - x1) - u * (x2 - x1))**2 + 
                       ((py - y1) - u * (y2 - y1))**2 + 
                       ((pz - z1) - u * (z2 - z1))**2)

def avgPDist(cSet, points):
    totalDist = 0
    for c in cSet:
        bestDist = float('inf')
        for i in range(len(points)-1):
            dist = pointLineSegDist(c, points[i], points[i+1])
            if dist >= bestDist: continue
            bestDist = dist
        totalDist += bestDist
    avgDist = totalDist / len(cSet)
    return avgDist

def avgPDistNP(cSet, points):
    l2 = points[1:]
    l1 = points[:-1]
    lDiff = l2-l1
    pl1Diff = cSet[:,None] - l1[None,:]
    du = np.sum(lDiff**2, axis=1)
    nu = np.sum(pl1Diff*lDiff, axis=2)
    u = np.zeros(nu.shape)
    u = np.divide(nu, du, out=u, where=du!=0)
    u = np.clip(u, 0, 1)
    u = np.repeat(u[:,:,np.newaxis], 3, axis=2)
    distVec = pl1Diff-u*lDiff
    dist = np.min(np.sqrt(np.sum(distVec**2, axis=2)), axis=1)
    avgDist = np.average(dist) # actual thing to solve for
    pDistFactor = np.sum(np.sqrt(du))/100 # prioritise P which form a shorter line
    lightnessIncreaseFactor = np.average(-lDiff[:,0])/20 # prioritise layers from dark to light
    # pDist = np.sqrt(np.sum(lDiff**2, axis=1))
    return avgDist+pDistFactor+lightnessIncreaseFactor

def layerObjective(cSet, points, llr):
    l2 = points[1:]
    l1 = points[:-1]
    lDiff = l2-l1
    cPoints = np.copy(l1[0])
    cSamples = l2[:,0]/llr+1 # num samples along line

    pl1Diff = cSet[:,None] - l1[None,:]
    du = np.sum(lDiff**2, axis=1) # square dist

    nu = np.sum(pl1Diff*lDiff, axis=2)
    u = np.zeros(nu.shape)
    u = np.divide(nu, du, out=u, where=du!=0)
    uClip = np.clip(u, 0, 1) # ratio along line 
    distVec = pl1Diff-uClip[:,:,None]*lDiff
    sqDist = np.sum(distVec**2, axis=2)

    # df2 = lambda r:3*r**2-6*r+3
    uAjd = 3*uClip**2-6*uClip+3

    avgHD = np.sqrt(du)/(4*cSamples) # average dist along line given samples
    avgHD = avgHD[None,:]
    # print(uAjd)
    # print(avgHD)
    adjDist = np.sqrt(avgHD**2+sqDist) # adjusted distance via avg hoz dist
    # print(adjDist)


    # print(u.shape)
    # print(uAjd.shape)
    # print(avgHD.shape)

    # exit()

    minDistArg = np.argmin(adjDist, axis=1)
    minDist = adjDist[np.arange(len(minDistArg)),minDistArg]
    return np.average(minDist)

def getImgColors(img, numColors=1000):
    kImg = scaleImg(img, numColors*100)
    imgColors = kImg.reshape(-1,3)

    with threadpool_limits(limits=1):
        kmeans = KMeans(n_clusters=numColors, random_state=42)
        kmeans.fit(imgColors)
        colors = kmeans.cluster_centers_
    sortIndex = np.argsort(colors[:,0])
    colors = colors[sortIndex,:]
    colors = np.clip(colors, 0, 1)
    return colors

def colorGradientMatch(img, filaments, llr, maxFilaments=4, maxIterations=10, threshold=0.001, change_threshold=1e-10):
    colors = getImgColors(img)
    colors = np.apply_along_axis(color.rgb2lab, 1, colors)

    # objective = lambda x: avgPDistNP(colors, x.reshape(-1,3))
    objective = lambda x: layerObjective(colors, x.reshape(-1,3), llr)
    randLAB = lambda x: np.array([np.random.uniform(0,100), np.random.uniform(-128,127), np.random.uniform(-128,127)])

    initGuess = np.repeat([[50,0,0]], maxFilaments, axis=0)
    cycleInitGuess = np.copy(initGuess)
    bounds = np.array([[0,100],[-128,127],[-128,127]]*maxFilaments)

    bestResult = None
    bestErr = float('inf')

    np.random.seed(42)

    counter = 0
    result = None
    prevErr = float('inf')
    l = len(str(maxIterations))
    print(f"Optimizing [{" "*(l-1)}{counter+1}/{maxIterations}]", end='', flush=True)
    while counter < maxIterations and (result == None or result.success == False or result.fun > threshold):
        if result:
            if prevErr - result.fun < change_threshold:
                if result.fun < bestErr:
                    bestResult = result
                    bestErr = result.fun
                cycleInitGuess = np.apply_along_axis(randLAB,1,np.copy(initGuess))
                result = None
                prevErr = float('inf')
                # break
        prevErr = result.fun if result != None else float('inf')
        sys.stderr = open(os.devnull, 'w')
        resultSLSQP = minimize(objective, np.copy(cycleInitGuess.flatten()), method="SLSQP", bounds=bounds, tol=0)
        sys.stderr = sys.__stderr__
        cycleInitGuess = resultSLSQP.x
        result = resultSLSQP
        print(f"\rOptimizing [{" "*(l-len(str(counter+1)))}{counter+1}/{maxIterations}]", end='', flush=True)
        counter += 1

    print("")
    
    avgPDistNP(colors, bestResult.x.reshape(-1,3))

    print(f"Done with sample err: {bestResult.fun}")
    path = bestResult.x.reshape(-1,3)
    path = np.apply_along_axis(color.lab2rgb, 1, path)
    return path

def normColor2Hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image', help='path to image file', type=str, required=True)
    argparser.add_argument('-f', '--filaments', help='number of filaments', required=False, default=4, type=int)
    args = argparser.parse_args()

    numFilaments = args.filaments
    path = args.image

    filaments = getFilaments()
    img = cv2.imread(path)
    img = cv2.flip(img, 0)
    img = cv2.filter2D(img, -1, np.ones((3,3),dtype=np.float32)/9)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    h, w, _ = img.shape
    if w*h > 2000000: img = scaleImg(img, 2000000, interpolation=cv2.INTER_AREA)

    path = colorGradientMatch(img, filaments, maxFilaments=numFilaments, maxIterations=40, threshold=0.001)
    colorPlot(img, path)

    # colors = getImgColors(img)
    # colors = np.apply_along_axis(color.rgb2lab, 1, colors)
    # path = np.apply_along_axis(color.rgb2lab, 1, path)
    # layerObjective(colors, path, 10)

    # [print(normColor2Hex(c)) for c in path]

