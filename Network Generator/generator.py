__author__ = 'hamed'
import sys
import math

# import xlwt
import os
import glob
from math import log
import networkx as nx

import random

# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy import stats
# import scipy.sparse as sp

import csv
import numpy as np
import numpy.random as rand


def Multi_deg_generator():
    # Number of nodes
    N = 1200
    # Number of layers
    M = 3
    # Number of comminuties
    Q = 12
    # density of one partite layers
    rokolmm = 0.6
    # density of bipartite layers
    densml = 0.05
    # cross layer community factor
    alpha = 0.5
    DegreeCorrect = 1
    # fraction of links connect inside layer communities
    mu = np.zeros(M, dtype=np.float)
    mu[0] = 4
    mu[1] = 4
    mu[2] = 4
    # fraction of links connect inside cross layer communities
    pi = np.zeros((M, M), dtype=np.float)
    pi[0, 1] = 4
    pi[0, 2] = 4
    pi[1, 2] = 4
    pi = pi + pi.transpose()
    # number of community in each layer
    QM = np.zeros(M, dtype=np.int)
    QM[0] = 4
    QM[1] = 4
    QM[2] = 4
    # Node types
    commufactor=np.zeros(QM[0])
    commufactor[0] = 1
    commufactor[1] = 1
    commufactor[2] = 1
    commufactor[3] = 1

    types = np.zeros(N, dtype=int)

    for i in range(600, 1000):
        types[i] = 1
    for i in range(1000, N):
        types[i] = 2

    tyu = []
    tyu1 = ['N', str(N)]
    tyu.append(tyu1)
    tyu1 = ['M', str(M)]
    tyu.append(tyu1)
    tyu1 = ['Q', str(Q)]
    tyu.append(tyu1)
    tyu1 = ['densml', str(densml)]
    tyu.append(tyu1)
    tyu1 = ['alpha', str(alpha)]
    tyu.append(tyu1)
    tyu1 = ['mu0', str(mu[0])]
    tyu.append(tyu1)
    tyu1 = ['mu1', str(mu[1])]
    tyu.append(tyu1)
    tyu1 = ['mu2', str(mu[2])]
    tyu.append(tyu1)
    tyu1 = ['pi01', str(pi[0, 1])]
    tyu.append(tyu1)
    tyu1 = ['pi02', str(pi[0, 2])]
    tyu.append(tyu1)
    tyu1 = ['pi12', str(pi[1, 2])]
    tyu.append(tyu1)
    tyu1 = ['QM0', str(QM[0])]
    tyu.append(tyu1)
    tyu1 = ['QM1', str(QM[1])]
    tyu.append(tyu1)
    tyu1 = ['QM2', str(QM[2])]
    tyu.append(tyu1)
    tyu1 = ['rokolmm ', str(rokolmm)]
    tyu.append(tyu1)

    np.set_printoptions(suppress=True)

    np.savetxt('variables.csv', tyu, fmt='%s', delimiter=',')
    # with open('DataTypeA1000.csv', mode='w') as employee_file:
    # employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    # employee_writer.writerow(types)

    np.savetxt('DataType.csv', types, fmt='%d', delimiter=',')
    # np.save('DataTypeA100.npy', types)

    T = np.zeros(M, dtype=int)
    for i in range(0, M):
        T[i] = len(np.where(types == i)[0])

    G = {}
    GM = {}
    QMM = np.cumsum(QM)
    #gama = rand.rand() + 2
    #d = nx.utils.powerlaw_sequence(int(QM[0]), gama)
    #d = np.asarray(d)
    #d = d * (T[0] / sum(d))
    poi=0
    for uui in range(0,len(commufactor)):
        poi += commufactor[uui]

    d = np.zeros(QM[0])
    for uui in range(0,len(commufactor)):
        d[uui] += T[0]*(commufactor[uui]/poi)


    d = np.round(d)
    d = np.cumsum(d)
    if d[int(QM[0]) - 1] < T[0]:
        d[int(QM[0]) - 1] += (T[0] - d[int(QM[0]) - 1])
    elif d[int(QM[0]) - 1] > T[0]:
        d[int(QM[0]) - 1] -= (d[int(QM[0]) - 1] - T[0])

    g = np.zeros(T[0], dtype=np.int)
    for i in range(0, int(d[0])):
        g[i] = 0
    for j in range(1, int(QM[0])):
        for i in range(int(d[j - 1]), int(d[j])):
            g[i] = j
    G[0] = g

    for m in range(1, M):
        #gama = rand.rand() + 2
        #d = nx.utils.powerlaw_sequence(int(QM[m]), gama)
        #d = np.asarray(d)
        #d = d * (T[m] / sum(d))
        poi = 0
        for uui in range(0, len(commufactor)):
            poi += commufactor[uui]

        d = np.zeros(QM[m])
        for uui in range(0, len(commufactor)):
            d[uui] += T[m] * (commufactor[uui] / poi)

        d = np.round(d)
        d = np.cumsum(d)
        if d[int(QM[m]) - 1] < T[m]:
            d[int(QM[m]) - 1] += (T[m] - d[int(QM[m]) - 1])
        elif d[int(QM[m]) - 1] > T[m]:
            d[int(QM[m]) - 1] -= (d[int(QM[m]) - 1] - T[m])

        g = np.zeros(T[m], dtype=np.int)
        for i in range(0, int(d[0])):
            g[i] = QMM[m - 1]
        for j in range(1, int(QM[m])):
            for i in range(int(d[j - 1]), int(d[j])):
                g[i] = j + QMM[m - 1]
        G[m] = g

    print(G)
    GM = {}
    for m in range(0, M):
        GM[m] = set(np.unique(G[m]))

    print(GM)

    for l in range(0, M):
        opl = []
        opm = []
        iool = set()
        for m in range(0, l):
            k = QM[m]
            if k > QM[l]:
                k = QM[l]
            k = math.floor(k * alpha)
            k = k - len(GM[m] & GM[l])
            ioo = GM[m] & GM[l]
            GM[m] = GM[m] - (ioo)
            GM[l] = GM[l] - (ioo)
            j = 0
            kkey = 0
            ioom = set()

            while j<k:

                if GM[m] != set():
                    kj = random.sample(GM[m], 1)

                    if kj[0] not in opm:
                       opm.append(kj[0])
                    else:
                        kkey = 1
                        j = j - 1
                        sss = set()
                        sss.add(kj[0])
                        ioom = ioom | sss
                        GM[m] = GM[m] - sss
                else:
                    break

                if GM[l] != set():
                  if kkey==0:
                    kjj = random.sample(GM[l], 1)
                    if kjj[0] not in opl:
                       opl.append(kjj[0])
                    else:
                        j = j - 1
                        del opm[-1]
                        sss = set()
                        sss.add(kjj[0])
                        iool = iool | sss
                        GM[l] = GM[l] - sss
                else:
                    break

                kkey = 0
                j = j +1

            GM[m] = GM[m] | (ioo) | ioom
            GM[l] = GM[l] | (ioo)

        GM[l] = GM[l] | iool
        for ij in range(0,len(opl)):
            G[l][np.where(G[l] == opl[ij])[0]] = opm[ij]



        GM[l] = GM[l] - set(opl)
        GM[l] = GM[l] | set(opm)

    print(G)
    print(GM)
    maxx2 = min(GM[0])

    for m in range(0, M):
        for i in GM[m]:
            if i > maxx2:
                if i - maxx2 > 1:
                    opm = set()
                    opm.add(i)
                    GM[m] = GM[m] - opm
                    opl = set()
                    opl.add(maxx2 + 1)
                    GM[m] = GM[m] | opl
                    G[m][np.where(G[m] == i)[0]] = maxx2 + 1
                    for l in range(m + 1, M):
                        if i in GM[l]:
                            GM[l] = GM[l] - opm
                            GM[l] = GM[l] | opl
                            G[l][np.where(G[l] == i)[0]] = maxx2 + 1

                    maxx2 = maxx2 + 1
                else:
                    maxx2 = i


    print(G)
    print(GM)
    myData = []
    for m in range(0, M):

        for mm in range(0, int(T[m])):
            myData.append(G[m][mm])

    Q = len(np.unique(myData))

    np.savetxt('DataCommualpha=0.5.csv', myData, fmt='%d', delimiter=',')
    # Define a member list for each community
    members = {}
    for uu in range(0, M):
        for j in range(0, Q):
            members[uu, j] = np.where(G[uu] == j)[0]

    print(members)
    # Block structure matrix, called OMEGA PLANTED in the paper.
    ro = np.zeros((M, M))
    for m in range(0, M):
        ro[m, m] = rokolmm * (T[m] - 1) * T[m]
    for m in range(0, M):
        for l in range(0, M):
            if m != l:
                ro[m, l] = densml * T[l] * T[m]


    s = np.zeros((M, M, Q, Q))

    for m in range(0, M):
        for l in range(m, M):
            if m == l:


                m2=(rokolmm*T[m]*(T[m]+1))
                for i in GM[m]:
                    for j in GM[l]:
                        if i==j:
                            s[m,l,i,j]=(mu[m]*(m2/(QM[m])))/(mu[m]+QM[m]-1)
                        else:
                            s[m,l,i,j]=((m2/(2*QM[m])))/(mu[m]+QM[m]-1)


            else:
                m2=2*densml*T[m]*T[l]
                for i in GM[m] & GM[l]:

                    s[m,l,i,i]=(pi[m, l]*(m2/(2*QM[m])))/(pi[m, l]+QM[m]-1)


                    s[l,m,i,i]=s[m,l,i,i]

                for i in GM[m] & GM[l]:
                    for j in GM[l]:
                        if j!=i:
                             s[m,l,i,j]=((m2/(2*QM[m])))/(pi[m, l]+QM[m]-1)
                             s[l,m,j,i]=s[m,l,i,j]

                    for j in GM[m]:
                        if j!=i:
                             s[l,m,i,j]=((m2/(2*QM[m])))/(pi[m, l]+QM[m]-1)
                             s[m,l,j,i]=s[l,m,i,j]

                for i in GM[m]:
                    for j in GM[l]:
                        if i not in GM[m] & GM[l] and j not in GM[m] & GM[l]:
                            s[m,l,i,j]=pow((m2/(2*QM[m])),2)/m2
                            s[l,m,j,i]=s[m,l,i,j]


    gama = rand.rand() + 2
    lamb = np.arange(0, 1.1, 0.1)
    print(s)
    dd = {}
    X = np.zeros((N, N),dtype=np.int)
    for m in range(0, M):
        for l in range(m, M):
            if m != l:

                print()

                # j1 = ro[m,l]
                # j2 = ro[l,m]
                # if j2 < j1:
                # ui = j1
                # j1 = j2
                # j2 = ui

                # d = (j2 - j1) * np.floor(2 * rand.rand(int(T[m]) + int(T[l]))) + j1

                #key = 0
                #qq = 0
                #while key == 0 and qq < 100000000000000000000:
                    #gama = rand.rand() + 2
                    #d1 = nx.utils.powerlaw_sequence((int(T[m])), gama)
                    #d1 = np.asarray(d1)
                    #dm = (ro[m, l]) / ((int(T[m])))
                    #d1 = d1 * (dm / (sum(d1) / (int(T[m]))))
                    #gama = rand.rand() + 2
                    #d2 = nx.utils.powerlaw_sequence((int(T[l])), gama)
                    #d2 = np.asarray(d2)
                    #dm = (ro[m, l]) / ((int(T[l])))
                    #d2 = d2 * (dm / (sum(d2) / (int(T[l]))))
                    #if max(d1) <= T[l] and max(d2) <= T[m]:
                        #key = 1
                    #qq = qq + 1

                gama = rand.rand() + 2
                d1 = nx.utils.powerlaw_sequence((int(T[m])), gama)
                d1 = np.asarray(d1)
                gama = rand.rand() + 2
                d2 = nx.utils.powerlaw_sequence((int(T[l])), gama)
                d2 = np.asarray(d2)

                d = np.zeros(int(T[m])+int(T[l]))
                for qqq in range(0,int(T[m])):
                    d[qqq] = d1[qqq]
                for qqq in range(0,int(T[l])):
                    d[qqq+int(T[m])] = d2[qqq]
                ffff=2*densml*T[m]*T[l]
                dddd=sum(d)
                d=d*(ffff/dddd)
                np.set_printoptions(suppress=True)
                for i in GM[m] & GM[l]:
                    t11 = np.where(G[m] == i)[0]
                    t22 = np.where(G[l] == i)[0] + int(T[m])
                    t = np.zeros(len(t11)+len(t22),dtype=np.int)
                    for j in range(0,len(t11)):
                        t[j] = t11[j]
                    for j in range(0,len(t22)):
                        t[j+len(t11)] = t22[j]

                    d[t] = d[t] / sum(d[t])

                #dd[m, l] = d




                for i in GM[m]:
                  if i not in GM[m]&GM[l]:
                    t = np.where(G[m] == i )[0]

                    d[t] = d[t] / sum(d[t])


                for i in GM[l]:
                  if i not in GM[m] & GM[l]:
                    t = np.where(G[l] == i)[0] + int(T[m])
                    #t = np.where(G[l] == i)[0]

                    d[t] = d[t] / sum(d[t])



                w = np.zeros((Q, Q))
                for i in GM[m]:
                    for j in GM[l]:
                        w[i, j] = rand.poisson(s[m, l, i, j])
                        if w[i, j] > (len(members[m, i]) * len(members[l, j])):
                            w[i, j] = len(members[m, i]) * len(members[l, j])
                ww = np.zeros((Q, Q))
                for i in GM[l]:
                    for j in GM[m]:
                        ww[i, j] = rand.poisson(s[l, m, i, j])
                        if ww[i, j] > (len(members[l, i]) * len(members[m, j])):
                            ww[i, j] = len(members[l, i]) * len(members[m, j])
                print(w)
                r = []
                c = []
                for i in GM[m]:
                    for j in GM[l]:
                      if i!=j:
                          if i in GM[m]&GM[l] or j in GM[m]&GM[l]:

                              R = []
                              C = []
                              tedad = w[i, j]+ww[i,j]
                              while tedad > 0:

                                  # azz = np.cumsum(dd[m, l][members[m, i]])
                                  zaq = np.zeros(len(members[m, i]) + len(members[l, i]))
                                  for ilmn in range(0, len(members[m, i])):
                                      zaq[ilmn] = d[members[m, i][ilmn]]
                                  for ilmn in range(0, len(members[l, i])):
                                      zaq[ilmn + len(members[m, i])] = d[members[l, i][ilmn] + int(T[m])]

                                  azz = np.cumsum(zaq)

                                  dice = rand.rand(int(tedad))

                                  for t in range(0, len(dice)):

                                      for y in range(0, len(azz)):

                                             if azz[y] > dice[t]:
                                                 if y < len(members[m, i]):
                                                       R.append(members[m, i][y])
                                                       break



                                  # to = np.cumsum(dd[m, l][members[l, j]])
                                  zaq = np.zeros(len(members[m, j]) + len(members[l, j]))
                                  for ilmn in range(0, len(members[m, j])):
                                      zaq[ilmn] = d[members[m, j][ilmn]]
                                  for ilmn in range(0, len(members[l, j])):
                                      zaq[ilmn + len(members[m, j])] = d[members[l, j][ilmn] + int(T[m])]

                                  to = np.cumsum(zaq)

                                  dice = rand.rand(int(tedad))

                                  for t in range(0, len(dice)):

                                      for y in range(0, len(to)):
                                          if to[y] > dice[t]:
                                              if y >= len(members[m, j]):
                                                  C.append(members[l, j][y- len(members[m, j])])
                                                  break


                                  minn = len(R)
                                  keyy = 0
                                  if len(C) < minn:
                                      minn = len(C)
                                      keyy = 1
                                  if len(R) != len(C):
                                      if keyy == 0:
                                          jklo = len(C) - minn
                                          del C[-jklo:]
                                      else:
                                          jklo = len(R) - minn
                                          del R[-jklo:]

                                  R = np.array(R)
                                  C = np.array(C)
                                  len1 = len(R)
                                  kabiii = []

                                  for tt in range(0, len(R) - 1):

                                      kr = np.where(R[tt + 1:] == R[tt])[0] + tt + 1
                                      kc = np.where(C[tt + 1:] == C[tt])[0] + tt + 1
                                      inter = np.intersect1d(kr, kc)
                                      for uuu in range(0, len(inter)):
                                          kabiii.append(inter[uuu])

                                  R = np.delete(R, kabiii)
                                  C = np.delete(C, kabiii)
                                  len2 = len(R)
                                  R = R.tolist()
                                  C = C.tolist()

                                  tedad = len1 - len2

                              for u in range(0, len(R)):
                                  r.append(R[u])
                              for u in range(0, len(C)):
                                  c.append(C[u])



                          else:

                               R = []
                               C = []
                               tedad = w[i, j]
                               while tedad > 0:

                                       #azz = np.cumsum(dd[m, l][members[m, i]])
                                       azz = np.cumsum(d[members[m, i]])

                                       dice = rand.rand(int(tedad))

                                       for t in range(0, len(dice)):

                                            for y in range(0, len(azz)):

                                                 if azz[y] > dice[t]:
                                                      R.append(members[m, i][y])
                                                      break

                                       #to = np.cumsum(dd[m, l][members[l, j]])
                                       to = np.cumsum(d[members[l, j]+ int(T[m])])

                                       dice = rand.rand(int(tedad))

                                       for t in range(0, len(dice)):

                                            for y in range(0, len(to)):
                                                 if to[y] > dice[t]:
                                                      C.append(members[l, j][y])
                                                      break


                                       R = np.array(R)
                                       C = np.array(C)
                                       len1 = len(R)
                                       kabiii = []

                                       for tt in range(0, len(R)-1):

                                               kr = np.where(R[tt + 1:] == R[tt])[0] + tt + 1
                                               kc = np.where(C[tt + 1:] == C[tt])[0] + tt + 1
                                               inter = np.intersect1d(kr, kc)
                                               for uuu in range(0, len(inter)):
                                                     kabiii.append(inter[uuu])



                                       R = np.delete(R, kabiii)
                                       C = np.delete(C, kabiii)
                                       len2 = len(R)
                                       R = R.tolist()
                                       C = C.tolist()


                                       tedad = len1 - len2

                               for u in range(0, len(R)):
                                       r.append(R[u])
                               for u in range(0, len(C)):
                                       c.append(C[u])
                      else:

                          R = []
                          C = []
                          tedad = w[i, j]

                          zaq = np.zeros(len(members[m, i])+len(members[l, j]))
                          while tedad > 0:
                              for ilmn in range(0,len(members[m, i])):
                                 zaq[ilmn] = d[members[m, i][ilmn]]
                              for ilmn in range(0,len(members[l, j])):
                                 zaq[ilmn+len(members[m, i])] = d[members[l, j][ilmn]+ int(T[m])]

                              azz = np.cumsum(zaq)

                              dice = rand.rand(int(tedad))

                              for t in range(0, len(dice)):

                                  for y in range(0, len(azz)):

                                      if azz[y] > dice[t]:
                                          if y<len(members[m, i]):
                                              R.append(members[m, i][y])
                                              break


                              to = np.cumsum(zaq)

                              dice = rand.rand(int(tedad))

                              for t in range(0, len(dice)):

                                  for y in range(0, len(to)):
                                      if to[y] > dice[t]:
                                          if y>=len(members[m, i]):

                                              C.append(members[l, j][y - len(members[m, i])])
                                              break


                              minn = len(R)
                              keyy = 0
                              if len(C) < minn:
                                  minn = len(C)
                                  keyy = 1
                              if len(R)!=len(C):
                                 if keyy == 0:
                                      jklo = len(C)-minn
                                      del C[-jklo:]
                                 else:
                                      jklo = len(R) - minn
                                      del R[-jklo:]



                              R = np.array(R)
                              C = np.array(C)
                              len1 = len(R)
                              kabiii = []

                              for tt in range(0, len(R)-1):
                                  kr = np.where(R[tt + 1:] == R[tt])[0] + tt + 1
                                  kc = np.where(C[tt + 1:] == C[tt])[0] + tt + 1
                                  inter = np.intersect1d(kr, kc)
                                  for uuu in range(0, len(inter)):
                                      kabiii.append(inter[uuu])

                                  kr = np.where(R[tt + 1:] == C[tt])[0] + tt + 1
                                  kc = np.where(C[tt + 1:] == R[tt])[0] + tt + 1
                                  inter = np.intersect1d(kr, kc)
                                  for uuu in range(0, len(inter)):
                                      kabiii.append(inter[uuu])



                              R = np.delete(R, kabiii)
                              C = np.delete(C, kabiii)
                              len2 = len(R)
                              R = R.tolist()
                              C = C.tolist()
                              tedad = len1 - len2

                          for u in range(0, len(R)):
                              r.append(R[u])
                          for u in range(0, len(C)):
                              c.append(C[u])

                A = np.zeros((int(T[m]), int(T[l])),dtype=np.int)


                for hh in range(0, len(r)):
                    A[r[hh], c[hh]] = 1

                xxm = 0
                for uu in range(0, m):
                    xxm += T[uu]
                xxl = 0
                for uu in range(0, l):
                    xxl += T[uu]
                for mm in range(0, int(T[m])):
                    for ll in range(0, int(T[l])):
                        X[mm + int(xxm), ll + int(xxl)] = A[mm, ll]

                B = np.transpose(A)
                for ll in range(0, int(T[l])):
                    for mm in range(0, int(T[m])):
                        X[ll + int(xxl), mm + int(xxm)] = B[ll, mm]


            else:

                #key = 0
                #qq = 0
                #while key == 0 and qq < 100000000000000000000:
                    #gama = rand.rand() + 2
                    #d = nx.utils.powerlaw_sequence(int(T[m]), gama)
                    #d = np.asarray(d)
                    #dm = ro[m, l] / (int(T[m]))
                    #d = d * (dm / (sum(d) / T[m]))
                    #if max(d) <= T[m]:
                        #key = 1
                    #qq = qq + 1

                gama = rand.rand() + 2
                d = nx.utils.powerlaw_sequence(int(T[m]), gama)
                d = np.asarray(d)
                ffff=rokolmm*T[m]*(T[m]+1)
                dddd=sum(d)
                d=d*(ffff/dddd)
                #j1 = 50
                #j2 = 30
                #if j2 < j1:
                    #ui = j1
                    #j1 = j2
                    #j2 = ui

                #d = (j2 - j1) * np.floor(2 * rand.rand(int(T[m]))) + j1


                for i in GM[m]:
                    # t = find(g==i)
                    t = np.where(G[m] == i)[0]

                    d[t] = d[t] / sum(d[t])

                #dd[m, l] = d


                w = np.zeros((Q, Q))
                for i in range(0, Q):
                    for j in range(i, Q):

                        w[i, j] = rand.poisson(s[m, l, i, j])
                        if w[i, j] > (len(members[m, i]) * len(members[l, j])):
                            w[i, j] = len(members[m, i]) * len(members[l, j])


                w = w + np.transpose(w)
                for i in range(0, Q):
                    w[i, i] = w[i, i] / 4


                print(w)
                r = []
                c = []
                for i in range(0, Q):
                    for j in range(i, Q):

                        R = []
                        C = []
                        tedad = w[i, j]
                        while tedad > 0:


                            azz = np.cumsum(d[members[m, i]])

                            dice = rand.rand(int(tedad))

                            for t in range(0, len(dice)):

                                for y in range(0, len(azz)):

                                    if azz[y] > dice[t]:
                                        R.append(members[m, i][y])
                                        break

                            to = np.cumsum(d[members[l, j]])

                            dice = rand.rand(int(tedad))

                            for t in range(0, len(dice)):

                                for y in range(0, len(to)):
                                    if to[y] > dice[t]:
                                        C.append(members[l, j][y])
                                        break


                            R = np.array(R)
                            C = np.array(C)
                            len1 = len(R)
                            kabiii = []

                            for tt in range(0, len(R)-1):
                                kr = np.where(R[tt + 1:] == R[tt])[0] + tt + 1
                                kc = np.where(C[tt + 1:] == C[tt])[0] + tt + 1
                                inter = np.intersect1d(kr, kc)
                                for uuu in range(0, len(inter)):
                                    kabiii.append(inter[uuu])

                                kr = np.where(R[tt + 1:] == C[tt])[0] + tt + 1
                                kc = np.where(C[tt + 1:] == R[tt])[0] + tt + 1
                                inter = np.intersect1d(kr, kc)
                                for uuu in range(0, len(inter)):
                                    kabiii.append(inter[uuu])

                            R = np.delete(R, kabiii)
                            C = np.delete(C, kabiii)
                            len2 = len(R)
                            R = R.tolist()
                            C = C.tolist()
                            tedad = len1 - len2

                        for u in range(0, len(R)):
                                r.append(R[u])
                        for u in range(0, len(C)):
                                c.append(C[u])





                A = np.zeros((int(T[m]), int(T[l])),dtype=np.int)


                for hh in range(0, len(r)):

                        A[r[hh], c[hh]] = +1

                A = A + np.transpose(A)


                for mm in range(0, int(T[m])):
                    #for ll in range(0, int(T[l])):
                        if A[mm, mm] == 2:
                            A[mm, mm] = 1

                xxm = 0
                for uu in range(0, m):
                    xxm += T[uu]
                xxl = 0
                for uu in range(0, l):
                    xxl += T[uu]
                for mm in range(0, int(T[m])):
                    for ll in range(0, int(T[l])):
                        X[mm + int(xxm), ll + int(xxl)] = A[mm, ll]


    for hh in range(0,N):
        X[hh,hh]=0

    np.savetxt('Data2.csv', X, fmt='%d', delimiter=',')
    # np.save('Data' + str(ee) + 'A1000.npy', X)
    # print('Network %i of %i created. lambda = %f.\n' % (ee, len(lamb), lamb2))
    print(GM)

    pppp=0
    for hh in range(0,N):
        for opp in range(0,N):
            pppp= pppp+X[hh,opp]
    print(pppp/(N*(N+1)))

    return M, Q, DegreeCorrect


Multi_deg_generator()

