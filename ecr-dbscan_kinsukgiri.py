import heapq
import sys
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
import heapq
from scipy.spatial import distance
import math
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import math
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from sklearn.metrics import accuracy_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import davies_bouldin_score
main = sys.stdout
dim = 0 # 0 for 2d and 1 for 3d
pc = 0
ep = input("Give 1 for ECs and 2 for KNN  :  ")
ep = int(ep) #1 for ECs and 2 for KNNs
tp = 1
coldata = (0,1)
coltar = (0)
state = 47
ds = 400
attr = 2
blob = 7
runar = list(range(2,15))
#Reading Data from an External Datafile #delimiter = ','
#x = load_breast_cancer().data
x=np.loadtxt(fname = "S1.txt",usecols = coldata)
target=np.loadtxt(fname = "s1-label.txt",usecols = coltar)
                     #for your run, give your own path
target = target-1
#target = load_breast_cancer().target
#x, y = make_blobs(n_samples = ds, centers = blob, n_features=attr, shuffle=True, random_state=state)
#x, y = make_moons(n_samples = ds, shuffle=True, random_state=state)
poin = np.array(x)
scaler = StandardScaler()
scaler.fit(poin)
poin1 = scaler.fit_transform(poin)
if pc == 1:
      pca = PCA(n_components = 2)
      pca.fit(poin1)
      points = pca.fit_transform(poin1)
else:
      points = poin
points1 = points

if dim == 0:
      plt.scatter(points[:,0],points[:,1],c="k")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.title("(a) Scatter plot of Dataset")
else:
      plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c="k")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.title("(a) Scatter plot of Dataset")
plt.show()

file2 = open("knn_rad.txt", "w+")
file3 = open("lec_rad.txt", "w+")
file5 = open("descrip.txt","w+")
file6 = open("glance.txt","w+")
file7 = open("shscs.txt","w+")
shar = []
shark = []
for run in [1,2]:
      print("------------------------------------------------")
      print("------------------------------------------------")
      #if run == 2 and ep == 1:
      if run == 2:
            bestmin = shi
            print("The bestmin from ECs = ",bestmin)
      #if run == 2 and ep == 2:
            #bestmin = shik
            #print("The bestmin is for KNN = ",bestmin)
      for min_s in runar:
            file4 = open('knndist_%s.txt' % str(min_s), 'w+')
                          #for your run, give your own path 
            if (run ==1 and min_s == 2):
                  vor = Voronoi(points)
                  hull = ConvexHull(points)
                  myList1=np.array(vor.vertices)            
                  myList= np.array(points[hull.vertices])                  
                  ncols=points.shape[1]
                  insideVor=[]
                  b= myList
                  for ij in range(0,len(myList1)):
                        x=(np.append(b,myList1[ij]))
                        x=np.array(np.reshape(np.array([x]), (-1, ncols)))
                        hull2=ConvexHull(x)
                        oldHull = np.array(points[hull.vertices])
                        newHull = np.array(x[hull2.vertices])
                        if np.array_equal(oldHull,newHull):
                             insideVor.append(myList1[ij])
                        else:
                             continue  
                  in_vor1=np.array(insideVor)
                  out_vor=np.array([elem for elem in myList1 if elem not in in_vor1])
                  rad_dist=(distance.cdist(in_vor1,points,metric='euclidean'))
                  radlist=[]
                  for ii in range(len(rad_dist)):
                      radlist.append(min(rad_dist[ii]))
                  radius1 = np.array(np.sort(radlist))
                  fact = len(radius1)/len(points)
                  xdc = range(len(radius1))
                  knc = KneeLocator(xdc, radius1, curve='convex', direction='increasing')
                  epsilon1 = radius1[knc.knee]
                  print("ECs :   ",knc.knee," ", epsilon1)
                  for hk in range(len(radius1)):
                        print(hk," ",radius1[hk], file = file3)
                  file3.close()

            #Knee Method for Best EPS
            nearest_neighbors = NearestNeighbors(n_neighbors=min_s)
            nearest_neighbors.fit(points)
            distances, indices = nearest_neighbors.kneighbors(points)
            dist = np.sort(distances[:,min_s-1])
            if run == 2 and min_s == bestmin:
                  for hk1 in range(len(dist)):
                        print(hk1," ",dist[hk1], file = file2)
                  file2.close()
            xd = range(len(dist))
            kn = KneeLocator(xd, dist, curve='convex', direction='increasing')
            epsilon2 = dist[kn.knee]
            if run == 2 and min_s == bestmin:
                  print("KNN :   ",kn.knee," ", epsilon2)
            if run == 2:
                  for hh in range(len(dist)):
                        print(hh,"  ",kn.knee, "  ",dist[hh], file = file4)
                  file4.close()
            if run == 2 and min_s == bestmin:
                  plt.plot(dist,"g")
                  plt.plot(radius1,"b")
                  plt.title("(a) Distance(KNN)/Radius(EC) Distributions")
                  plt.xlabel("Count Index")
                  plt.ylabel("Distance(KNN) / Radius(EC)")
                  plt.scatter(kn.knee,dist[kn.knee],marker = "o", c= "m",s = 90)
                  plt.scatter(knc.knee,radius1[knc.knee],marker = "*",c= "r", s = 130)
                  plt.legend(["KNN Distance","EC Radius","eps from KNN", "eps from EC"])
                  plt.savefig("distri.png")
                  plt.show()
            if ep == 1:
                  epsilon = epsilon1
            else:
                  epsilon = epsilon2
            
            model = DBSCAN(eps=epsilon1, min_samples=min_s)
            modelk = DBSCAN(eps=epsilon2, min_samples=min_s)
            model.fit(points)
            modelk.fit(points)
            label =(model.labels_)
            labelk =(modelk.labels_)
            label1 = label.copy()
            labelk1 = labelk.copy()
            if run == 2 and min_s == bestmin:
                  if ep ==1:
                        if dim ==0:
                              plt.scatter(points[:,0],points[:,1],c=label,cmap='rainbow')
                        else:
                              plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c=label,cmap='rainbow')
                  else:
                        if dim ==0 :
                              plt.scatter(points[:,0],points[:,1],c=labelk,cmap='rainbow')
                        else:
                              plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c=labelk,cmap='rainbow')
                  plt.savefig("old.png")
                  plt.show()
            noc = len(set(label))-(1 if -1 in label else 0)
            nock = len(set(labelk))-(1 if -1 in labelk else 0)
            core=model.core_sample_indices_
            clus=np.array(points[core])
            #*****************************************
            clus_points=[]
            clus_cen=[]
            if run==2 and ep ==1:
                  for c in range(0,noc):
                      cp=points[(np.where(label==c))]
                      centroid=np.mean(cp,axis=0)
                      clus_points.append(cp)
                      clus_cen.append(centroid)
                  noise_points=points[np.where(label==-1)]
                  noise_dist=(distance.cdist(noise_points,clus_cen,metric='euclidean'))
                  nn=[]
                  for m in range(len(noise_dist)):
                        noise_idx=(np.where(noise_dist[m]==(min(noise_dist[m]))))[0]
                        nn.extend(noise_idx)
                  mod_label=label1
                  ex=np.array(np.where(mod_label==-1)[0])
                  for xx in range(len(ex)):
                      mod_label[ex[xx]]=nn[xx]
            if run==2 and ep ==2:
                  for c in range(0,nock):
                      cp=points[(np.where(labelk==c))]
                      centroid=np.mean(cp,axis=0)
                      clus_points.append(cp)
                      clus_cen.append(centroid)
                  noise_points=points[np.where(labelk==-1)]
                  noise_dist=(distance.cdist(noise_points,clus_cen,metric='euclidean'))
                  nn=[]
                  for m in range(len(noise_dist)):
                        noise_idx=(np.where(noise_dist[m]==(min(noise_dist[m]))))[0]
                        nn.extend(noise_idx)
                  mod_labelk=labelk1
                  ex=np.array(np.where(mod_labelk==-1)[0])
                  for xx in range(len(ex)):
                       mod_labelk[ex[xx]]=nn[xx]
            if run ==1:
                  if noc >= 2:
                        sh = silhouette_score(points, label)
                  else:
                        sh = 0
                  shar.append(sh)
                  if nock >= 2:
                        shk = silhouette_score(points, labelk)
                  else:
                        shk = 0
                  shark.append(shk)
            if run == 2 and min_s == bestmin:
                  if ep == 1:
                        eucdist = np.round(euclidean(target, mod_label))
                        mandist = np.round(cityblock(target, mod_label))
                        if noc ==1:
                              dbs = 1
                              shc = 0
                        else:
                              dbs = (davies_bouldin_score(points1, mod_label))
                              shc = silhouette_score(points1, mod_label)
                        acc = np.round(accuracy_score(target, mod_label))
                        hom = np.round(homogeneity_score(target, mod_label))
                        com = np.round(completeness_score(target, mod_label))
                  else:
                        eucdist = np.round(euclidean(target, mod_labelk))
                        mandist = np.round(cityblock(target, mod_labelk))
                        if nock ==1:
                              dbs = 1
                              shc = 0
                        else:
                              dbs = (davies_bouldin_score(points1, mod_labelk))
                              shc = silhouette_score(points1, mod_labelk)
                        acc = (accuracy_score(target, mod_labelk))
                        hom = (homogeneity_score(target, mod_labelk))
                        com = (completeness_score(target, mod_labelk))
                  print("Min_Sample = ", min_s, file = file5)
                  if ep == 1:
                        print("Final Counter=",Counter(mod_label),file = file5)
                  else:
                        print("Final Counter=",Counter(mod_labelk),file = file5)
                  if tp ==1:
                        print("Target Counter=",Counter(target),file = file5)
                  print("Eps = ",epsilon,file = file5)
                  print("Eps_EC = ",epsilon1,file = file5)
                  print("Eps_KNN = ",epsilon2,file = file5)
                  if ep ==1:
                        print("Number of Cluster = ",noc,file = file5)
                  else:
                        print("Number of Cluster = ",nock,file = file5)
##                  if ep ==1:
##                        print("Sil Scores with Noise =  ", shar[min_s-2], file = file5)
##                  else:
##                        print("Sil Scores =  ", shark[min_s-2], file = file5)
                  print("Shiloutte Scores  =  ",   shc, file = file5)
                  print("DB Scores  =  ",  dbs,file = file5)
                  print("Accuracy  =  ",  acc,file = file5)
                  print("Completeness  =  ",  com,file = file5)
                  print("Homogenity  =  ",  hom,file = file5)
                  print("Eclidean Scores  =  ",  eucdist,file = file5)
                  print("Manhatten Scores  =  ", mandist,file = file5)
                  print(" ",file = file5)

            if run == 2 and min_s == bestmin:
                  if ep ==1:
                        if dim ==0:
                              plt.scatter(points[:,0],points[:,1],c=mod_label,cmap='rainbow')
                              plt.xlabel("x")
                              plt.ylabel("y")
                              plt.title("(d)  Clustering using ECR-DBSCAN")
                        else:
                              plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c=mod_label,cmap='rainbow')
                              plt.xlabel("x")
                              plt.ylabel("y")
                              plt.clabel("z")
                              plt.title("(d)  Clustering using ECR-DBSCAN")
                  else:
                        if dim ==0 :
                              plt.scatter(points[:,0],points[:,1],c=mod_labelk,cmap='rainbow')
                              plt.xlabel("x")
                              plt.ylabel("y")
                              plt.title("(c) Clustering using DBSCAN")
                        else:
                              plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c=mod_labelk,cmap='rainbow')
                              plt.xlabel("x")
                              plt.ylabel("y")
                              plt.clabel("z")
                              plt.title("(c) Clustering using DBSCAN")
                  plt.savefig("new.png")
                  plt.show()
            if run == 2:
                  if (min_s ==2):
                        print("Mpt   ","eps_ec   ","eps_kn   ","eps   ","noc   ","SCs   ", file = file6)
                  if ep ==1:
                        print(min_s," ", epsilon1, " ",epsilon2," ", epsilon, " ", noc," ", sh, file = file6)
                  else:
                       print(min_s," ", epsilon1, " ",epsilon2," ", epsilon, " ", nock," ", shk, file = file6)                                                 
      shi = shar.index(max(shar))+2
      shik = shark.index(max(shark))+2
      print("Best Clustering for min_s in ECs = ", shi," ", max(shar))
      print("Best Clustering for min_s in KNN = ", shik," ", max(shark))
      print(" ")
      if ep == 1:
            print(" epsilon selected for ECs")
      else:
            print("eps selected for KNN")
      
      if tp ==1 and run == 2:
            if dim ==0:
                  plt.scatter(points1[:,0],points1[:,1],c=target,cmap='rainbow')
                  plt.title("(b) Target Clustering of Dataset")
                  plt.xlabel("x")
                  plt.ylabel("y")
            else:
                  plt.axes(projection="3d").scatter(points[:,0],points[:,1],points[:,2],c=target,cmap='rainbow')
                  plt.title("(b) Target Clustering of Dataset")
                  plt.xlabel("x")
                  plt.ylabel("y")
                  plt.clabel("z")
            plt.savefig("target.png")
            plt.show()
      if run == 1:
            bb = []
            for ii in range(len(shar)):
                  print(ii+2," ",shar[ii]," ",shark[ii],file = file7)
                  bb.append(ii+2)
            plt.plot(bb,shar,"--bo")
            plt.plot(bb,shark,"--kP")
            plt.legend(["LECs","KNN"])
            plt.savefig("shsc.png")
            plt.show()
file5.close()
file6.close()
file6.close()
file7.close()
