"""
Graph Builder Version 2
-----------------------
Enhancements over Version 1:
- Stores node attributes (x, y, degree, node_type)
- Stores edge attributes (length, path coordinates)
- Exports GraphML
"""

import os
import cv2
import json
import numpy as np
import networkx as nx
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

INPUT_DIR = "Dataset/OCTA-500/Output_clDice"
OUTPUT_DIR = "Graph/Graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

KERNEL = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def load_mask(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img>0).astype(np.uint8)

def get_skeleton(mask):
    return skeletonize(mask).astype(np.uint8)

def detect_nodes(skel):
    conv=convolve(skel.astype(np.uint8),KERNEL)
    endpoints=np.argwhere((skel==1)&(conv==11))
    branchpoints=np.argwhere((skel==1)&(conv>=13))
    return endpoints,branchpoints

def neighbours(skel,pixel,prev=None):
    y,x=pixel
    out=[]
    for dy,dx in OFFSETS:
        yy,xx=y+dy,x+dx
        if 0<=yy<skel.shape[0] and 0<=xx<skel.shape[1]:
            if skel[yy,xx] and (yy,xx)!=prev:
                out.append((yy,xx))
    return out

def build_graph(skel):
    ep,bp=detect_nodes(skel)
    G=nx.Graph()
    node_map={}
    nid=0

    for pts,tp in [(ep,"endpoint"),(bp,"branch")]:
        for y,x in pts:
            node_map[(int(y),int(x))]=nid
            G.add_node(
                nid,
                x=int(x),
                y=int(y),
                node_type=tp
            )
            nid+=1

    visited_edges=set()

    for start_pixel,start_id in node_map.items():
        for nxt in neighbours(skel,start_pixel):
            prev=start_pixel
            cur=nxt
            path=[start_pixel]

            while True:
                path.append(cur)

                if cur in node_map and cur!=start_pixel:
                    end_id=node_map[cur]
                    key=tuple(sorted((start_id,end_id)))
                    if key not in visited_edges:
                        visited_edges.add(key)
                        G.add_edge(
                            start_id,
                            end_id,
                            length=len(path),
                            path=json.dumps([(int(p[1]),int(p[0])) for p in path])
                        )
                    break

                nbs=neighbours(skel,cur,prev)

                if len(nbs)==0:
                    break

                prev=cur
                cur=nbs[0]

    for n in G.nodes:
        G.nodes[n]["degree"]=G.degree[n]

    return G,ep,bp

def visualize(skel,ep,bp):
    vis=np.dstack([skel*255]*3)
    for y,x in ep:
        cv2.circle(vis,(int(x),int(y)),3,(0,255,0),-1)
    for y,x in bp:
        cv2.circle(vis,(int(x),int(y)),3,(0,0,255),-1)
    return vis

def process(image_path):
    name=os.path.basename(image_path)
    mask=load_mask(image_path)
    skel=get_skeleton(mask)

    G,ep,bp=build_graph(skel)

    out=os.path.join(OUTPUT_DIR,os.path.splitext(name)[0]+".graphml")
    nx.write_graphml(G,out)

    print("="*60)
    print("Image :",name)
    print("Nodes :",G.number_of_nodes())
    print("Edges :",G.number_of_edges())
    print("Endpoints :",len(ep))
    print("Branch Points :",len(bp))
    print("Connected Components :",nx.number_connected_components(G))
    print("="*60)

    img=visualize(skel,ep,bp)
    cv2.imshow("Skeleton Graph V2",img)
    if cv2.waitKey(0)==27:
        return False
    return True

def main():
    files=sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp"))
    print("Images Found:",len(files))
    for f in files:
        if not process(os.path.join(INPUT_DIR,f)):
            break
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
