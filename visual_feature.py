import cv2
import os
import json
import numpy as np
def read_json(fname):
    with open(fname, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


path1 = '15_37.6792_126.7539_300.0'
path2 = '14_37.6792_126.7539_290.0'

img_path1 = os.path.join('data/Daesan2', path1 + '.jpg')
img_path2 = os.path.join('data/Daesan2', path2 + '.jpg')

json_path1 = os.path.join('data/Daesan2', path1 + '.json')
json_path2 = os.path.join('data/Daesan2', path2 + '.json')

pt1 = [[[331.797409675636, 564.0537441020632], [362.81521326327515, 530.6818490335395], [366.19494070739825, 227.6404583998067], [366.19494070739825, 93.88406208047712], [330.62622094517843, 37.257954060575514], [330.62622094517843, 318.98873566832174]], [[506.920475515217, 326.0443505369773], [679.6612033031743, 321.22072754064754], [898.7549242961991, 316.7551644170052], [1120.1958797652567, 312.60710722840963], [1123.3456951991443, 227.68398151668828], [897.7053311030793, 231.33030565562115], [745.5583653759985, 233.79798912333513], [616.712682323616, 236.2326645973476], [506.920475515217, 239.55019090227933]]]
pt2 = [[[483.7082866384076, 294.13364758145315], [484.92023411985326, 202.92153729072365], [622.4829905777467, 210.06989663637808], [730.9565369265563, 217.0684716619601], [926.849973169486, 229.91748216211724], [1014.6941437291772, 235.7570651480749], [1012.0403352366982, 309.69335361352756], [777.1438710370608, 302.38438555006417], [555.5756506680856, 295.3637830557653]]]

img1 = cv2.imread(img_path1)
for p1 in pt1:
    temp = np.array(p1).astype(np.int64)
    xmin = np.min(temp[:,0])
    xmax = np.max(temp[:,0])
    ymin = np.min(temp[:,1])
    ymax = np.max(temp[:,1])
    img1 = cv2.rectangle(img1, (ymin, xmin), (ymax, xmax), (255,0,0),3 )
img2 = cv2.imread(img_path2)
for p1 in pt2:
    temp = np.array(p1).astype(np.int64)
    xmin = np.min(temp[:,0])
    xmax = np.max(temp[:,0])
    ymin = np.min(temp[:,1])
    ymax = np.max(temp[:,1])
    img2 = cv2.rectangle(img2, (ymin, xmin), (ymax, xmax), (255,0,0),3 )



con_img = cv2.hconcat([img1, img2])

cv2.imwrite('test1.jpg', con_img)
breakpoint()
