"""
11_tanima_demo.py — OBJECT RECOGNITION: the robot's HEAD CAMERA SEES the colored objects
in front of it (cube / sphere / cylinder) and RECOGNIZES them by COLOR + SHAPE (OpenCV).

A head camera + 3 colored objects are injected into the scene (base model unchanged).
Pipeline: MuJoCo camera render -> HSV color thresholding -> contour -> shape classification -> label.
Output: videos/11_tanima.png (boxed+labeled) + a "seen" list in the terminal.
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np
import mujoco
import cv2
import imageio.v2 as imageio

ROOT = "/home/yasin/Workspace/humanoid_rl"
with open(f"{ROOT}/models/mini_humanoid.xml") as f:
    xml = f.read()
# 1) forward-facing head camera on torso
xml = xml.replace('<geom name="torso_g"',
    '<camera name="kafa" pos="0.06 0 0.14" xyaxes="0 -1 0 0 0 1" fovy="75"/>\n      <geom name="torso_g"', 1)
# 2) 3 colored objects in front (cube / sphere / cylinder)
xml = xml.replace('</worldbody>',
    '  <geom name="o_kirmizi" type="box"      pos="1.05 -0.30 0.32" size="0.09 0.09 0.09" rgba="0.9 0.07 0.07 1"/>\n'
    '  <geom name="o_yesil"   type="sphere"   pos="1.05  0.00 0.32" size="0.11"           rgba="0.07 0.8 0.12 1"/>\n'
    '  <geom name="o_mavi"    type="cylinder" pos="1.05  0.30 0.34" size="0.08 0.14"       rgba="0.1 0.25 0.92 1"/>\n'
    '  </worldbody>', 1)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 640, 480)
r.update_scene(data, camera="kafa")
img = r.render()                                   # RGB (480,640,3)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
RENKLER = {
    "kirmizi": [((0, 110, 60), (10, 255, 255)), ((170, 110, 60), (180, 255, 255))],
    "yesil":   [((38, 70, 40), (88, 255, 255))],
    "mavi":    [((95, 110, 40), (132, 255, 255))],
}
annot = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # BGR for drawing
bulunan = []
for renk, araliklar in RENKLER.items():
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in araliklar:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 350:
            continue
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri + 1e-6)
        asp = h / (w + 1e-6)
        if circ > 0.80:
            sekil = "kure"
        elif asp >= 1.35:
            sekil = "silindir"
        else:
            sekil = "kup"
        etiket = f"{renk} {sekil}"
        bulunan.append(etiket)
        cv2.rectangle(annot, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(annot, etiket, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

imageio.imwrite(f"{ROOT}/videos/11_tanima.png", cv2.cvtColor(annot, cv2.COLOR_BGR2RGB))
print("ROBOT HEAD CAMERA SAW:", bulunan if bulunan else "(nothing — threshold/position tuning needed)")
print("-> videos/11_tanima.png")
