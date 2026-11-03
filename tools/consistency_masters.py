import sys,json,cv2
sys.path.insert(0,'tools'); import consistency_metrics as cm
M={"D4":"output/day4_romanian_deadlift_4step_master.mp4","D5":"output/day5_dumbbell_lateral_raise_4step_master_v5.mp4",
   "D6":"output/day6_incline_dumbbell_press_muscle2_master.mp4"}
import glob,os
out={}
for d,p in M.items():
    if not os.path.exists(p):
        g=sorted(glob.glob(f"output/day{d[1]}*master*.mp4")); 
        if not g: continue
        p=g[-1]
    cap=cv2.VideoCapture(p); fps=cap.get(5); n=cap.get(7)
    dur=n/fps; ts=[dur*(k+0.5)/4 for k in range(4)] if d in("D4","D5") else [dur*0.25,dur*0.75]
    r=[]
    for t in ts:
        cap.set(1,int(t*fps)); ok,f=cap.read(); f=cv2.resize(f,(cm.W,cm.H),interpolation=cv2.INTER_AREA)
        m,_=cm.frame_metrics(f); r.append({k:m.get(k) for k in ("skin_L_p50","skin_L_p90","view_ratio","body_h_frac","bg_rgb")})
    out[d]={"path":p,"steps":r}; print(d,p,r)
json.dump(out,open('output/_consistency/masters_graded.json','w'),indent=1)
