import sys,json,cv2,numpy as np
sys.path.insert(0,'tools'); import consistency_metrics as cm
pairs=[("D6 v5 wrong","inputs/day6_v5/day6_v5_wrong_start.png","output/assets/day6_v5_wrong_raw.mp4"),
("D6 v5 right","inputs/day6_v5/day6_v5_right_start.png","output/assets/day6_v5_right_raw.mp4"),
("D7 v3 wrong","inputs/day7/day7_v3_wrong_start_720.png","output/assets/day7_wrong_raw.mp4"),
("D7 v3 right","inputs/day7/day7_v3_right_start_720.png","output/assets/day7_right_raw.mp4"),
("D7 v2 wrong","inputs/day7/day7_v2_wrong_start_720.png","output/assets/day7_wrong_raw_v2.mp4"),
("D7 v2 right","inputs/day7/day7_v2_right_start_720.png","output/assets/day7_right_raw_v2.mp4"),
("D7 v1 wrong","inputs/day7/day7_wrong_start_720.png","output/assets/day7_wrong_raw_v1.mp4"),
("D7 v1 right","inputs/day7/day7_right_start_720.png","output/assets/day7_right_raw_v1.mp4")]
from skimage.metrics import structural_similarity as ssim
out={}
for n,sf,clip in pairs:
    s=cv2.resize(cv2.imread(sf),(cm.W,cm.H),interpolation=cv2.INTER_AREA)
    frs,idx,_,_=cm.read_samples(clip)
    ms,_=cm.frame_metrics(s); m0,_=cm.frame_metrics(frs[0]); ml,_=cm.frame_metrics(frs[-1])
    g=lambda m:(m.get('glow_red_frac',0) or 0)+(m.get('glow_green_frac',0) or 0)
    gl=[g(cm.frame_metrics(f)[0]) for f in frs]
    out[n]=dict(ssim_start_f0=round(float(ssim(cv2.cvtColor(s,6),cv2.cvtColor(frs[0],6),data_range=255)),3),
      ssim_start_last=round(float(ssim(cv2.cvtColor(s,6),cv2.cvtColor(frs[-1],6),data_range=255)),3),
      skin_start=ms.get('skin_L_p50'),skin_f0=m0.get('skin_L_p50'),skin_last=ml.get('skin_L_p50'),
      bg_start=ms['bg_rgb'],bg_f0=m0['bg_rgb'],
      glow_start=round(g(ms),3),glow_series=[round(x,3) for x in gl])
    print(n,out[n])
json.dump(out,open('output/_consistency/startframe_vs_clip.json','w'),indent=1)
