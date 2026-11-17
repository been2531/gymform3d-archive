"""One-off: 6 palette variants of the style-A puppet, same pose + ghost."""
import sys, skia
sys.path.insert(0, "/workspace/daily_info_reels")
from pilots.rig import Body, pushup_pose, hand_x_for
from pilots.puppet import Puppet, PALETTES

OUT = sys.argv[1]
body = Body(H=820)
names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["grey", "porcelain", "coral", "mint", "cobalt", "lilac"]
CW, CH = 1080, 760
COLS = 3 if len(names) > 6 else 2; ROWS = (len(names) + COLS - 1) // COLS
sheet = skia.Surface(CW * COLS, CH * ROWS)
sc = sheet.getCanvas()
font = skia.Font(skia.Typeface.MakeFromFile("/usr/share/fonts/truetype/sand-box/google/Anton/Anton-Regular.ttf"), 120)
for i, n in enumerate(names):
    pal = PALETTES[n]
    cell = skia.Surface(CW, CH); cv = cell.getCanvas(); cv.clear(pal["bg"])
    floor = 640
    cv.drawLine(30, floor, CW - 30, floor, skia.Paint(Color=0x33FFFFFF if pal["bg"] == 0xFF2E3033 else 0x33000000, StrokeWidth=4, AntiAlias=True))
    pp = Puppet(body, n); HX = hand_x_for(body, floor, 130)
    ghost = 0xFFFF4D4D if n not in ("coral",) else 0xFFFFE14D
    pp.draw(cv, pushup_pose(body, 0.55, {"sag": 24, "head": 30}, floor, 130, HX), "ghost", ghost=ghost)
    pp.draw(cv, pushup_pose(body, 0.55, {}, floor, 130, HX))
    txtc = 0xFFFFFFFF if pal["bg"] in (0xFF2E3033,) else 0xFF111111
    cv.drawString(str(i + 1), 40, 150, font, skia.Paint(Color=txtc, AntiAlias=True))
    sc.drawImage(cell.makeImageSnapshot(), (i % COLS) * CW, (i // COLS) * CH)
    sc.drawRect(skia.Rect.MakeXYWH((i % COLS) * CW, (i // COLS) * CH, CW, CH), skia.Paint(Color=0xFF000000, Style=skia.Paint.kStroke_Style, StrokeWidth=6))
img = sheet.makeImageSnapshot()
img.save(OUT, skia.kJPEG, 90)
