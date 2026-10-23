# Room backgrounds

Put PNG/JPG photos of the room around the course here (people, furniture,
windows, lights - anything the real camera may see above the walls).
Each simulated run wraps one random image around the arena as a dome
(`hamster_super/sim/backgrounds.py`), so the camera pipeline does not rely on
a plain background. The Pages 3D view uses the same images. When this folder
has no images, procedural rooms are generated instead (they are also mixed
in 30% of the time when images exist). `$HAMSTER_BG_DIR` points elsewhere.
