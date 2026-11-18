# PiDog mole scan

Uses OpenCV on the PiDog's camera to find mole-like dark spots on skin, show them live in a browser, and save dated snapshots so you can compare them over time.

> This is a tracking aid, not a diagnosis. The "notes" (asymmetric, irregular border, uneven colour, >6mm) are geometric measurements from a low-res camera. If a spot is new, changing, itching or bleeding, have a dermatologist look at it.

## Install (on the PiDog)

Target: pi5w (Pi 5, Debian trixie, user `ddietrich`, 192.168.1.191). See `claude/pi5w-setup.md`.

```bash
cd ~/pidog && git fetch origin && git checkout mole-scan   # or merge it into main
sudo apt install -y python3-opencv python3-picamera2   # vilib's installer usually pulls both in
```

`--pidog` uses the editable `~/pidog` install. Like the SunFounder examples, run it with `sudo` (`sudo python3 mole_scan.py --pidog`).

## Run

```bash
cd ~/pidog/examples/mole_scan
python3 mole_scan.py                  # camera only
sudo python3 mole_scan.py --pidog     # sit, sweep head, LEDs go pink when it finds a spot
sudo python3 mole_scan.py --pidog --no-sweep --bark
```

Open `http://192.168.1.191:8000`. The page shows the annotated stream, a live JSON list of the spots, and a **Save snapshot** button. It also saves automatically once a spot has been visible for 6 frames (`--stable-frames`), at most once every 15s (`--cooldown`). Each snapshot is saved to `scans/YYYY-MM-DD/` as `HHMMSS_raw.jpg`, `_annotated.jpg` and `.json`.

Close the SunFounder app and any vilib script first, because only one process can use the camera at a time.

## Getting good results

- **Distance and focus.** The stock OV5647 module has a fixed focus and is soft closer than about 25–30cm. Hold skin around 30cm from the nose, or fit a Camera Module 3 (autofocus is turned on automatically).
- **Light.** Use even, diffuse light. Shadows and skin creases are what cause false positives.
- **Real sizes.** Put a ruler in frame, measure how many pixels make up 10mm in a raw snapshot, and pass `--px-per-mm <px/10>`. This only holds at that same distance.
- **Sensitivity.** `--contrast 10` finds fainter spots (and more noise). `--contrast 25` keeps only dark spots.
- **Skin detection.** Use `--show-skin` to see what counts as skin. The YCrCb range is in `skin_mask()`. Very light or very dark skin may need it widened.
- **Testing on a laptop.** `python3 mole_scan.py --source 0` uses a webcam, and `--source photo.jpg` reads a photo. For a single image, `python3 mole_detector.py photo.jpg out.jpg` prints JSON and writes an annotated image.

## How it works

1. **Skin mask.** A YCrCb colour range picks out skin. Holes are filled so dark moles inside it aren't cut out, and the edge is eroded so the arm's outline isn't counted as a spot.
2. **Dark spots.** A black-hat transform on LAB lightness finds small areas darker than the skin around them.
3. **Filtering.** Spots are kept by area, circularity (≥0.30, which drops hairs) and solidity (≥0.75, which drops creases).
4. **Measurements.** Each spot gets asymmetry (mirror-flip mismatch about its principal axes), border (P²/4πA, where 1 is a circle), colour spread (LAB std inside the spot) and diameter.
