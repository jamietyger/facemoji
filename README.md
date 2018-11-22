# Facemoji

Program that identifies faces in an image, recognizes their emotion and overlays the appropriate emoji. (Python 3)

## To build classifier:

Populate ```/project/assets/dataset/<emotion>``` with images.

Current emotions supported are `happy`,`angry`, `neutral` and `suprised`

`python build_data.py`
 
`python build_svm.py`

## To run program

`python facemoji.py <image_path>`

## Output

### Before:

![Screenshot](before.jpeg)

### After:

![Screenshot](after.jpg)

## Requirements

OpenCV

Sklearn

csv
