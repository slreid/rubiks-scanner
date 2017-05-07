# rubiks-scanner

A prototype computer vision application to replace human data entry at Rubik's Cube competitions.

Created by **Bridget Andersen**, **Samantha Reid**, and **Steven Stetzler**.

For an in-depth description of our project, read our write-up contained in *rubiks_scanner_info.pdf*

Software Requirements:
- Python 3.5.3
- OpenCV 3.2.0
- Tensorflow 1.0.0
- Keras 2.0.2

Run one of either of the following in order to test our project.
- `rubiks_scanner_static_image.py`
- `rubiks_scanner_webcam_video.py`
- `rubiks_scanner_phone_video.py`

The *static_image* version uses a saved image of a scorecard and matches it to a template. The *webcam_video* version uses the output of the current computer's default webcam as a video capture device. The *phone_video* version connects to an IP camera at a specified IP address, hardcoded into that file.

Note on Windows/OS X/Linux compatibility: Some file loctions are hardcoded using Windows path syntax in our files. Make changes at the following locations in order to allow the code to run on OS X/Linux
- `rubiks_database.py`: line 6, 7,
- `rubiks_scanner_core.py`: line 72, 
- `rubiks_scanner_phone_video.py`: line 15
- `rubiks_scanner_static_image.py`: line 6, 7
- `rubiks_scanner_webcam_video.py`: line 6

You can find a database storing information from all processed scorecards at
https://rubiksscanner.firebaseio.com/

If you need access to view the database, send a request to slr4ee@virginia.edu

References:

Many sources and tutorials were used in the creation of this project. You can find them at the following locations
1. [SIFT Feature Matching](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#brute-force-matching-with-sift-descriptors-and-ratio-test)
2. [Rectangle Contour Extraction](http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)
3. [OpenCV Video Capture](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html)