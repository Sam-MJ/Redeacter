# Audio and video Redacter
## story/requirements
- The tool will find all people in the audiovisual input
    - voice recognition/diarization
    - face recognition
- The user will select the person they intend to keep, then all audio apart from the selected persons will be muted.
- The user will select the face of the person they intend on keeping, the rest will be blacked out or blurred

## Audio
### Nvidia NeMo Diarizer
- extract speakers
    - Neural Diarizer is better for overlapping audio
### Editing
- soundfile
- numpy
- short crossfade in/out
## Video
- OpenCV face recognition
