from django import forms
from .models import UploadedVideo, Camera

_DARK = {'class': 'form-control bg-dark text-light border-secondary'}
_SEL  = {'class': 'form-select bg-dark text-light border-secondary'}


class VideoUploadForm(forms.ModelForm):
    class Meta:
        model   = UploadedVideo
        fields  = ['video_file', 'location', 'camera']
        widgets = {
            'video_file': forms.FileInput(attrs=_DARK),
            'location':   forms.Select(attrs=_SEL),
            'camera':     forms.Select(attrs=_SEL),
        }


class CameraForm(forms.ModelForm):
    class Meta:
        model   = Camera
        fields  = ['name', 'location', 'stream_url', 'is_active']
        widgets = {
            'name':       forms.TextInput(attrs=_DARK),
            'location':   forms.Select(attrs=_SEL),
            'stream_url': forms.TextInput(attrs={**_DARK, 'placeholder':
                          'rtsp://ip:port/stream  or  /path/to/video.mp4'}),
            'is_active':  forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        help_texts = {
            'stream_url': 'RTSP URL, HTTP stream URL, or absolute path to a local video file.',
        }
