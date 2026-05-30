from django import forms
from .models import UploadedVideo

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video_file', 'location']
        widgets = {
            'video_file': forms.FileInput(attrs={'class': 'form-control bg-dark text-light border-secondary'}),
            'location': forms.Select(attrs={'class': 'form-select bg-dark text-light border-secondary'}),
        }
