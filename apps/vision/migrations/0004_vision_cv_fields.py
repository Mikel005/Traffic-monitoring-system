import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('traffic', '0001_initial'),
        ('vision', '0003_uploadedvideo_location'),
    ]

    operations = [
        # ── Per-class counts on UploadedVideo ──────────────────────
        migrations.AddField(
            model_name='uploadedvideo',
            name='car_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='uploadedvideo',
            name='truck_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='uploadedvideo',
            name='motorcycle_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='uploadedvideo',
            name='bus_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='uploadedvideo',
            name='queue_length',
            field=models.FloatField(default=0.0),
        ),
        # ── FK to Camera on UploadedVideo ──────────────────────────
        migrations.AddField(
            model_name='uploadedvideo',
            name='camera',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='videos',
                to='vision.camera',
            ),
        ),
    ]
