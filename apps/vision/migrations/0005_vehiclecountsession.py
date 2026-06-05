import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('traffic', '0001_initial'),
        ('vision',  '0004_vision_cv_fields'),
    ]

    operations = [
        migrations.CreateModel(
            name='VehicleCountSession',
            fields=[
                ('id',            models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('session_tag',   models.CharField(max_length=64, unique=True)),
                ('started_at',    models.DateTimeField(default=django.utils.timezone.now)),
                ('ended_at',      models.DateTimeField(blank=True, null=True)),
                ('total_count',   models.IntegerField(default=0)),
                ('car_count',     models.IntegerField(default=0)),
                ('truck_count',   models.IntegerField(default=0)),
                ('bus_count',     models.IntegerField(default=0)),
                ('motorcycle_count', models.IntegerField(default=0)),
                ('inbound_count',  models.IntegerField(default=0)),
                ('outbound_count', models.IntegerField(default=0)),
                ('avg_speed',      models.FloatField(default=0.0)),
                ('peak_congestion',models.CharField(default='FREE FLOW', max_length=20)),
                ('csv_file',       models.FileField(blank=True, null=True, upload_to='counts/')),
                ('location', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    to='traffic.location',
                )),
                ('source_camera', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='count_sessions',
                    to='vision.camera',
                )),
                ('source_video', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='count_sessions',
                    to='vision.uploadedvideo',
                )),
            ],
            options={'ordering': ['-started_at']},
        ),
    ]
