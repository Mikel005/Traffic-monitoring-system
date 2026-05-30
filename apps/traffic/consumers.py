"""
Django Channels WebSocket consumer.
Broadcasts live traffic updates every 10 seconds to all connected browsers.
"""
import json
import asyncio
from datetime import datetime

from channels.generic.websocket import AsyncWebsocketConsumer


class TrafficConsumer(AsyncWebsocketConsumer):
    GROUP_NAME = 'traffic_live'

    async def connect(self):
        await self.channel_layer.group_add(self.GROUP_NAME, self.channel_name)
        await self.accept()
        # Start sending live updates immediately
        self.push_task = asyncio.create_task(self._push_loop())

    async def disconnect(self, code):
        if hasattr(self, 'push_task'):
            self.push_task.cancel()
        await self.channel_layer.group_discard(self.GROUP_NAME, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        """Handle ping from client (keep-alive)."""
        pass

    async def traffic_update(self, event):
        """Called when group_send is used from a Celery task."""
        await self.send(text_data=json.dumps(event['data']))

    async def _push_loop(self):
        """Send live data snapshot every 10 seconds."""
        from channels.db import database_sync_to_async
        import random, math

        while True:
            try:
                data = await self._get_live_data()
                await self.send(text_data=json.dumps({
                    'type':      'traffic_update',
                    'timestamp': datetime.now().isoformat(),
                    'readings':  data,
                }))
            except Exception:
                break
            await asyncio.sleep(10)

    @staticmethod
    async def _get_live_data():
        from channels.db import database_sync_to_async

        @database_sync_to_async
        def fetch():
            from apps.traffic.models import Location
            result = []
            for loc in Location.objects.filter(is_active=True):
                r = loc.readings.first()
                if r:
                    result.append({
                        'location_id':      loc.id,
                        'location_name':    loc.name,
                        'congestion_index': r.congestion_index,
                        'congestion_level': r.congestion_level,
                        'avg_speed':        r.avg_speed,
                        'vehicle_count':    r.vehicle_count,
                        'latitude':         loc.latitude,
                        'longitude':        loc.longitude,
                    })
            return result

        return await fetch()
