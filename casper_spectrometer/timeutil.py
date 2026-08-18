"""Timezone constants shared across the package (UTC and IST), plus the
lookup table used by the GUI's date/time picker.
"""

from datetime import timezone, timedelta

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
TZ_MAP = {'UTC': UTC, 'IST': IST}
