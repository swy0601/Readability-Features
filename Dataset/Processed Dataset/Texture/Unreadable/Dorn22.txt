from taskcoachlib.syncml.basesource import BaseSource
from taskcoachlib.persistence.icalendar import ical

from taskcoachlib.i18n import _

import wx, inspect

class NoteSource(BaseSource):
    def __init__(self, callback, noteList, categoryList, *args, **kwargs):
        super(NoteSource, self).__init__(callback, noteList, *args, **kwargs)
