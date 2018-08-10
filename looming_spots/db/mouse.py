import datetime


class Mouse(object):

    def __init__(self, mouse_id=None, strain='C57BL/6',
                 entry_date=datetime.date.today(), dob=None, gender='male', tag=None):

        self.id = mouse_id
        self.strain = strain
        self.dob = dob
        self.entry_date = entry_date
        self.gender = gender
        self.tag = tag