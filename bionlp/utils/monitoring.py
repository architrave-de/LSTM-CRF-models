"""
I wanted to create a Singleton for data storage, but several discussions in forums suggest that the
'module-approach' is more pythonic:
https://stackoverflow.com/a/31887/2191154
https://stackoverflow.com/a/6842257/2191154
"""

TYPE_VALIDATION = 'validation'
TYPE_TRAINING = 'training'

KEY_ACC = 'accuracy'
KEY_LOSS_TOT = 'loss_total'
KEY_LOSS_CRF = 'loss_crf'
KEY_LOSS_NET_CRF = 'loss_net_crf'

data = {}


def get_data_keys():
    return [
        KEY_ACC,
        KEY_LOSS_CRF,
        KEY_LOSS_NET_CRF,
        KEY_LOSS_TOT
    ]


class MonitoringDataObject(object):
    def __init__(self):
        super(MonitoringDataObject, self).__init__()
        self.__data = dict((key, []) for key in get_data_keys())

    def add_iteration_data(self, value_dict):
        assert set(value_dict.keys()).issubset(set(get_data_keys())), \
            "input data does not contain the necessary keys {}".format(get_data_keys())
        [self.__data[key].append(value) for key, value in value_dict.items()]

    def get_data(self):
        return self.__data


def get_init_data():
    return {
        TYPE_VALIDATION: MonitoringDataObject(),
        TYPE_TRAINING: MonitoringDataObject()
    }


def add_iteration_data(data_type, data_dict):
    assert data_type in data, "provided data type {0} not available".format(data_type)
    data[data_type].add_iteration_data(data_dict)


def get_data():
    return {data_type: data_object.get_data() for data_type, data_object in data.items()}
