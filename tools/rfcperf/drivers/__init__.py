# Import all available drivers.
from . import balsa
from . import lightgbm
from . import ranger
from . import sklearn

_DRIVERS = {
	"balsa": balsa.Driver,
	"lightgbm": lightgbm.Driver,
	"ranger": ranger.Driver,
	"sklearn": sklearn.Driver
}

def get_drivers():

	return [name for name in _DRIVERS.keys()]

def get_driver(name):

	return _DRIVERS[name]
