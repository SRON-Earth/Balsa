import configparser
import pathlib

class Configuration:

	def __init__(self):

		self.cache_dir = pathlib.Path("cache")
		self.run_dir = pathlib.Path("run")
		self.classifiers = {}

	def add_classifier(self, name, *, driver, **kwargs):

		self.classifiers[name] = {"driver": driver, "args": dict(**kwargs)}

	def get_classifier(self, name):

		return self.classifiers[name]

def load_config(filename):

	parser = configparser.ConfigParser()

	with open(filename) as config_file:
		parser.read_file(config_file)

	config = Configuration()

	for section_name in parser.sections():

		section = parser[section_name]

		if section_name == "classifiert":
			config.cache_dir = pathlib.Path(section.get("cache_dir", "cache"))
			config.run_dir = pathlib.Path(section.get("run_dir", "run"))
		else:
			arguments = dict(section.items())
			driver = arguments.pop("driver")
			config.add_classifier(section_name, driver=driver, **arguments)

	return config

def store_config(filename, config):

	parser = configparser.ConfigParser()

	parser["classifiert"] = {
		"cache_dir": config.cache_dir,
		"run_dir": config.run_dir
	}

	for name, descriptor in config.classifiers.items():
		parser[name] = {"driver": descriptor["driver"]} | descriptor["args"]

	with open(filename, "w") as config_file:
		parser.write(config_file)
