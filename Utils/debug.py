# ----------------------------------------------------------------------------------------------------------------------------------------

import sys
from typing import Optional
import logging

# ----------------------------------------------------------------------------------------------------------------------------------------

class Logger:

	"""
	
	Class for logging settings.
	Example of usage:

	import logging

	# Init logger
	logger = Logger(
	    log_file='app.log',
	    logger_name='my_app',
	    level_console=logging.INFO,
	    level_file=logging.DEBUG,
	    fmt_console='%(asctime)s - %(levelname)s - %(message)s',
	    fmt_file='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	).get_logger()

	# Usage
	logger.debug("Debug message")
	logger.info("Info message")
	
	"""

	def __init__(self,
				 log_file: str,
				 logger_name: Optional[str] = None,
				 level_console: int = logging.INFO,
				 level_file: int = logging.DEBUG,
				 fmt_console: Optional[str] = None,
				 fmt_file: Optional[str] = None,
				 datefmt: str = '%Y-%m-%d %H:%M:%S',
				 clear_handlers: bool = False):

		self.log_file = log_file
		self.logger_name = logger_name
		self.level_console = level_console
		self.level_file = level_file
		self.fmt_console = fmt_console or '%(asctime)s - %(levelname)s - %(message)s'
		self.fmt_file = fmt_file or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		self.datefmt = datefmt
		self.clear_handlers = clear_handlers
		self._logger = None

	def setup(self) -> logging.Logger:
		"""
		Setup and return logger with handlers.
		"""
		logger = logging.getLogger(self.logger_name)

		if self.clear_handlers:
			logger.handlers.clear()

		if not logger.handlers:
			logger.setLevel(logging.DEBUG)
			logger.addHandler(self._get_console_handler())
			logger.addHandler(self._get_file_handler())

		self._logger = logger
		return logger

	def _get_console_handler(self) -> logging.Handler:
		"""
		Create console logger.
		"""
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(self.level_console)
		console_formatter = logging.Formatter(
			fmt=self.fmt_console,
			datefmt=self.datefmt
		)
		
		console_handler.setFormatter(console_formatter)
		return console_handler

	def _get_file_handler(self) -> logging.Handler:
		"""
		Create file logger.
		"""
		file_handler = logging.FileHandler(
			filename=self.log_file,
			mode='a',
			encoding='utf-8'
		)

		file_handler.setLevel(self.level_file)
		file_formatter = logging.Formatter(
			fmt=self.fmt_file,
			datefmt=self.datefmt
		)

		file_handler.setFormatter(file_formatter)
		return file_handler

	def get_logger(self) -> logging.Logger:
		"""
		Return setuped logger.
		"""

		if self._logger is None:
			return self.setup()
		
		return self._logger

# ----------------------------------------------------------------------------------------------------------------------------------------
