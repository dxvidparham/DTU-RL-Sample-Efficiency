"""
This is a class for some helper methods to make logging more nice.
"""

import logging
import sys

colors = {
    "RESET": '\x1B[0m',
    "RED": '\x1B[31m',
    "YELLOW": "\x1B[33m",
    "BRGREEN": '\x1B[01;32m',
    "DARKGREEN": '\x1B[0;32m',
    "BOLD": '\x1b[5m',

    "BRIGHTGREY": "\x1b[1;37m",
    "DARKGREY": "\x1b[0;37m",

    "DARKBLUE": '\x1B[0;36m',
    "BRIGHTBLUE": '\x1B[1;36m'
}


def colored_log_text(txt, color):
    return colors.get(color) + txt + colors.get('RESET')


def print_dict(dictionary: dict, header: str):
    """
    Method to print a dict in nice.
    :param dictionary:
    :param header:
    :return:
    """
    logging.info(colored_log_text(f"\n\n ---{header.ljust(70, '-')}", "DARKBLUE") + \
                 '\n' + '\n'.join((colored_log_text('{}', 'DARKBLUE') + '\t' +
                                   (colored_log_text('{}', 'BRIGHTBLUE'))).format(k.ljust(20).upper(), v) for k, v in
                                  dictionary.items()) + \
                 colored_log_text(f"\n ---{''.ljust(70, '-')}\n\n", "DARKBLUE"))


def print_step_log(text: str, log_level: str = "INFO") -> None:
    """
    Helper Method to plot a step.
    :param text:
    :param log_level:
    :return:
    """
    _level = logging.getLevelName(log_level.upper())
    logging.log(_level, colored_log_text(f"{text} ...", "DARKBLUE"))
    logging.log(_level, colored_log_text("-------------------------------------------", "DARKBLUE"))


def print_big_log(text: str, log_level: str = "INFO") -> None:
    """
    A method to log headlines.
    :param text: Text to print
    :return:
    """
    _level = logging.getLevelName(log_level.upper())
    logging.log(_level, colored_log_text("#########################################", "DARKBLUE"))
    logging.log(_level, colored_log_text(f"### {text}", "DARKBLUE"))
    logging.log(_level, colored_log_text("########################################", "DARKBLUE"))


class ColouredFormatter(logging.Formatter):
    def format(self, record, colour=False):
        message = super().format(record)

        if not colour:
            return message

        level_no = record.levelno
        if level_no >= logging.CRITICAL:
            colour = colors.get("RED")
        elif level_no >= logging.ERROR:
            colour = colors.get("RED")
        elif level_no >= logging.WARNING:
            colour = colors.get("YELLOW")
        elif level_no >= logging.INFO:
            colour = colors.get("BRIGHTGREY")
        elif level_no >= logging.DEBUG:
            colour = colors.get("DARKGREY")
        else:
            colour = colors.get("RESET")

        message = colour + message + colors.get("RESET")

        return message


class ColouredHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)

    def format(self, record, colour=False):
        if not isinstance(self.formatter, ColouredFormatter):
            self.formatter = ColouredFormatter()

        return self.formatter.format(record, colour)

    def emit(self, record):
        stream = self.stream
        try:
            msg = self.format(record, stream.isatty())
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
